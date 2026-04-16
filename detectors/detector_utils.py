import threading
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple


def clamp_bbox(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    # Clip the box to the image and cast to int.
    return (
        max(0, min(int(x1), img_w)),
        max(0, min(int(y1), img_h)),
        max(0, min(int(x2), img_w)),
        max(0, min(int(y2), img_h)),
    )


def iou_score(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    # Standard Intersection-over-Union between two (x1,y1,x2,y2) boxes.
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------- Post-processing (runs in the API layer, not inside detectors) ----------

def apply_nms(detections: list, iou_threshold: float = 0.45) -> list:
    # Non-Maximum Suppression: when two boxes overlap a lot, keep only the
    # one with the higher confidence.
    if len(detections) <= 1:
        return detections

    ranked = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list = []
    while ranked:
        best = ranked.pop(0)
        kept.append(best)
        ranked = [d for d in ranked if iou_score(best.bbox, d.bbox) < iou_threshold]
    return kept


def filter_detections(
    detections: list,
    *,
    min_confidence: float = 0.3,
    min_face_ratio: float = 0.005,
    max_aspect_ratio: float = 2.0,
    max_faces: Optional[int] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> list:
    # Cleanup pass applied to raw detector output:
    #   - drop low-confidence hits (min_confidence)
    #   - drop tiny specks (min_face_ratio — area vs. image area)
    #   - drop extreme rectangles (max_aspect_ratio — faces are roughly square)
    #   - run NMS to remove duplicates
    #   - keep the top-N most confident if max_faces is set
    img_area = (image_shape[0] * image_shape[1]) if image_shape else None
    filtered: list = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        w, h = x2 - x1, y2 - y1

        # Skip degenerate boxes (zero or negative size).
        if w <= 0 or h <= 0:
            continue

        if det.confidence < min_confidence:
            continue

        # Real faces are close to square; very long/thin boxes are usually noise.
        aspect = max(w, h) / min(w, h)
        if aspect > max_aspect_ratio:
            continue

        # Tiny boxes relative to the whole image are almost always false positives.
        if img_area:
            if (w * h) / img_area < min_face_ratio:
                continue

        filtered.append(det)

    filtered = apply_nms(filtered)

    if max_faces is not None:
        filtered = sorted(filtered, key=lambda d: d.confidence, reverse=True)[:max_faces]

    return filtered


def scale_detections(detections: list, scale: float) -> list:
    # Undo a resize: multiply boxes and landmarks by 1/scale so they line
    # up with the original image.
    if scale == 1.0:
        return detections

    inv = 1.0 / scale
    result = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        new_bbox = (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))
        new_lms = (
            [(int(lx * inv), int(ly * inv)) for lx, ly in det.landmarks]
            if det.landmarks
            else None
        )
        result.append(replace(det, bbox=new_bbox, landmarks=new_lms))
    return result


# ---------- Model download helpers ----------

def download_if_missing(url: str, path: str, label: str = "Model") -> None:
    # Download the weights on first run. Skipped if the file already exists.
    if Path(path).exists():
        return
    print(f"[{label}] Downloading → {path}")
    urllib.request.urlretrieve(url, path)
    print(f"[{label}] Download complete.")


def download_with_fallback(urls: List[str], path: str, label: str = "Model") -> None:
    # Same as above, but tries several mirrors in order until one works.
    # Useful when the primary host (HuggingFace, etc.) is temporarily down.
    if Path(path).exists():
        return

    errors = []
    for url in urls:
        try:
            print(f"[{label}] Downloading from {url} → {path}")
            urllib.request.urlretrieve(url, path)
            print(f"[{label}] Download complete.")
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            print(f"[{label}] Failed: {exc}, trying next mirror...")

    raise RuntimeError(
        f"[{label}] All download mirrors failed:\n" + "\n".join(errors)
    )


# ---------- Thread safety ----------

# MediaPipe and YOLO aren't safe to call from multiple threads at once.
# Detectors that inherit this mixin get a lock around their detect() call.
# Haar doesn't need it (OpenCV's detectMultiScale is stateless).
class LockedInferenceMixin:

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

    def __init__(self) -> None:
        self._infer_lock = threading.Lock()
