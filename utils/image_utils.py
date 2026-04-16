import base64
import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from models.schemas import FaceDetection


# Hard upper bound
MAX_DIMENSION = 4096

# Resize target before inference
INFERENCE_MAX_DIM = 1280


def bytes_to_image(data: bytes) -> np.ndarray:
    # Decode any image bytes into an RGB numpy array. Using PIL here
    # (instead of cv2.imdecode) gives us consistent channel order for
    # every format PIL supports.
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.ascontiguousarray(np.array(image))


def decode_base64_image(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64_string)
    except Exception as exc:
        raise ValueError(f"Invalid base64 data: {exc}") from exc
    return bytes_to_image(raw)


def encode_image_base64(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_image_metadata(image: np.ndarray, content_type: str = "unknown") -> dict:
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    return {"width": w, "height": h, "channels": channels, "format": content_type}


def validate_image(image: np.ndarray, max_dim: int = MAX_DIMENSION) -> None:
    # Reject oversized or weirdly-shaped inputs before they hit inference.
    h, w = image.shape[:2]
    if h > max_dim or w > max_dim:
        raise ValueError(
            f"Image {w}×{h} px exceeds the {max_dim} px limit per dimension. "
            "Resize the image before uploading."
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel (RGB) image.")


def smart_resize(
    image: np.ndarray, max_dim: int = INFERENCE_MAX_DIM
) -> Tuple[np.ndarray, float]:
    # Shrink the image so the longest side is at most max_dim.
    # Returns (image, scale) — scale < 1 means we actually resized.
    # If the image already fits, we return it as-is (no copy).
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0

    scale = max_dim / longest
    new_w, new_h = int(w * scale), int(h * scale)
    # INTER_AREA is the standard choice for downscaling — fewer aliasing artifacts.
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def annotate_image(image: np.ndarray, detections: List[FaceDetection]) -> np.ndarray:
    # Draw green bboxes + "#N confidence%" labels + red landmark dots.
    # Used for the "Annotated" preview on the detection page.
    pil = Image.fromarray(image)
    out = pil.copy()
    draw = ImageDraw.Draw(out)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        label = f"#{i+1} {det.confidence*100:.0f}%"

        draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=2)

        # Put the label just above the box. If the box is near the top
        # edge, clamp to 0 so the label doesn't go off-screen.
        text_y = max(0, y1 - 16)
        # rough text width — good enough for a label background
        tw = len(label) * 7
        draw.rectangle([x1, text_y, x1 + tw, text_y + 14], fill=(0, 220, 0))
        draw.text((x1 + 2, text_y), label, fill=(0, 0, 0))

        if det.landmarks:
            for lx, ly in det.landmarks:
                draw.ellipse([lx - 3, ly - 3, lx + 3, ly + 3], fill=(255, 50, 50))

    return np.array(out)
