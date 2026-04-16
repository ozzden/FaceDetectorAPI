import os
from pathlib import Path
from typing import List

import numpy as np

from detectors.base import BaseFaceDetector
from detectors.detector_utils import LockedInferenceMixin, clamp_bbox, download_if_missing
from models.schemas import FaceDetection


# YOLOv8-face nano model. Heavier than Haar/MediaPipe but usually the most
# accurate of the three, and gives 5-point landmarks.
# TODO: try yolov8s-face later, might be worth the extra latency.
DEFAULT_MODEL = "yolov8n-face.pt"
MODEL_URL = "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt"
LANDMARK_NAMES = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]


class YOLODetector(LockedInferenceMixin, BaseFaceDetector):

    def __init__(self, confidence_threshold=0.25):
        super().__init__()
        self.model_path = os.environ.get("YOLO_MODEL_PATH", DEFAULT_MODEL)
        self.conf_threshold = confidence_threshold
        self.model = None

    @property
    def name(self) -> str:
        return "yolo"

    def load(self) -> None:
        from ultralytics import YOLO

        download_if_missing(MODEL_URL, self.model_path, label="YOLO")
        self.model = YOLO(self.model_path)

        # warm-up call — first inference is always slower because of
        # lazy init inside ultralytics. Running a dummy image once here
        # means real requests get a clean latency number.
        self.model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        self._require_loaded(self.model, "model")

        h, w = image.shape[:2]

        # ultralytics is not thread-safe, lock around the call
        with self._infer_lock:
            results = self.model(image, conf=self.conf_threshold, verbose=False)

        detections: List[FaceDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            kps = result.keypoints

            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # landmarks are optional (model variant-dependent)
                landmarks = None
                if kps is not None and i < len(kps.xy):
                    # drop (0,0) points — yolov8-face puts placeholders there
                    # when a keypoint is missing
                    pts = [(int(kp[0]), int(kp[1])) for kp in kps.xy[i].tolist()
                           if kp[0] > 0 or kp[1] > 0]
                    landmarks = pts or None

                detections.append(FaceDetection(
                    bbox=clamp_bbox(x1, y1, x2, y2, w, h),
                    confidence=float(box.conf[0]),
                    landmarks=landmarks,
                    metadata={
                        "model": Path(self.model_path).name,
                        "landmark_names": LANDMARK_NAMES[:len(landmarks)] if landmarks else [],
                    },
                ))

        return detections
