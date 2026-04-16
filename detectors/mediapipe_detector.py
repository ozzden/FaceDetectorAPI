import os
from typing import List

import numpy as np

from detectors.base import BaseFaceDetector
from detectors.detector_utils import LockedInferenceMixin, clamp_bbox, download_if_missing
from models.schemas import FaceDetection


# BlazeFace short-range via MediaPipe Tasks API. Good balance of speed and
# accuracy, gives us 6 landmarks for free.
LANDMARK_NAMES = [
    "right_eye", "left_eye", "nose_tip",
    "mouth", "right_ear_tragion", "left_ear_tragion",
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
DEFAULT_MODEL = "blaze_face_short_range.tflite"


class MediaPipeDetector(LockedInferenceMixin, BaseFaceDetector):

    def __init__(self, min_detection_confidence=0.5):
        super().__init__()  # sets up _infer_lock
        self.min_conf = min_detection_confidence
        # allow overriding the model path via env var (useful in docker)
        self.model_path = os.environ.get("MEDIAPIPE_MODEL_PATH", DEFAULT_MODEL)
        self.detector = None

    @property
    def name(self) -> str:
        return "mediapipe"

    def load(self) -> None:
        # imports kept inside load() so the whole app doesn't crash if
        # mediapipe fails to import on a weird platform
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

        download_if_missing(MODEL_URL, self.model_path, label="MediaPipe")

        opts = FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
            min_detection_confidence=self.min_conf,
        )
        self.detector = FaceDetector.create_from_options(opts)

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        self._require_loaded(self.detector, "detector")

        import mediapipe as mp

        h, w = image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # mediapipe instance isn't thread-safe, serialise with the mixin lock
        with self._infer_lock:
            results = self.detector.detect(mp_image)

        if not results.detections:
            return []

        out: List[FaceDetection] = []
        for det in results.detections:
            bb = det.bounding_box
            bbox = clamp_bbox(
                bb.origin_x, bb.origin_y,
                bb.origin_x + bb.width, bb.origin_y + bb.height,
                w, h,
            )

            # categories list can technically be empty, fall back to 0
            conf = float(det.categories[0].score) if det.categories else 0.0

            # landmarks come normalised [0, 1] — scale to pixel coords
            landmarks = [(int(kp.x * w), int(kp.y * h)) for kp in det.keypoints]

            out.append(FaceDetection(
                bbox=bbox,
                confidence=conf,
                landmarks=landmarks or None,
                metadata={"landmark_names": LANDMARK_NAMES[:len(landmarks)]},
            ))

        return out

    def cleanup(self) -> None:
        # mediapipe detector holds some C++ resources
        if self.detector is not None:
            self.detector.close()
            self.detector = None
