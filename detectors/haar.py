from typing import List

import cv2
import numpy as np

from detectors.base import BaseFaceDetector
from detectors.detector_utils import clamp_bbox
from models.schemas import FaceDetection


# Classic OpenCV Haar cascade. Fast on CPU, not great at non-frontal faces
# but kept as a lightweight baseline.
class HaarCascadeDetector(BaseFaceDetector):

    # Haar doesn't give a real probability — cap the pseudo-score at 0.9
    # so users don't see confusing 100% confidence numbers.
    MAX_CONF = 0.90

    def __init__(self, scale_factor=1.1, min_neighbors=6, min_size=(30, 30)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors   # higher = less false positives
        self.min_size = min_size
        self.classifier = None

    @property
    def name(self) -> str:
        return "haar"

    def load(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(path)
        if self.classifier.empty():
            raise RuntimeError(f"Could not load haar cascade from {path}")

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        self._require_loaded(self.classifier, "classifier")

        # Haar runs on grayscale. equalizeHist helps a lot on dim images.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = image.shape[:2]

        faces, confs = self._run(gray)

        results = []
        for (x, y, fw, fh), c in zip(faces, confs):
            results.append(FaceDetection(
                bbox=clamp_bbox(x, y, x + fw, y + fh, w, h),
                confidence=float(c),
                landmarks=None,
                metadata={
                    "confidence_type": "proxy",  # not a real probability
                    "detector_params": {
                        "scale_factor": self.scale_factor,
                        "min_neighbors": self.min_neighbors,
                    },
                },
            ))
        return results

    def _run(self, gray):
        # Try the extended API first (gives reject-levels we can turn into
        # a pseudo confidence). Fall back to the plain call if not available.
        try:
            faces, _, weights = self.classifier.detectMultiScale3(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                outputRejectLevels=True,
            )
            if len(faces) == 0:
                return [], []
            raw = [float(w) for w in weights]
            max_w = max(raw) or 1.0
            # Normalise to [0, MAX_CONF] — for ranking only.
            confs = [min(v / max_w, 1.0) * self.MAX_CONF for v in raw]
            return list(faces), confs
        except cv2.error:
            pass

        # Fallback path: no confidence info, just return 1.0 for everyone
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        if len(faces) == 0:
            return [], []
        return list(faces), [1.0] * len(faces)
