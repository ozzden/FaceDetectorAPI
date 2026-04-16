from abc import ABC, abstractmethod
from typing import List

import numpy as np

from models.schemas import FaceDetection


# Every detector (Haar, MediaPipe, YOLO...) inherits from this class.
# As long as a new detector implements load() and detect(), the rest of
# the app works without any other change.
class BaseFaceDetector(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        # Short ID used in the API and registry, e.g. "haar".
        ...

    @abstractmethod
    def load(self) -> None:
        # Called once at startup — loads model weights, opens files, etc.
        # Should raise if the detector can't start so we can skip it.
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        # Runs face detection on an RGB image (H×W×3, uint8).
        # Returns an empty list if no faces are found.
        ...

    def cleanup(self) -> None:
        # Optional — override if the detector holds resources to release.
        pass

    def _require_loaded(self, obj: object, attr: str = "model") -> None:
        # Guards against calling detect() before load() — gives a clear error
        # instead of a cryptic AttributeError later on.
        if obj is None:
            raise RuntimeError(
                f"[{self.name}] '{attr}' is not initialised — call load() first."
            )
