from typing import Dict, List, Type

from detectors.base import BaseFaceDetector
from detectors.haar import HaarCascadeDetector
from detectors.mediapipe_detector import MediaPipeDetector
from detectors.yolo_detector import YOLODetector


# Single source of truth for which detectors exist in the service.
# Adding a new one only takes: write the class, import it, add it here.
DETECTOR_REGISTRY: Dict[str, Type[BaseFaceDetector]] = {
    "haar": HaarCascadeDetector,
    "mediapipe": MediaPipeDetector,
    "yolo": YOLODetector,
}


def create_detector(name: str) -> BaseFaceDetector:
    # Build a detector instance from its registered name.
    cls = DETECTOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown detector '{name}'. Available: {available_detectors()}"
        )
    return cls()


def available_detectors() -> List[str]:
    return list(DETECTOR_REGISTRY.keys())
