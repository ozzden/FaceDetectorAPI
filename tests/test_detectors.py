"""Unit tests for individual face detector implementations."""
import numpy as np
import pytest

from models.schemas import FaceDetection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_valid_detection(det: FaceDetection, image: np.ndarray) -> None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = det.bbox
    assert isinstance(x1, int) and isinstance(y1, int)
    assert isinstance(x2, int) and isinstance(y2, int)
    assert x2 > x1 and y2 > y1, "bbox must have positive area"
    assert 0.0 <= det.confidence <= 1.0, f"confidence out of range: {det.confidence}"
    if det.landmarks:
        for lx, ly in det.landmarks:
            assert isinstance(lx, int) and isinstance(ly, int)


# ---------------------------------------------------------------------------
# Haar
# ---------------------------------------------------------------------------


class TestHaarCascadeDetector:
    def test_loads_without_error(self):
        from detectors.haar import HaarCascadeDetector
        d = HaarCascadeDetector()
        d.load()

    def test_detect_returns_list(self, test_image_rgb):
        from detectors.haar import HaarCascadeDetector
        d = HaarCascadeDetector()
        d.load()
        result = d.detect(test_image_rgb)
        assert isinstance(result, list)

    def test_detect_output_schema(self, test_image_rgb):
        from detectors.haar import HaarCascadeDetector
        d = HaarCascadeDetector()
        d.load()
        for det in d.detect(test_image_rgb):
            _assert_valid_detection(det, test_image_rgb)

    def test_name(self):
        from detectors.haar import HaarCascadeDetector
        assert HaarCascadeDetector().name == "haar"


# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------


class TestMediaPipeDetector:
    def test_loads_without_error(self):
        from detectors.mediapipe_detector import MediaPipeDetector
        d = MediaPipeDetector()
        d.load()
        d.cleanup()

    def test_detect_returns_list(self, test_image_rgb):
        from detectors.mediapipe_detector import MediaPipeDetector
        d = MediaPipeDetector()
        d.load()
        result = d.detect(test_image_rgb)
        assert isinstance(result, list)
        d.cleanup()

    def test_detect_output_schema(self, test_image_rgb):
        from detectors.mediapipe_detector import MediaPipeDetector
        d = MediaPipeDetector()
        d.load()
        for det in d.detect(test_image_rgb):
            _assert_valid_detection(det, test_image_rgb)
        d.cleanup()

    def test_name(self):
        from detectors.mediapipe_detector import MediaPipeDetector
        assert MediaPipeDetector().name == "mediapipe"


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------


class TestYOLODetector:
    def test_loads_without_error(self):
        from detectors.yolo_detector import YOLODetector
        d = YOLODetector()
        d.load()

    def test_detect_returns_list(self, test_image_rgb):
        from detectors.yolo_detector import YOLODetector
        d = YOLODetector()
        d.load()
        result = d.detect(test_image_rgb)
        assert isinstance(result, list)

    def test_detect_output_schema(self, test_image_rgb):
        from detectors.yolo_detector import YOLODetector
        d = YOLODetector()
        d.load()
        for det in d.detect(test_image_rgb):
            _assert_valid_detection(det, test_image_rgb)

    def test_name(self):
        from detectors.yolo_detector import YOLODetector
        assert YOLODetector().name == "yolo"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_all_detectors():
    from detectors.registry import DETECTOR_REGISTRY, available_detectors
    for name in ("haar", "mediapipe", "yolo"):
        assert name in DETECTOR_REGISTRY
    assert set(available_detectors()) == {"haar", "mediapipe", "yolo"}


def test_create_detector_unknown_raises():
    from detectors.registry import create_detector
    with pytest.raises(ValueError, match="Unknown detector"):
        create_detector("nonexistent")
