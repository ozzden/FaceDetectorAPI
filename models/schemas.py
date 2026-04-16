from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DetectorName(str, Enum):
    HAAR = "haar"
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"


# ---------------------------------------------------------------------------
# Core detection result (passed between detectors and API layer)
# ---------------------------------------------------------------------------


@dataclass
class FaceDetection:
    """Single face detection result returned by every detector."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None  # list of (x, y) points
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        return {
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "confidence": round(self.confidence, 4),
            "landmarks": (
                [{"x": lm[0], "y": lm[1]} for lm in self.landmarks]
                if self.landmarks
                else None
            ),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Landmark(BaseModel):
    x: int
    y: int


class FaceDetectionResponse(BaseModel):
    bbox: BoundingBox
    confidence: float
    landmarks: Optional[List[Landmark]] = None
    metadata: Optional[Dict[str, Any]] = None


class DetectResponse(BaseModel):
    request_id: str
    detector: str
    face_count: int
    faces: List[FaceDetectionResponse]
    processing_time_ms: float
    image_width: int
    image_height: int
    annotated_image_base64: Optional[str] = None


class DetectBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image (with or without data URI prefix)")
    detector: str = Field("haar", description="Detector name: haar, mediapipe, yolo")
    user_id: Optional[str] = Field(None, description="Caller identifier")
    return_annotated: bool = Field(False, description="Include annotated image in response")
    min_confidence: float = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_faces: Optional[int] = Field(None, ge=1, description="Maximum faces to return")
    log_request: bool = Field(True, description="Persist this request to the SQLite log")


class BenchmarkResult(BaseModel):
    detector: str
    face_count: int
    processing_time_ms: float
    error: Optional[str] = None


class BenchmarkResponse(BaseModel):
    request_id: str
    image_width: int
    image_height: int
    results: List[BenchmarkResult]


class HealthResponse(BaseModel):
    status: str
    detectors_loaded: List[str]
    version: str = "1.0.0"
