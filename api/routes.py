import asyncio
import time
import uuid
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from detectors.detector_utils import filter_detections, scale_detections
from models.schemas import (
    BenchmarkResponse,
    BenchmarkResult,
    BoundingBox,
    DetectBase64Request,
    DetectResponse,
    FaceDetectionResponse,
    HealthResponse,
    Landmark,
)
from utils.image_utils import (
    annotate_image,
    bytes_to_image,
    decode_base64_image,
    encode_image_base64,
    get_image_metadata,
    smart_resize,
    validate_image,
)
from utils.logger import RequestLogger, setup_logger

logger = setup_logger()
router = APIRouter()


async def run_detection(
    *,
    image: np.ndarray,
    detector_name: str,
    user_id: Optional[str],
    return_annotated: bool,
    min_confidence: float,
    max_faces: Optional[int],
    request: Request,
    request_id: str,
    content_type: str = "unknown",
    log_request: bool = True,
) -> DetectResponse:
    """Shared detection pipeline used by every endpoint.

    Steps: check inputs, resize if needed, run the detector, filter the
    results, log the request, and build the response.
    """
    detectors = request.app.state.detectors
    req_logger = RequestLogger()

    # Make sure the requested detector actually loaded at startup.
    if detector_name not in detectors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_detector",
                "message": f"Detector '{detector_name}' is not loaded.",
                "loaded": list(detectors.keys()),
            },
        )

    # Refuse images that are too big before we even touch the detector.
    try:
        validate_image(image)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail={"error": "image_too_large", "message": str(exc)})

    image_meta = get_image_metadata(image, content_type)

    # Big phone photos are slow to process — shrink them first, then
    # remember the scale so we can map the bboxes back to the original.
    # TODO: let each detector pick its own target size; YOLO likes bigger inputs.
    infer_image, scale = smart_resize(image)
    if scale < 1.0:
        logger.info(
            "Image resized for inference",
            extra={"extra": {
                "request_id": request_id,
                "original": f"{image_meta['width']}×{image_meta['height']}",
                "resized": f"{infer_image.shape[1]}×{infer_image.shape[0]}",
                "scale": round(scale, 3),
            }},
        )

    # The actual face detection call. We push it to a thread so the
    # server can still handle other requests while this one runs.
    detector = detectors[detector_name]
    start = time.perf_counter()
    try:
        raw_faces = await run_in_threadpool(detector.detect, infer_image)
    except MemoryError:
        raise HTTPException(
            status_code=507,
            detail={"error": "out_of_memory", "message": "Image too large for available memory."},
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        if log_request:
            req_logger.log_request(
                request_id=request_id, user_id=user_id, detector=detector_name,
                image_metadata=image_meta, face_count=0,
                processing_time_ms=elapsed_ms, status="error", error_message=str(exc),
            )
        logger.error("Inference failed", extra={"extra": {"request_id": request_id, "error": str(exc)}})
        raise HTTPException(
            status_code=500,
            detail={"error": "inference_failed", "message": str(exc)},
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    # If we shrank the image earlier, scale the boxes back up.
    if scale < 1.0:
        raw_faces = scale_detections(raw_faces, scale)

    # Drop low-confidence detections, apply NMS, cap at max_faces.
    faces = filter_detections(
        raw_faces,
        min_confidence=min_confidence,
        max_faces=max_faces,
        image_shape=image.shape[:2],
    )

    # Save the request to the log. Live camera skips this so we don't
    # fill the database with 2 fps of almost-identical frames.
    if log_request:
        req_logger.log_request(
            request_id=request_id, user_id=user_id, detector=detector_name,
            image_metadata=image_meta, face_count=len(faces),
            processing_time_ms=elapsed_ms,
        )

    # Turn our internal face objects into the API response format.
    face_responses: List[FaceDetectionResponse] = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        lms = [Landmark(x=lx, y=ly) for lx, ly in face.landmarks] if face.landmarks else None
        face_responses.append(FaceDetectionResponse(
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            confidence=round(face.confidence, 4),
            landmarks=lms,
            metadata=face.metadata,
        ))

    # Drawing boxes on the image is optional — only do it if the caller asked.
    annotated_b64: Optional[str] = None
    if return_annotated:
        annotated = await run_in_threadpool(annotate_image, image, faces)
        annotated_b64 = encode_image_base64(annotated)

    return DetectResponse(
        request_id=request_id,
        detector=detector_name,
        face_count=len(faces),
        faces=face_responses,
        processing_time_ms=round(elapsed_ms, 2),
        image_width=image_meta["width"],
        image_height=image_meta["height"],
        annotated_image_base64=annotated_b64,
    )


# -------- Endpoints ----------

@router.get("/health", response_model=HealthResponse, tags=["utility"])
async def health_check(request: Request) -> HealthResponse:
    """Simple health check — returns which detectors are loaded."""
    return HealthResponse(
        status="healthy",
        detectors_loaded=list(request.app.state.detectors.keys()),
    )


@router.post("/detect", response_model=DetectResponse, tags=["detection"])
async def detect_file(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, WEBP …)"),
    detector: str = Form("haar", description="haar | mediapipe | yolo"),
    user_id: Optional[str] = Form(None),
    return_annotated: bool = Form(False),
    min_confidence: float = Form(0.3, ge=0.0, le=1.0,
                                  description="Drop detections below this score"),
    max_faces: Optional[int] = Form(None, ge=1, description="Return at most N faces"),
    x_user_id: Optional[str] = Header(None),
) -> DetectResponse:
    """Detect faces in an uploaded image file (multipart/form-data)."""
    request_id = str(uuid.uuid4())

    # Skip anything that doesn't look like an image.
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail={"error": "unsupported_media_type",
                    "message": f"Expected an image file, got '{file.content_type}'."},
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_file", "message": "Uploaded file is empty."},
        )

    # Decode the bytes into a numpy array. If the file is corrupt or an
    # unsupported format, give the user a hint about what we accept.
    try:
        image = bytes_to_image(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "decode_failed",
                    "message": f"Cannot decode image: {exc}. "
                               "Ensure the file is a valid JPEG, PNG, BMP, or WEBP."},
        )

    return await run_detection(
        image=image,
        detector_name=detector,
        user_id=user_id or x_user_id,
        return_annotated=return_annotated,
        min_confidence=min_confidence,
        max_faces=max_faces,
        request=request,
        request_id=request_id,
        content_type=file.content_type or "unknown",
    )


@router.post("/detect/base64", response_model=DetectResponse, tags=["detection"])
async def detect_base64(
    request: Request,
    body: DetectBase64Request,
    x_user_id: Optional[str] = Header(None),
) -> DetectResponse:
    """Detect faces in a base64-encoded image (JSON body).

    Used by the live camera page and by anyone calling the service from
    code that prefers JSON over multipart uploads.
    """
    request_id = str(uuid.uuid4())

    try:
        image = decode_base64_image(body.image_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "decode_failed", "message": str(exc)},
        )

    return await run_detection(
        image=image,
        detector_name=body.detector,
        user_id=body.user_id or x_user_id,
        return_annotated=body.return_annotated,
        min_confidence=body.min_confidence,
        max_faces=body.max_faces,
        request=request,
        request_id=request_id,
        log_request=body.log_request,
    )


@router.post("/detect/batch", response_model=List[DetectResponse], tags=["detection"])
async def detect_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="One or more image files"),
    detector: str = Form("haar"),
    user_id: Optional[str] = Form(None),
    min_confidence: float = Form(0.3, ge=0.0, le=1.0),
    max_faces: Optional[int] = Form(None, ge=1),
    x_user_id: Optional[str] = Header(None),
) -> List[DetectResponse]:
    """Run detection on multiple images at once."""
    effective_user = user_id or x_user_id

    if not files:
        raise HTTPException(status_code=400, detail={"error": "no_files",
                                                      "message": "At least one file is required."})

    # Process each file the same way as /detect, just in parallel.
    async def _one(f: UploadFile) -> DetectResponse:
        rid = str(uuid.uuid4())
        raw = await f.read()
        if not raw:
            raise HTTPException(status_code=400,
                                detail={"error": "empty_file", "message": f"'{f.filename}' is empty."})
        try:
            image = bytes_to_image(raw)
        except Exception as exc:
            raise HTTPException(status_code=400,
                                detail={"error": "decode_failed", "message": str(exc)})
        return await run_detection(
            image=image, detector_name=detector, user_id=effective_user,
            return_annotated=False, min_confidence=min_confidence, max_faces=max_faces,
            request=request, request_id=rid, content_type=f.content_type or "unknown",
        )

    # TODO: if one image fails the whole batch fails. Returning per-file
    # errors would be nicer.
    return list(await asyncio.gather(*[_one(f) for f in files]))


@router.post("/benchmark", response_model=BenchmarkResponse, tags=["utility"])
async def benchmark(
    request: Request,
    file: UploadFile = File(...),
    min_confidence: float = Form(0.3, ge=0.0, le=1.0),
    x_user_id: Optional[str] = Header(None),
) -> BenchmarkResponse:
    """Run every detector on the same image and compare how long they take.

    Useful for seeing which model is faster/more accurate on a given photo.
    """
    raw = await file.read()
    try:
        image = bytes_to_image(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": "decode_failed", "message": str(exc)})

    try:
        validate_image(image)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail={"error": "image_too_large", "message": str(exc)})

    # Resize once so every detector sees the same input and the timings
    # are actually comparable.
    infer_image, scale = smart_resize(image)
    h, w = image.shape[:2]
    detectors = request.app.state.detectors
    request_id = str(uuid.uuid4())

    async def _bench_one(name: str) -> BenchmarkResult:
        detector = detectors[name]
        start = time.perf_counter()
        try:
            raw_faces = await run_in_threadpool(detector.detect, infer_image)
            faces = filter_detections(
                scale_detections(raw_faces, scale),
                min_confidence=min_confidence,
                image_shape=image.shape[:2],
            )
            return BenchmarkResult(
                detector=name,
                face_count=len(faces),
                processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            )
        except Exception as exc:
            # If one detector crashes, still return results for the others.
            return BenchmarkResult(
                detector=name, face_count=0,
                processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
                error=str(exc),
            )

    # TODO: add a warm-up run — the very first call is usually slower
    # than the rest, which skews the comparison.
    results = list(await asyncio.gather(*[_bench_one(n) for n in detectors]))

    # Log the benchmark as a single entry so it shows up alongside
    # regular requests in the logs page.
    RequestLogger().log_request(
        request_id=request_id, user_id=x_user_id, detector="benchmark",
        image_metadata=get_image_metadata(image, file.content_type or "unknown"),
        face_count=max((r.face_count for r in results), default=0),
        processing_time_ms=sum(r.processing_time_ms for r in results),
    )

    return BenchmarkResponse(request_id=request_id, image_width=w, image_height=h, results=results)
