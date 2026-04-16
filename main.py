"""Face Detection Service — application entry point."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.ui import ui_router
from detectors.registry import DETECTOR_REGISTRY
from utils.logger import setup_logger

_logger = setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load all detectors at startup; clean up at shutdown."""
    app.state.detectors = {}

    for name, cls in DETECTOR_REGISTRY.items():
        try:
            detector = cls()
            detector.load()
            app.state.detectors[name] = detector
            _logger.info("Loaded detector", extra={"extra": {"detector": name}})
        except Exception as exc:
            _logger.warning(
                "Skipping detector — failed to load",
                extra={"extra": {"detector": name, "error": str(exc)}},
            )

    loaded = list(app.state.detectors.keys())
    _logger.info(
        "Face Detection Service ready",
        extra={"extra": {"loaded_detectors": loaded}},
    )

    yield

    for name, detector in app.state.detectors.items():
        try:
            detector.cleanup()
        except Exception:
            pass
    _logger.info("Face Detection Service shutdown complete")


app = FastAPI(
    title="Face Detection Service",
    description=(
        "Modular face detection service supporting Haar Cascade, MediaPipe, "
        "and YOLOv8-face detectors via a clean REST API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(ui_router)
app.mount("/samples", StaticFiles(directory="samples"), name="samples")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1, 
        reload=False,
        log_level="info",
    )
