"""Shared pytest fixtures for the face detection service tests."""
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Synthetic test image
# ---------------------------------------------------------------------------


def _make_face_image(width: int = 200, height: int = 200) -> np.ndarray:
    """Return a synthetic RGB image with a drawn face-like oval."""
    img = Image.new("RGB", (width, height), color=(200, 180, 160))
    draw = ImageDraw.Draw(img)
    # Face oval
    draw.ellipse([60, 40, 140, 160], fill=(240, 200, 170), outline=(100, 80, 60), width=2)
    # Eyes
    draw.ellipse([75, 75, 90, 88], fill=(60, 40, 30))
    draw.ellipse([110, 75, 125, 88], fill=(60, 40, 30))
    # Nose
    draw.polygon([(100, 95), (93, 115), (107, 115)], fill=(200, 150, 130))
    # Mouth
    draw.arc([85, 120, 115, 140], start=0, end=180, fill=(150, 80, 80), width=2)
    return np.array(img)


@pytest.fixture(scope="session")
def test_image_rgb() -> np.ndarray:
    return _make_face_image()


@pytest.fixture(scope="session")
def test_image_bytes(test_image_rgb: np.ndarray) -> bytes:
    pil = Image.fromarray(test_image_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def client() -> TestClient:
    from main import app

    with TestClient(app) as c:
        yield c
