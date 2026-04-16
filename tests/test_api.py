"""Integration tests for the FastAPI endpoints."""
import base64
import io

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert isinstance(data["detectors_loaded"], list)
    assert len(data["detectors_loaded"]) > 0


def test_health_has_version(client):
    data = client.get("/health").json()
    assert "version" in data


# ---------------------------------------------------------------------------
# POST /detect  (file upload)
# ---------------------------------------------------------------------------


def test_detect_haar_file_upload(client, test_image_bytes):
    resp = client.post(
        "/detect",
        data={"detector": "haar"},
        files={"file": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detector"] == "haar"
    assert isinstance(body["face_count"], int)
    assert isinstance(body["faces"], list)
    assert body["processing_time_ms"] >= 0
    assert body["image_width"] == 200
    assert body["image_height"] == 200
    assert "request_id" in body


def test_detect_mediapipe_file_upload(client, test_image_bytes):
    resp = client.post(
        "/detect",
        data={"detector": "mediapipe"},
        files={"file": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200


def test_detect_yolo_file_upload(client, test_image_bytes):
    resp = client.post(
        "/detect",
        data={"detector": "yolo"},
        files={"file": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200


def test_detect_invalid_detector(client, test_image_bytes):
    resp = client.post(
        "/detect",
        data={"detector": "nonexistent"},
        files={"file": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 400


def test_detect_no_file_returns_422(client):
    resp = client.post("/detect", data={"detector": "haar"})
    assert resp.status_code == 422


def test_detect_annotated_image(client, test_image_bytes):
    resp = client.post(
        "/detect",
        data={"detector": "haar", "return_annotated": "true"},
        files={"file": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    # annotated_image_base64 is present (may be None if no faces found, but field exists)
    assert "annotated_image_base64" in body


def test_detect_user_id_header(client, test_image_bytes):
    resp = client.post(
        "/detect",
        data={"detector": "haar"},
        files={"file": ("test.png", test_image_bytes, "image/png")},
        headers={"X-User-Id": "test-user-42"},
    )
    assert resp.status_code == 200


def test_detect_response_bbox_fields(client, test_image_bytes):
    resp = client.post(
        "/detect",
        data={"detector": "haar"},
        files={"file": ("test.png", test_image_bytes, "image/png")},
    )
    body = resp.json()
    for face in body["faces"]:
        bbox = face["bbox"]
        for key in ("x1", "y1", "x2", "y2"):
            assert key in bbox
        assert face["confidence"] >= 0.0


# ---------------------------------------------------------------------------
# POST /detect/base64
# ---------------------------------------------------------------------------


def test_detect_base64_endpoint(client, test_image_bytes):
    b64 = base64.b64encode(test_image_bytes).decode()
    resp = client.post(
        "/detect/base64",
        json={"image_base64": b64, "detector": "haar"},
    )
    assert resp.status_code == 200
    assert resp.json()["detector"] == "haar"


def test_detect_base64_data_uri_prefix(client, test_image_bytes):
    b64 = "data:image/png;base64," + base64.b64encode(test_image_bytes).decode()
    resp = client.post(
        "/detect/base64",
        json={"image_base64": b64, "detector": "haar"},
    )
    assert resp.status_code == 200


def test_detect_base64_invalid_image(client):
    resp = client.post(
        "/detect/base64",
        json={"image_base64": "bm90YW5pbWFnZQ==", "detector": "haar"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /detect/batch
# ---------------------------------------------------------------------------


def test_detect_batch(client, test_image_bytes):
    resp = client.post(
        "/detect/batch",
        data={"detector": "haar"},
        files=[
            ("files", ("img1.png", test_image_bytes, "image/png")),
            ("files", ("img2.png", test_image_bytes, "image/png")),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2


# ---------------------------------------------------------------------------
# POST /benchmark
# ---------------------------------------------------------------------------


def test_benchmark_endpoint(client, test_image_bytes):
    resp = client.post(
        "/benchmark",
        files={"file": ("test.png", test_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    assert len(body["results"]) > 0
    for r in body["results"]:
        assert "detector" in r
        assert "processing_time_ms" in r
        assert "face_count" in r
