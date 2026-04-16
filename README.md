# Face Detection Service

## Overview

A small REST service that runs face detection with three interchangeable backends: OpenCV Haar Cascade, MediaPipe BlazeFace, and YOLOv8-face. Pick the detector per request, get back bounding boxes, confidence scores, and landmarks. Every request is logged to a rotating JSON file and a SQLite database, and there's a small web UI for uploading images, trying the live camera, and browsing the logs.

## Detectors

| Name | Landmarks | Notes |
|------|-----------|-------|
| `haar` | None | OpenCV, fast, CPU. Confidence is a normalised proxy, not a real probability. |
| `mediapipe` | 6 points | BlazeFace short-range via MediaPipe Tasks. |
| `yolo` | 5 points | YOLOv8-nano face, downloads weights on first run (~6 MB). |

## Installation

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the API

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

API: `http://localhost:8000`  |  Swagger: `http://localhost:8000/docs`  |  UI: `http://localhost:8000/ui`

### Docker

[Download the Docker image here](https://drive.google.com/file/d/1xmsGmvc6FUGP97TKlJ27qcDqZA-YRD04/view?usp=sharing) (2.24GB)

```bash
docker load -i face-detection.tar
docker run -d -p 8000:8000 --name face-detection-api face-detection
```

After that, the API is available at `http://localhost:8000`.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | Service + loaded detectors |
| `POST` | `/detect` | Detect on an uploaded file |
| `POST` | `/detect/base64` | Detect on a base64 image (JSON) |
| `POST` | `/detect/batch` | Multiple images in one request |
| `POST` | `/benchmark` | Run every detector on the same image |

## Example curl request

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@face.jpg" \
  -F "detector=yolo" \
  -F "min_confidence=0.3" \
  -H "X-User-Id: user-123"
```

## Example response

```json
{
  "request_id": "3f4a1b2c-9d8e-4f7a-b6c5-1234567890ab",
  "detector": "yolo",
  "face_count": 1,
  "faces": [
    {
      "bbox": {"x1": 120, "y1": 80, "x2": 280, "y2": 300},
      "confidence": 0.92,
      "landmarks": [
        {"x": 165, "y": 140}, {"x": 235, "y": 140},
        {"x": 200, "y": 180}, {"x": 170, "y": 230}, {"x": 230, "y": 230}
      ],
      "metadata": {
        "model": "yolov8n-face.pt",
        "landmark_names": ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
      }
    }
  ],
  "processing_time_ms": 38.71,
  "image_width": 640,
  "image_height": 480,
  "annotated_image_base64": null
}
```

## How to switch detectors

Pass the `detector` field on any detection endpoint:

| Detector | Value |
|----------|-------|
| Haar Cascade | `detector=haar` |
| MediaPipe | `detector=mediapipe` |
| YOLOv8-face | `detector=yolo` |

Same for the JSON endpoint:

```bash
curl -X POST http://localhost:8000/detect/base64 \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "...", "detector": "mediapipe"}'
```

## Logging

- `logs/app.log` stores rotating JSON lines (10 MB, 5 backups)
- `logs/requests.db` is a SQLite database, viewable at `/ui/logs`

Fields: `request_id`, `user_id`, `timestamp`, `detector`, image size + format, `face_count`, `processing_time_ms`, `status`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOG_DIR` | `./logs` | Where logs and SQLite live |
| `YOLO_MODEL_PATH` | `yolov8n-face.pt` | Skip the first-run download |
| `MEDIAPIPE_MODEL_PATH` | `blaze_face_short_range.tflite` | Same, for MediaPipe |

## Tests

```bash
pytest tests/ -v
```

## Assumptions

- Images are decoded with Pillow and handed to every detector as an RGB numpy array. This avoids BGR/RGB mismatches between OpenCV and the ML models.
- MediaPipe and YOLO weights are downloaded on the first run; set `YOLO_MODEL_PATH` / `MEDIAPIPE_MODEL_PATH` for air-gapped setups.
- Each uvicorn worker loads its own detector instances, so there is no shared state between workers.
- Only request metadata is logged, never the uploaded image itself.
- Input images are assumed to contain standard photographic content (portraits, group photos). **The detectors are not tuned for edge cases**
- All three detectors are optimised for frontal and near-frontal faces. Profile or extreme-angle faces will likely be missed, especially by Haar.

## Limitations

- Haar is noticeably less accurate than MediaPipe/YOLO, especially for non-frontal faces. It also struggles in low-light conditions even with histogram equalisation.
- MediaPipe and YOLO aren't thread safe, so within one worker their inference calls are serialised with a lock. Throughput scales with more uvicorn workers, not with threads.
- Images are capped at 4096 px per side and resized to 1280 px before inference. Very small faces in huge photos may be missed after the resize.
- Everything runs on CPU. There's no GPU path wired up right now.
- No face tracking across frames so live camera mode runs independent detections per frame, so there's no temporal smoothing or ID assignment.
- Batch endpoint fails entirely if any single image in the batch fails, rather than returning partial results.


### Other frameworks considered

- dlib (HOG + CNN): Solid accuracy, but the CNN model is slow on CPU and the Python bindings require cmake/boost to build. M1 Chip macbooks have version issues a lot(i use it thats why not preffered).

## Architectural Decisions

Three goals shaped how the code is organized: keep the detectors **swappable**, make the service **safe under parallel requests**, and keep every piece **simple and focused**.

### 1. Strategy pattern for detectors

Every detector (Haar, MediaPipe, YOLO) follows the same interface: `load()`, `detect(image)`, and `cleanup()`. The rest of the app doesn't know or care which detector it's talking to. It just calls these three methods. If we want to add a fourth detector, we just write one file, add one line to the registry. Nothing else changes.

### 2. Registry

There's a simple dictionary in `detectors/registry.py` that maps names like `"haar"` or `"yolo"` to their classes. When the app starts, it loops through this dictionary and creates one instance of each. This is the only place that knows which detectors exist.

### 3. Models load once at startup

Instead of loading model weights on every request (which would add 50 to 500 ms each time), we load everything once when the server starts using FastAPI's `lifespan` hook. If a detector fails to load (for example, YOLO can't download its weights), the service still starts with whatever detectors did load. The `/health` endpoint shows which ones are available.

### 4. Handling parallel requests

The server is async, face detection itself is heavy CPU work. We use `run_in_threadpool` to run detection in a separate thread so the server doesn't freeze while processing one image.

The catch: MediaPipe and YOLO keep internal state, so two threads can't call them at the same time safely. We put a lock on each detector instance so only one thread enters at a time.

| Detector | Safe to call from multiple threads? | How we handle it |
|----------|--------------------------------------|-----------------|
| Haar | Yes | No lock needed |
| MediaPipe | No | Lock per instance |
| YOLOv8 | No | Lock per instance |

To actually run detections in parallel, run more server workers (`uvicorn --workers 4`). Each worker gets its own detector instances and its own lock, so they don't share anything.

### 5. Internal data vs API data

Inside the pipeline, detections are plain Python `dataclass` objects, which are lightweight and have no validation overhead. At the API boundary (where data goes out to the user), we convert them to Pydantic models which handle validation, type checking, and auto-generate the Swagger docs. This way the detectors don't depend on the web framework at all.

### 6. Logging

Every request goes to two places:

1. `logs/app.log` which stores one JSON line per request, easy to search or pipe to a log tool.
2. `logs/requests.db` which is a SQLite database, easy to query and powers the `/ui/logs` page.

The logger is a singleton with a lock on database writes, so multiple threads don't corrupt the SQLite file.

### 7. Image handling

All images come in through Pillow and get converted to RGB numpy arrays. This avoids the BGR/RGB confusion that OpenCV would introduce. Haar then converts to grayscale internally; MediaPipe and YOLO take RGB directly.

Big images (over 4096 px) are rejected outright. Anything above 1280 px on the longest side gets resized before detection to keep inference fast. The bounding boxes are scaled back to the original size afterwards.

### 8. API design

Two ways to send an image, same pipeline behind both:

- `POST /detect` to upload a file (like a browser form or curl).
- `POST /detect/base64` to send base64 in JSON (like the live camera page does, or another service calling ours).

Both go through the same `run_detection()` function, so the filtering, logging, and response format are identical regardless of how the image arrived.

### Trade-offs

- **One detection at a time per worker.** The lock means requests queue up inside a single worker. More workers = more parallel detections, but also more memory (each worker loads its own copy of the models).
- **Haar confidence isn't real.** It's a rough score derived from OpenCV internals. Good enough for sorting results, but don't treat it like a probability.
- **Model download on first run.** YOLO and MediaPipe weights are pulled from the internet the first time. After that they're cached locally (or baked into the Docker image).
