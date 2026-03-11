# Player Jersey Identification API

FastAPI micro-service that detects a target player in a video by jersey color and number.  
Returns timestamped detections for downstream integration.

## Architecture

```
Video → ffmpeg (2 FPS frames)
  → YOLOv8-seg person detection (yolo26n-seg.pt, batch)
  → HSV color-ratio filter (18-color range table)
  → Jersey number model on torso crops (jersey_number_yolo11m.pt, imgsz=320)
  → Position-prior scoring → confidence export filter
```

**Dual-model pipeline (`detection_first` strategy):**

| Model | File | Purpose |
|-------|------|---------|
| YOLOv8n-seg | `app/model/yolo26n-seg.pt` | Person instance segmentation (COCO) |
| YOLOv11m | `app/model/jersey_number_yolo11m.pt` | Jersey number recognition (classes 0-99) |

## Folder Structure

```text
app/
  model/
    jersey_number_yolo11m.pt   # jersey number model (0-99)
    yolo26n-seg.pt             # person segmentation model
  routes/
    detect.py                  # POST /detect endpoint
    health.py                  # GET /health endpoint
  schemas/
    detect.py                  # Pydantic request/response models
  services/
    detection_runtime.py       # PipelineSettings, dataclasses
    detection_detector.py      # YOLO inference wrappers
    detection_pipeline.py      # Main detection orchestrator
    detection_service.py       # Top-level entry point
  main.py                      # FastAPI app factory + lifespan
asgi.py                        # ASGI entry point for gunicorn/uvicorn
Dockerfile
requirements.txt
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

The repository includes the default jersey-number and person-detection weights under `app/model/`. You can still override them with `YOLO_MODEL_SOURCE` and `PERSON_MODEL_SOURCE` if needed.

## Run Locally

```bash
python -m uvicorn asgi:app --host 0.0.0.0 --port 8000
```

## API Contract

### Health

`GET /live`

```json
{ "status": "ok" }
```

`GET /ready` and `GET /health` report detector readiness. If required model assets or runtime binaries are missing, they return `503` with an error detail instead of reporting a false healthy state.

### Detect

`POST /detect`

Request:

```json
{
  "video_url": "https://www.youtube.com/watch?v=EXAMPLE",
  "jersey_number": 2,
  "jersey_color": "white",
  "sport": "basketball",
  "position": "guard"
}
```

Alternative source fields (mutually exclusive):
- `video_path` — local/server file path
- `video_bytes_b64` — base64-encoded video bytes

Response:

```json
[
  {
    "timestamp": 8.4,
    "confidence": 0.92,
    "bbox": {
      "x1": 340,
      "y1": 180,
      "x2": 490,
      "y2": 520,
      "x1_pct": 21.25,
      "y1_pct": 15.0,
      "x2_pct": 30.63,
      "y2_pct": 43.33
    }
  }
]
```

## Environment Variables

All settings can be tuned via env vars. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_SOURCE` | `app/model/jersey_number_yolo11m.pt` | Path to jersey number model |
| `PERSON_MODEL_SOURCE` | `app/model/yolo26n-seg.pt` | Path to person segmentation model |
| `DETECTION_STRATEGY` | `detection_first` | Pipeline strategy |
| `FPS` | `2` | Frames per second to sample |
| `CONF_THRESHOLD_EXPORT` | `0.55` | Minimum confidence for exported detections |
| `POSITION_PRIOR_WEIGHT` | `0.10` | Weight for position-based scoring |
| `YOUTUBE_CLIP_SECONDS` | _(unset)_ | Optional cap for YouTube downloads; leave unset for full videos |
| `DEBUG_VIDEO_PATH` | _(unset)_ | Set a path to write annotated debug video |

## Docker

CPU-friendly build:

```bash
docker build -t layer1-cv:latest .
```

GPU build:

```bash
docker build -f Dockerfile.gpu -t layer1-cv:gpu .
```

Run:

```bash
docker run --rm -p 8000:8000 layer1-cv:latest
```

Run on GPU host:

```bash
docker run --rm --gpus all -p 8000:8000 layer1-cv:gpu
```

Override settings:

```bash
docker run --rm -p 8000:8000 \
  -e FPS=3 \
  -e CONF_THRESHOLD_EXPORT=0.60 \
  -e DEBUG_VIDEO_PATH=/tmp/debug.mp4 \
  layer1-cv:latest
```

Railway deployment handoff:

- See `docs/RAILWAY_DEPLOYMENT.md` for step-by-step Railway setup, required env vars, health checks, sample verification requests, and CPU vs GPU image guidance.

### Browser Testing / CORS

By default, the API allows browser requests from local Next.js development origins:

- `http://localhost:3000`
- `http://127.0.0.1:3000`

Override this with `CORS_ALLOW_ORIGINS` for production deployments:

```bash
CORS_ALLOW_ORIGINS=https://your-app.com,https://www.your-app.com
```
