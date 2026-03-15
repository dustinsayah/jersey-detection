# Railway Deployment Guide

This document covers the exact handoff steps needed to deploy the Layer 1 jersey-detection API on Railway and verify that it is live.

## Recommended Compute

For long-form production processing, use a GPU-enabled machine. The recommended minimum is a mid-tier NVIDIA GPU in the `T4 / L4 / RTX 3060+` class.

- `GPU target`: best fit for long game footage and the 6-12 minute target discussed for 60-minute videos
- `CPU-only deployment`: supported for testing and low volume, but it will be significantly slower on long videos

Important note:

- The current repo `Dockerfile` installs CPU-only PyTorch to keep the image size smaller for Railway-style deployment limits.
- The repo also includes `Dockerfile.gpu` for CUDA-based deployment on a GPU machine.
- For the current CPU Railway path, keep the Python dependency matrix aligned with `torch==2.1.0+cpu` / `torchvision==0.16.0+cpu`. In practice that means using a NumPy 1.x line and an OpenCV build compatible with it, rather than NumPy 2.x.

## What The Client Needs Before Deploying

1. A Railway project with a web service.
2. A GPU-enabled instance if production long-form performance is required.
3. The jersey-number model file available at `app/model/jersey_number_yolo11m.pt`, or a derivative image that already bakes that file in.
4. A public frontend origin for CORS, for example `https://cliptapp.com`.

## Model File Requirement

The person-segmentation model is bundled during image build, but the jersey-number model is project-specific.

Before deploying, make sure one of these is true:

1. The repo contains `app/model/jersey_number_yolo11m.pt` before the image is built.
2. A custom image is built that copies the jersey-number model into `app/model/jersey_number_yolo11m.pt`.
3. The deployment mounts the model at runtime and `YOLO_MODEL_SOURCE` points to that mounted path.

If this file is missing, `/health` will report the service as not ready.

## Deployment Steps

### Option A: Deploy From GitHub Repo

1. Push the repo to GitHub.
2. In Railway, create a new project.
3. Add a new service from the GitHub repo.
4. Let Railway detect the `Dockerfile` in the repo root.
5. Choose a GPU-enabled machine for the service if you want production long-form performance.
6. Set the required environment variables listed below.
7. Deploy the service.

### Option B: Deploy From Docker Image

1. Build and push the image to Docker Hub or another registry.
2. In Railway, create a new project.
3. Add a new service from the container image.
4. Point Railway to the image tag.
5. Choose a GPU-enabled machine for the service if you want production long-form performance.
6. Set the required environment variables listed below.
7. Deploy the service.

## Build Commands

### CPU-friendly image

```bash
docker build -t layer1-cv:latest .
```

### GPU image

```bash
docker build -f Dockerfile.gpu -t layer1-cv:gpu .
```

If you are pushing to a registry:

```bash
docker tag layer1-cv:gpu YOUR_DOCKERHUB_USER/layer1-cv:gpu
docker push YOUR_DOCKERHUB_USER/layer1-cv:gpu
```

## Required Environment Variables

Set these in Railway before or immediately after the first deploy:

```bash
PORT=8000
GUNICORN_TIMEOUT=1800
CORS_ALLOW_ORIGINS=https://cliptapp.com,https://www.cliptapp.com
YOLO_MODEL_SOURCE=app/model/jersey_number_yolo11m.pt
PERSON_MODEL_SOURCE=app/model/yolo26n-seg.pt
DETECTION_STRATEGY=detection_first
FPS=2
CONF_THRESHOLD_EXPORT=0.55
YT_DLP_JS_RUNTIMES=deno
```

Recommended additions:

```bash
LOG_LEVEL=INFO
PIPELINE_WORKERS=4
SKIP_SIMILARITY_THRESHOLD=0.97
EARLY_EXIT_CONSECUTIVE=0
```

Device setting:

- For the current CPU-friendly image, set `YOLO_DEVICE=cpu`.
- For `Dockerfile.gpu`, set `YOLO_DEVICE=cuda`.

## Service / Health Settings

Use these values in Railway:

- `Port`: `8000`
- `Start command`: use the container default command from the `Dockerfile`
- `Health check path`: `/live`

Useful endpoints:

- `GET /live` returns basic liveness
- `GET /health` or `GET /ready` returns readiness and startup dependency status

Expected healthy responses:

```json
{ "status": "ok" }
```

If the model or runtime binaries are missing, `/health` will return `503` with an error detail.

## First Live Checks

After Railway gives you the base URL, verify the deployment in this order.

### 1. Liveness

```bash
curl https://YOUR-RAILWAY-URL/live
```

Expected:

```json
{ "status": "ok" }
```

### 2. Readiness

```bash
curl https://YOUR-RAILWAY-URL/health
```

Expected when ready:

```json
{ "status": "ok" }
```

If it returns `503`, check:

- model file path
- `ffmpeg`
- `yt-dlp`
- environment variable values

### 3. Detection Request

For first verification, use a direct public MP4 or Cloudinary video URL. That avoids extra YouTube-download variables while confirming the API is live.

```bash
curl -X POST https://YOUR-RAILWAY-URL/detect \
  -H "Content-Type: application/json" \
  -d '{
    "videoUrl": "https://res.cloudinary.com/demo/video/upload/v1615311206/dog.mp4",
    "jerseyNumber": 2,
    "jerseyColor": "white",
    "sport": "basketball",
    "position": "guard"
  }'
```

Expected response shape:

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

## Frontend Integration Notes

- The frontend should call the Railway base URL directly or through the existing backend.
- For browser-based calls, make sure `CORS_ALLOW_ORIGINS` includes the real frontend domains.
- For first production validation, direct public `Cloudinary` MP4 URLs are the safest input format.

## YouTube Inputs

The service supports YouTube URLs through `yt-dlp`, but for deployment validation it is better to first confirm:

1. the service is healthy
2. direct MP4 or Cloudinary URLs work
3. the Railway environment allows the desired network/runtime behavior for YouTube downloads

For production, if YouTube downloads become unreliable in the hosting environment, the fallback is:

1. download or normalize the video on the Clipt side
2. pass a direct public video URL into `/detect`

## Performance Expectations

- `GPU deployment`: recommended for long game footage
- `Minimum GPU target`: `T4 / L4 / RTX 3060+`
- `CPU deployment`: acceptable for testing and low volume, but not ideal for long-form production workloads

If the plan is to process many long videos concurrently, the next production step is a queued worker architecture rather than sending all jobs straight to a single web instance.

## Troubleshooting

### `/health` returns `503`

Common causes:

- `app/model/jersey_number_yolo11m.pt` is missing
- `YOLO_MODEL_SOURCE` points to the wrong path
- `ffmpeg` is missing
- `yt-dlp` is missing

### The app is live but very slow

Common causes:

- CPU-only deployment
- long YouTube/HLS source
- too many large requests on one instance

For long-form production speed, use a GPU-enabled deployment and a direct MP4 source when possible.

### Browser tester gets blocked

Check:

- `CORS_ALLOW_ORIGINS`
- frontend domain spelling
- whether the request is being sent from the browser or from the backend

## Handoff Summary

When closing M2, send the client:

1. The Railway base URL
2. This deployment guide
3. The required environment variables
4. One sample `/detect` request
5. The expected response shape
