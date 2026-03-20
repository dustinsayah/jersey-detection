FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    YOLO_MODEL_SOURCE=app/model/jersey_number_yolo11m.pt \
    PERSON_MODEL_SOURCE=app/model/yolo26n-seg.pt \
    DETECTION_STRATEGY=detection_first \
    JERSEY_READER_BACKEND=public_reader_ensemble \
    PUBLIC_READER_ALLOW_LEGACY_FALLBACK=true \
    FPS=2 \
    CONF_THRESHOLD_EXPORT=0.55 \
    GUNICORN_TIMEOUT=1800 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl unzip git \
    && rm -rf /var/lib/apt/lists/*

# deno is needed by yt-dlp for YouTube JS extraction
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh
ENV DENO_DIR=/tmp/deno

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements.txt

# bundle the person seg model so it doesn't download at runtime
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n-seg.pt')" \
    && mkdir -p /app/app/model \
    && mv yolo26n-seg.pt /app/app/model/yolo26n-seg.pt

# The jersey-number model is project-specific and must be provided separately
# at runtime or baked into a derivative image at app/model/jersey_number_yolo11m.pt.

# Bootstrap public reader: clone external repos + download public checkpoints
COPY scripts /app/scripts
RUN python /app/scripts/bootstrap_public_reader.py

COPY app /app/app
COPY asgi.py /app/asgi.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD sh -c "curl --fail http://127.0.0.1:${PORT:-8000}/live || exit 1"

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout ${GUNICORN_TIMEOUT:-1800} asgi:app"]
