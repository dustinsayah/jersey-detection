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
    CONF_THRESHOLD_INTERNAL=0.3 \
    GUNICORN_TIMEOUT=1800 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    curl unzip git \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 LTS — required for bgutil PO Token server
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    node --version && npm --version

# deno is needed by yt-dlp for YouTube JS extraction
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh
ENV DENO_DIR=/tmp/deno

# wireproxy: lightweight WireGuard SOCKS5 proxy for Cloudflare WARP
# YouTube blocks sports video downloads from datacenter IPs (Railway).
# WARP tunnels traffic through Cloudflare's non-datacenter network.
RUN ARCH=$(dpkg --print-architecture) && \
    curl -fsSL "https://github.com/pufferffish/wireproxy/releases/download/v1.0.9/wireproxy_linux_${ARCH}.tar.gz" \
    | tar xz -C /usr/local/bin wireproxy && \
    chmod +x /usr/local/bin/wireproxy && \
    wireproxy --version || echo "wireproxy install: will retry with amd64" && \
    if [ ! -x /usr/local/bin/wireproxy ]; then \
      curl -fsSL "https://github.com/pufferffish/wireproxy/releases/download/v1.0.9/wireproxy_linux_amd64.tar.gz" \
      | tar xz -C /usr/local/bin wireproxy; \
    fi

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && (pip install --no-cache-dir tflite-runtime==2.14.0 || echo "tflite-runtime not available, YAMNet will be skipped")

# bundle the person seg model so it doesn't download at runtime
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n-seg.pt')" \
    && mkdir -p /app/app/model \
    && mv yolo26n-seg.pt /app/app/model/yolo26n-seg.pt

# bundle YOLO pose model for pose estimation
RUN python -c "from ultralytics import YOLO; m=YOLO('yolo11n-pose.pt'); import shutil; shutil.move('yolo11n-pose.pt', '/app/app/model/yolo11n-pose.pt')" || true

# Pre-download YAMNet TFLite model for audio classification (non-fatal if fails)
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite', '/app/app/model/yamnet.tflite'); print('YAMNet downloaded')" \
    || echo "YAMNet download skipped"

# The jersey-number model is project-specific and must be provided separately
# at runtime or baked into a derivative image at app/model/jersey_number_yolo11m.pt.

# Bootstrap public reader: clone external repos + download public checkpoints
COPY scripts /app/scripts
RUN python /app/scripts/bootstrap_public_reader.py

# Keep yt-dlp updated at build time — try nightly, fall back to stable
RUN pip install --upgrade --pre yt-dlp \
    || pip install --upgrade yt-dlp

# bgutil PO Token provider — generates YouTube Proof-of-Origin tokens
# Required for web/mweb clients to bypass bot detection.
# Server runs as background process (start.sh starts it on port 4416).
# Plugin auto-intercepts yt-dlp web client requests to add PO tokens.
RUN git clone --depth=1 https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git /app/bgutil-pot && \
    cd /app/bgutil-pot/server && npm ci && npx -p typescript tsc && \
    echo "bgutil PO Token server built successfully" || \
    echo "bgutil PO Token build failed (non-fatal — WARP strategies still work)"
# NOTE: Do NOT install yt-dlp-get-pot — it's deprecated (archived Nov 2025)
# and conflicts with yt-dlp's native PO Token Provider Framework (since 2025.05.22).
# bgutil-ytdlp-pot-provider v1.0.0+ registers directly with the native framework.
RUN pip install --no-cache-dir bgutil-ytdlp-pot-provider || \
    echo "PO Token pip package failed (non-fatal)"

# Pre-cache EJS challenge solver script so yt-dlp doesn't download at runtime.
# This is REQUIRED for YouTube n-challenge solving (unlocks 720p+ DASH formats).
RUN yt-dlp --remote-components ejs:github --extractor-args "youtube:player_client=android_vr" \
    --print "%(id)s" --no-download "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>/dev/null \
    || echo "EJS pre-cache: solver will be downloaded on first use"

# Ensure model directory exists for Roboflow trained weights
RUN mkdir -p /app/app/model

# Cache bust for code changes (update this on each deploy)
# Uninstall deprecated yt-dlp-get-pot if leftover from Docker cache layers
RUN pip uninstall -y yt-dlp-get-pot 2>/dev/null || true

ARG CACHE_BUST=v8.19.2
RUN echo "Build version: $CACHE_BUST"

COPY app /app/app
COPY asgi.py /app/asgi.py
COPY layers /app/layers

# Startup script: WARP proxy + volume models + gunicorn
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

# Railway uses its own healthcheck via railway.toml (healthcheckPath = "/health")
# Docker HEALTHCHECK removed to avoid conflict with Railway's zero-downtime deploy

CMD ["/app/start.sh"]
