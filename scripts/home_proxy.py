#!/usr/bin/env python3
"""
Clipt Home Proxy — runs on your computer, gives Railway 720p+ YouTube downloads.

Your home IP is residential, so YouTube serves full quality (720p/1080p).
Cloudflare Tunnel gives Railway a permanent HTTPS URL to reach this server.

Setup (one time):
    pip install flask yt-dlp
    # Then set up Cloudflare Tunnel (see setup_home_proxy.bat)

Run:
    python scripts/home_proxy.py

Keep this terminal open while using Clipt AI Highlights.
When this is closed, Railway falls back to 360p (still works, just lower quality).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("home_proxy")

SECRET = os.environ.get("HOME_PROXY_SECRET", "clipt-home-proxy-2026")

# Track active downloads to prevent overload
_active_downloads = 0
_lock = threading.Lock()
MAX_CONCURRENT = 2


def _cleanup_later(path: str, delay: int = 120):
    """Delete a temp file after a delay."""
    def _rm():
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass
    threading.Timer(delay, _rm).start()


@app.route("/health")
def health():
    """Health check — Railway pings this to see if proxy is online."""
    try:
        import yt_dlp
        ver = yt_dlp.version.__version__
    except Exception:
        ver = "unknown"
    return jsonify({
        "status": "ok",
        "type": "home_proxy",
        "yt_dlp_version": ver,
        "active_downloads": _active_downloads,
    })


@app.route("/download", methods=["POST"])
def download():
    """Download a YouTube video segment and return the file.

    Expects JSON body:
        url: YouTube URL
        start_sec: start time in seconds (default 0)
        end_sec: end time in seconds (default 60)
        quality: max height in pixels (default 720)
        secret: auth secret
    """
    global _active_downloads

    data = request.json or {}

    # Auth check
    if data.get("secret") != SECRET:
        return jsonify({"error": "unauthorized"}), 401

    url = data.get("url")
    if not url:
        return jsonify({"error": "no url"}), 400

    start_sec = float(data.get("start_sec", 0))
    end_sec = float(data.get("end_sec", 60))
    quality = int(data.get("quality", 720))

    # Concurrency limit
    with _lock:
        if _active_downloads >= MAX_CONCURRENT:
            return jsonify({"error": "busy", "active": _active_downloads}), 503
        _active_downloads += 1

    tmp_dir = tempfile.mkdtemp(prefix="clipt_home_")
    output_path = os.path.join(tmp_dir, "video.mp4")

    try:
        log.info("Downloading: %s [%s-%s] @ %dp", url, start_sec, end_sec, quality)
        t0 = time.time()

        # Use yt-dlp CLI for reliability (handles all edge cases)
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--format", f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality}][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "--no-warnings",
            "--quiet",
        ]

        # Add time range if specified
        if end_sec > start_sec > 0:
            cmd += ["--download-sections", f"*{start_sec}-{end_sec}"]
        elif end_sec > 0:
            cmd += ["--download-sections", f"*0-{end_sec}"]

        cmd += ["-o", output_path, url]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            log.error("yt-dlp failed: %s", result.stderr[:500])
            return jsonify({"error": "download_failed", "detail": result.stderr[:300]}), 500

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
            # Check for .mkv or other extension (merge sometimes produces different ext)
            for ext in [".mkv", ".webm", ".mp4.part"]:
                alt = output_path.replace(".mp4", ext)
                if os.path.exists(alt) and os.path.getsize(alt) > 10000:
                    output_path = alt
                    break
            else:
                return jsonify({"error": "file_too_small"}), 500

        # Get video height via ffprobe
        height = 0
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=height", "-of", "csv=p=0", output_path],
                capture_output=True, text=True, timeout=10)
            height = int(probe.stdout.strip())
        except Exception:
            pass

        elapsed = round(time.time() - t0, 1)
        size_mb = round(os.path.getsize(output_path) / 1_048_576, 1)
        log.info("Done: %dp, %sMB, %ss", height, size_mb, elapsed)

        # Schedule cleanup
        _cleanup_later(output_path, delay=120)
        _cleanup_later(tmp_dir, delay=180)

        return send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="video.mp4",
        ), {"X-Video-Height": str(height), "X-Download-Time": str(elapsed)}

    except subprocess.TimeoutExpired:
        return jsonify({"error": "timeout"}), 504
    except Exception as e:
        log.exception("Download error")
        return jsonify({"error": str(e)}), 500
    finally:
        with _lock:
            _active_downloads -= 1


@app.route("/extract-info", methods=["POST"])
def extract_info():
    """Extract video info (formats, duration) without downloading.

    Note: Stream URLs are IP-bound and CANNOT be used from Railway.
    This endpoint is for metadata only (duration, available formats).
    """
    data = request.json or {}
    if data.get("secret") != SECRET:
        return jsonify({"error": "unauthorized"}), 401

    url = data.get("url")
    if not url:
        return jsonify({"error": "no url"}), 400

    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            formats = info.get("formats", [])
            video_formats = [
                {"height": f.get("height"), "ext": f.get("ext"),
                 "vcodec": f.get("vcodec", ""), "fps": f.get("fps")}
                for f in formats if f.get("height") and f.get("vcodec") != "none"
            ]
            return jsonify({
                "title": info.get("title"),
                "duration": info.get("duration"),
                "best_height": max((f["height"] for f in video_formats), default=0),
                "formats": sorted(video_formats, key=lambda f: f["height"] or 0, reverse=True)[:10],
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("HOME_PROXY_PORT", 5000))

    print(f"""
====================================================
  CLIPT HOME PROXY
====================================================
  Local:  http://localhost:{port}
  Health: http://localhost:{port}/health

  Next steps:
  1. In a SECOND terminal, start Cloudflare Tunnel:
     cloudflared tunnel run clipt-proxy

  2. Add to Railway env vars:
     HOME_PROXY_URL = https://proxy.cliptapp.com
     HOME_PROXY_SECRET = {SECRET}

  3. Keep BOTH terminals open while using AI Highlights
====================================================
""")

    app.run(host="0.0.0.0", port=port, threaded=True)
