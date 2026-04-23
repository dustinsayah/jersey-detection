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

import requests as _requests

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("home_proxy")

SECRET = os.environ.get("HOME_PROXY_SECRET", "clipt-home-proxy-2026")

# Railway API config (for auto-updating HOME_PROXY_URL when tunnel URL changes)
RAILWAY_TOKEN = os.environ.get("RAILWAY_TOKEN", "")
if not RAILWAY_TOKEN:
    # Fallback: read from .railway-token file in the scripts directory
    _token_file = Path(__file__).parent / ".railway-token"
    if _token_file.exists():
        RAILWAY_TOKEN = _token_file.read_text().strip()
        log.info("Loaded Railway token from %s", _token_file)
RAILWAY_PROJECT_ID = "ac3e09ae-e3c2-41f8-8636-63f5b50d6936"
RAILWAY_SERVICE_ID = os.environ.get("RAILWAY_SERVICE_ID", "9aa045d4-2040-4c3d-b67d-cb14ed9e7a03")
RAILWAY_ENV_ID = os.environ.get("RAILWAY_ENV_ID", "4b08fce3-a80f-41f0-8fe1-2aaef8a015ba")

# Track active downloads to prevent overload
_active_downloads = 0
_tunnel_url = ""
_start_time = time.time()
_lock = threading.Lock()
MAX_CONCURRENT = 2


def update_railway_proxy_url(tunnel_url: str) -> bool:
    """Update Railway HOME_PROXY_URL env var via GraphQL API."""
    if not RAILWAY_TOKEN:
        log.warning("No RAILWAY_TOKEN set — skipping Railway update")
        return False

    query = """
    mutation UpsertVariables($input: VariableCollectionUpsertInput!) {
      variableCollectionUpsert(input: $input)
    }
    """
    variables = {
        "input": {
            "projectId": RAILWAY_PROJECT_ID,
            "serviceId": RAILWAY_SERVICE_ID,
            "environmentId": RAILWAY_ENV_ID,
            "variables": {"HOME_PROXY_URL": tunnel_url},
        }
    }

    try:
        resp = _requests.post(
            "https://backboard.railway.app/graphql/v2",
            json={"query": query, "variables": variables},
            headers={
                "Authorization": f"Bearer {RAILWAY_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        if resp.status_code == 200 and "errors" not in resp.json():
            log.info("Railway updated: HOME_PROXY_URL = %s", tunnel_url)
            return True
        else:
            log.error("Railway update failed: %s %s", resp.status_code, resp.text[:200])
            return False
    except Exception as e:
        log.error("Railway update error: %s", e)
        return False


def _cleanup_later(path: str, delay: int = 120):
    """Delete a temp file after a delay."""
    def _rm():
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass
    threading.Timer(delay, _rm).start()


def _find_cookie_file() -> str | None:
    """Find YouTube cookies file if it exists."""
    script_dir = Path(__file__).parent
    candidates = [
        script_dir / "youtube_cookies.txt",
        script_dir / "cookies.txt",
        script_dir.parent / "youtube_cookies.txt",
        script_dir.parent / "app" / "youtube_cookies.txt",
        Path.home() / "youtube_cookies.txt",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 100:
            return str(p)
    return None


# YouTube client strategies to try (in order)
_YT_CLIENTS = [
    "android",          # Best for bypassing "video not available" errors
    "web_creator",      # Reliable but may require cookies
    "mweb",             # Mobile web fallback
    None,               # Default (no override)
]


@app.route("/health")
def health():
    """Health check — Railway pings this to see if proxy is online."""
    import shutil

    try:
        import yt_dlp
        ver = yt_dlp.version.__version__
    except Exception:
        ver = "unknown"
    cookie_file = _find_cookie_file()
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    return jsonify({
        "status": "ok",
        "type": "home_proxy",
        "yt_dlp_version": ver,
        "active_downloads": _active_downloads,
        "has_cookies": cookie_file is not None,
        "cookie_file": cookie_file,
        "ffmpeg_available": bool(ffmpeg_path),
        "ffmpeg_path": ffmpeg_path,
        "ffprobe_available": bool(ffprobe_path),
        "max_quality": "720p" if ffmpeg_path else "360p (ffmpeg missing!)",
    })


@app.route("/status")
def status():
    """Extended status for monitoring."""
    import platform
    return jsonify({
        "status": "ok",
        "type": "home_proxy",
        "active_downloads": _active_downloads,
        "uptime_seconds": int(time.time() - _start_time),
        "platform": platform.system(),
        "python": platform.python_version(),
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
    preferred_client = data.get("preferred_client")  # Railway can hint which client

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

        # Use yt-dlp Python API for reliability (avoids Windows exe alias issues)
        import yt_dlp as _ytdl

        # Check if ffmpeg is available (needed for merging separate video+audio)
        has_ffmpeg = False
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            has_ffmpeg = True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        # Format selection: with ffmpeg we can merge best video + audio.
        # Without ffmpeg, we must use pre-muxed formats (lower quality but no merge needed).
        # NOTE: Do NOT constrain bestvideo to [ext=mp4] — YouTube's 720p streams
        # are often VP9/webm. yt-dlp + ffmpeg can merge any format into mp4.
        if has_ffmpeg:
            fmt = (
                f"bestvideo[height<={quality}]+bestaudio/"
                f"best[height<={quality}][ext=mp4]/"
                f"best[height={quality}]/"
                f"bestvideo[height<=480]+bestaudio/"
                f"best[height<=480]/best"
            )
        else:
            fmt = (
                f"best[height<={quality}][ext=mp4]/"
                f"best[height={quality}]/"
                f"best[height<=480]/best"
            )
            log.info("ffmpeg not installed — using pre-muxed format (may be limited to 360p)")

        base_opts = {
            "format": fmt,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "outtmpl": output_path,
        }

        if has_ffmpeg:
            base_opts["merge_output_format"] = "mp4"

        # Add time range if specified (requires ffmpeg)
        if has_ffmpeg and end_sec > start_sec > 0:
            base_opts["download_ranges"] = _ytdl.utils.download_range_func(
                None, [(start_sec, end_sec)]
            )
        elif has_ffmpeg and end_sec > 0:
            base_opts["download_ranges"] = _ytdl.utils.download_range_func(
                None, [(0, end_sec)]
            )
        elif not has_ffmpeg and (end_sec > start_sec > 0 or end_sec > 0):
            log.warning("ffmpeg not installed — downloading full video (no time trimming)")

        # Add cookies if available
        cookie_file = _find_cookie_file()
        if cookie_file:
            base_opts["cookiefile"] = cookie_file
            log.info("Using cookies from: %s", cookie_file)

        # Build client list — preferred client first if specified
        clients = list(_YT_CLIENTS)
        if preferred_client and preferred_client in clients:
            clients.remove(preferred_client)
            clients.insert(0, preferred_client)
        elif preferred_client:
            clients.insert(0, preferred_client)

        # Try multiple YouTube client strategies
        last_error = None
        for client in clients:
            ydl_opts = dict(base_opts)
            if client:
                ydl_opts["extractor_args"] = {"youtube": {"player_client": [client]}}
                log.info("Trying client: %s", client)
            else:
                log.info("Trying default client")

            # Clean up any partial files from previous attempt
            for f in Path(tmp_dir).glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass

            try:
                with _ytdl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                # Check if download produced a valid file
                if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                    log.info("Download succeeded with client: %s", client or "default")
                    break
                # Check for files with different names
                found = False
                for fp in Path(tmp_dir).iterdir():
                    if fp.stat().st_size > 10000:
                        found = True
                        break
                if found:
                    log.info("Download succeeded (alt path) with client: %s", client or "default")
                    break
            except _ytdl.DownloadError as e:
                last_error = str(e)[:300]
                log.warning("Client %s failed: %s", client or "default", last_error[:100])
                continue
            except Exception as e:
                last_error = str(e)[:300]
                log.warning("Client %s error: %s", client or "default", last_error[:100])
                continue
        else:
            # All clients failed
            log.error("All download strategies failed: %s", last_error)
            return jsonify({"error": "download_failed", "detail": last_error or "all clients failed"}), 500

        # Log everything in the temp directory for debugging
        all_files = []
        for root, dirs, files in os.walk(tmp_dir):
            for f in files:
                fp = os.path.join(root, f)
                all_files.append((fp, os.path.getsize(fp)))
        log.info("Files in temp dir after download: %s", all_files)

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
            # Check for any video file in the temp directory
            for fp, sz in all_files:
                if sz > 10000:
                    log.info("Found valid file at different path: %s (%d bytes)", fp, sz)
                    output_path = fp
                    break
            else:
                # Also check for .mkv or other extension (merge sometimes produces different ext)
                for ext in [".mkv", ".webm", ".mp4.part"]:
                    alt = output_path.replace(".mp4", ext)
                    if os.path.exists(alt) and os.path.getsize(alt) > 10000:
                        output_path = alt
                        break
                else:
                    log.error("No valid file found. Expected at: %s", output_path)
                    return jsonify({"error": "file_too_small", "files_found": [(f, s) for f, s in all_files]}), 500

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
        quality_label = (
            "1080p" if height >= 1080 else
            "720p" if height >= 720 else
            "480p" if height >= 480 else
            f"{height}p (LOW — ffmpeg may be missing)"
        )
        log.info("Done: %s (%dp), %sMB, %ss, ffmpeg=%s", quality_label, height, size_mb, elapsed, has_ffmpeg)

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


@app.route("/test-url", methods=["POST"])
def test_url():
    """Test if a YouTube URL can be resolved — tries all clients, reports which work."""
    data = request.json or {}
    if data.get("secret") != SECRET:
        return jsonify({"error": "unauthorized"}), 401

    url = data.get("url")
    if not url:
        return jsonify({"error": "no url"}), 400

    import yt_dlp as _ytdl

    cookie_file = _find_cookie_file()
    results = {}

    for client in _YT_CLIENTS:
        label = client or "default"
        opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
        if cookie_file:
            opts["cookiefile"] = cookie_file
        if client:
            opts["extractor_args"] = {"youtube": {"player_client": [client]}}

        try:
            with _ytdl.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = info.get("formats", [])
                video_fmts = [f for f in formats if f.get("height") and f.get("vcodec") != "none"]
                best_h = max((f.get("height", 0) for f in video_fmts), default=0) if video_fmts else 0
                results[label] = {
                    "status": "ok",
                    "title": info.get("title"),
                    "duration": info.get("duration"),
                    "best_height": best_h,
                    "format_count": len(video_fmts),
                }
        except Exception as e:
            results[label] = {"status": "failed", "error": str(e)[:200]}

    return jsonify({
        "url": url,
        "has_cookies": cookie_file is not None,
        "results": results,
    })


@app.route("/set-tunnel-url", methods=["POST"])
def set_tunnel_url():
    """Called by run_home_proxy.bat after tunnel URL is detected.
    Updates Railway HOME_PROXY_URL automatically."""
    global _tunnel_url
    data = request.json or {}
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "no url"}), 400

    _tunnel_url = url
    log.info("Tunnel URL set: %s", url)

    # Update Railway in background thread
    def _bg():
        update_railway_proxy_url(url)
    threading.Thread(target=_bg, daemon=True).start()

    return jsonify({"status": "ok", "url": url, "railway_update": "queued"})


def _find_free_port(preferred: int = 5050) -> int | None:
    """Find a free port starting from preferred port.

    On Windows, SO_REUSEADDR allows multiple processes to bind the same port,
    so we test by binding WITHOUT that flag, then immediately release.
    """
    import socket
    for port in range(preferred, preferred + 20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.bind(("0.0.0.0", port))
                s.listen(1)
                # Successfully bound and listening — port is free
                return port
        except OSError:
            continue
    return None


if __name__ == "__main__":
    preferred = int(os.environ.get("HOME_PROXY_PORT", 5050))
    port = _find_free_port(preferred)

    if port is None:
        print("FATAL: No free port found between 5050-5069", flush=True)
        sys.exit(1)

    # Write port to temp file so bat file can read it
    port_file = os.path.join(tempfile.gettempdir(), "clipt_proxy_port.txt")
    with open(port_file, "w") as f:
        f.write(str(port))

    # Emit machine-readable line that bat file parses
    print(f"CLIPT_PORT={port}", flush=True)
    print(f"Starting Clipt Home Proxy on port {port}...", flush=True)

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
