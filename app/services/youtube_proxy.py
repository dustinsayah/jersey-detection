# YouTube download proxy — 5-strategy robust download chain
# Strategies tried in order until one succeeds:
# 1. Render server proxy (cloudinary cache)
# 2. yt-dlp with ANDROID client (proven working Apr 2026)
# 3. yt-dlp with android+web combo
# 4. yt-dlp Python library with android client (no subprocess)
# 5. Render server /extract-frames (last resort)

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)

# Render server URL
RENDER_SERVER_URL = os.getenv(
    "RENDER_SERVER_URL",
    "https://clipt-render-server-production.up.railway.app",
).rstrip("/")

_YT_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/live/)([A-Za-z0-9_-]{11})"
)

# Matches &t=36s, &t=36, ?t=1m30s, &time_continue=36, etc.
_YT_TIMESTAMP_RE = re.compile(
    r"[?&](?:t|time_continue)=(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?",
    re.IGNORECASE,
)

# Format string — prefer highest quality muxed MP4 for jersey OCR accuracy
# Priority: muxed 720p mp4 (safe for OpenCV) → muxed best mp4 → merged 1080p → format 18 fallback
_ANDROID_FORMAT = "best[height<=720][ext=mp4]/best[ext=mp4]/bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/18/best"


def is_youtube_url(url: str) -> bool:
    return bool(_YT_PATTERN.search(url))


def extract_video_id(url: str) -> str | None:
    m = _YT_PATTERN.search(url)
    return m.group(1) if m else None


def normalize_youtube_url(url: str) -> tuple[str, float]:
    """Normalize a YouTube URL to a clean canonical form.

    Returns:
        (clean_url, extracted_start_seconds)
    """
    video_id = extract_video_id(url)
    if not video_id:
        return url, 0.0

    start_seconds = 0.0
    ts_match = _YT_TIMESTAMP_RE.search(url)
    if ts_match:
        hours = int(ts_match.group(1) or 0)
        minutes = int(ts_match.group(2) or 0)
        seconds = int(ts_match.group(3) or 0)
        start_seconds = float(hours * 3600 + minutes * 60 + seconds)

    clean_url = f"https://www.youtube.com/watch?v={video_id}"
    return clean_url, start_seconds


def _trim_video(
    input_path: Path,
    start: float,
    end: float,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Trim a downloaded video to the requested time range using ffmpeg -c copy."""
    trimmed_path = input_path.parent / "trimmed.mp4"
    try:
        cmd = [
            ffmpeg_binary, "-y",
            "-i", str(input_path),
            "-ss", str(start),
            "-to", str(end),
            "-c", "copy",
            str(trimmed_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and trimmed_path.exists() and trimmed_path.stat().st_size > 1000:
            original_mb = round(input_path.stat().st_size / 1024 / 1024, 1)
            trimmed_mb = round(trimmed_path.stat().st_size / 1024 / 1024, 1)
            LOGGER.info("Trimmed to %ss-%ss: %sMB → %sMB", start, end, original_mb, trimmed_mb)
            input_path.unlink()
            trimmed_path.rename(input_path)
            return input_path
        LOGGER.warning("ffmpeg trim failed (returncode=%d), using untrimmed: %s", result.returncode, result.stderr[:200])
    except Exception as exc:
        LOGGER.warning("ffmpeg trim failed, using untrimmed: %s", exc)
    return input_path


def _yt_dlp_download(
    url: str,
    output_path: Path,
    yt_dlp_binary: str,
    ffmpeg_binary: str,
    client: str,
    start_time: float = 0,
    end_time: float = 0,
    timeout: int = 180,
    strategy_name: str = "",
) -> bool:
    """Run yt-dlp subprocess with given client. Returns True on success."""
    if output_path.exists():
        output_path.unlink()

    cmd = [
        yt_dlp_binary,
        "--no-check-certificate",
        "--extractor-args", f"youtube:player_client={client}",
        "--downloader-args", "ffmpeg_i:-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
        "--format", _ANDROID_FORMAT,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--socket-timeout", "30",
        "-o", str(output_path),
    ]

    # Add EJS support if deno is available (helps with some extraction)
    cmd.extend(["--remote-components", "ejs:github"])

    if start_time > 0 or end_time > 0:
        section = f"*{start_time}-{end_time}" if end_time > 0 else f"*{start_time}-inf"
        cmd.extend(["--download-sections", section, "--force-keyframes-at-cuts"])

    cmd.append(url)

    LOGGER.info("%s: running yt-dlp client=%s", strategy_name, client)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
        file_mb = round(output_path.stat().st_size / 1024 / 1024, 1)
        LOGGER.info("%s: SUCCESS — %sMB downloaded (client=%s)", strategy_name, file_mb, client)
        return True

    err = result.stderr[:300] if result.stderr else "no stderr"
    LOGGER.warning("%s: FAILED (client=%s): %s", strategy_name, client, err)
    return False


def _yt_dlp_python_download(
    url: str,
    output_path: Path,
    client: str = "android",
    start_time: float = 0,
    end_time: float = 0,
    strategy_name: str = "Strategy 4",
) -> bool:
    """Use yt-dlp as Python library (no subprocess). Returns True on success."""
    try:
        import yt_dlp

        if output_path.exists():
            output_path.unlink()

        ydl_opts = {
            "format": _ANDROID_FORMAT,
            "merge_output_format": "mp4",
            "outtmpl": str(output_path),
            "extractor_args": {"youtube": {"player_client": [client]}},
            "no_check_certificate": True,
            "socket_timeout": 30,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }

        if (start_time > 0 or end_time > 0):
            ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
                None, [(start_time, end_time if end_time > 0 else float("inf"))]
            )

        LOGGER.info("%s: yt-dlp Python library (client=%s)", strategy_name, client)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if output_path.exists() and output_path.stat().st_size > 1000:
            file_mb = round(output_path.stat().st_size / 1024 / 1024, 1)
            LOGGER.info("%s: SUCCESS — %sMB (Python lib, client=%s)", strategy_name, file_mb, client)
            return True

        LOGGER.warning("%s: file missing or too small after download", strategy_name)
        return False
    except Exception as exc:
        LOGGER.warning("%s: FAILED (Python lib, client=%s): %s", strategy_name, client, str(exc)[:200])
        return False


def _render_server_download(
    url: str,
    output_path: Path,
    start_time: float,
    end_time: float,
    ffmpeg_binary: str,
    client: httpx.Client,
    strategy_name: str = "Strategy 1",
) -> bool:
    """Try render server download. Returns True on success."""
    if not RENDER_SERVER_URL:
        return False

    try:
        LOGGER.info("%s: render server at %s", strategy_name, RENDER_SERVER_URL)
        payload: dict = {"youtubeUrl": url}
        if start_time > 0:
            payload["startTime"] = start_time
        if end_time > 0:
            payload["endTime"] = end_time

        resp = client.post(f"{RENDER_SERVER_URL}/download-youtube", json=payload)

        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "")

            # Direct video bytes
            if "video" in content_type or "octet-stream" in content_type:
                output_path.write_bytes(resp.content)
                file_mb = round(len(resp.content) / 1024 / 1024, 1)
                LOGGER.info("%s: SUCCESS — %sMB (render server bytes)", strategy_name, file_mb)
                return True

            # JSON with cloudinary URL
            if "json" in content_type or "text" in content_type:
                data = resp.json()
                download_url = (
                    data.get("cloudinaryUrl")
                    or data.get("downloadUrl")
                    or data.get("url")
                    or data.get("videoUrl")
                )
                if download_url:
                    LOGGER.info("%s: downloading from cloudinary: %s", strategy_name, download_url[:80])
                    dl_resp = client.get(download_url, follow_redirects=True)
                    if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                        output_path.write_bytes(dl_resp.content)
                        file_mb = round(len(dl_resp.content) / 1024 / 1024, 1)
                        LOGGER.info("%s: SUCCESS — %sMB (cloudinary)", strategy_name, file_mb)
                        return True

        LOGGER.warning("%s: FAILED — render server returned %d: %s",
                       strategy_name, resp.status_code, resp.text[:200])
    except Exception as exc:
        LOGGER.warning("%s: FAILED — %s", strategy_name, str(exc)[:200])

    return False


def download_youtube_sync(
    url: str,
    *,
    start_time: float = 0,
    end_time: float = 0,
    yt_dlp_binary: str = "yt-dlp",
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Synchronous 5-strategy YouTube download chain.

    Strategy order:
    1. Render server proxy (cloudinary cache — fastest)
    2. yt-dlp subprocess with android client (proven working Apr 2026)
    3. yt-dlp subprocess with android+web combo
    4. yt-dlp Python library with android client
    5. Render server /extract-frames (last resort)
    """
    LOGGER.info("youtube_proxy_sync called with URL: %s", url)

    original_url = url
    url, url_start_seconds = normalize_youtube_url(url)
    if url != original_url:
        LOGGER.info("youtube_proxy_sync: normalized %s → %s (t=%.0fs)", original_url, url, url_start_seconds)
    if url_start_seconds > 0 and start_time == 0:
        start_time = url_start_seconds

    LOGGER.info("youtube_proxy_sync: RENDER_SERVER_URL=%s, time=%s-%s", RENDER_SERVER_URL, start_time, end_time)
    dl_start = time.perf_counter()
    tmp_dir = Path(tempfile.mkdtemp(prefix="clipt_yt_sync_"))
    output_path = tmp_dir / "video.mp4"

    strategy_errors: list[str] = []

    # ── Strategy 1: Render server proxy ──
    with httpx.Client(timeout=httpx.Timeout(90)) as client:
        if _render_server_download(url, output_path, start_time, end_time, ffmpeg_binary, client, "Strategy 1"):
            elapsed = round(time.perf_counter() - dl_start, 1)
            LOGGER.info("Sync downloaded in %ss via Strategy 1 (render server)", elapsed)
            if start_time > 0 or end_time > 0:
                output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
            return output_path
    strategy_errors.append("1=render_server_failed")

    # ── Strategy 2: yt-dlp android client (PROVEN WORKING) ──
    if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                        client="android", start_time=start_time, end_time=end_time,
                        strategy_name="Strategy 2"):
        elapsed = round(time.perf_counter() - dl_start, 1)
        LOGGER.info("Sync downloaded in %ss via Strategy 2 (yt-dlp android)", elapsed)
        if start_time > 0 or end_time > 0:
            output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
        return output_path
    strategy_errors.append("2=yt-dlp_android_failed")

    # ── Strategy 3: yt-dlp android+web combo ──
    if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                        client="android,web", start_time=start_time, end_time=end_time,
                        strategy_name="Strategy 3"):
        elapsed = round(time.perf_counter() - dl_start, 1)
        LOGGER.info("Sync downloaded in %ss via Strategy 3 (yt-dlp android+web)", elapsed)
        if start_time > 0 or end_time > 0:
            output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
        return output_path
    strategy_errors.append("3=yt-dlp_android+web_failed")

    # ── Strategy 4: yt-dlp Python library with android client ──
    if _yt_dlp_python_download(url, output_path, client="android",
                               start_time=start_time, end_time=end_time,
                               strategy_name="Strategy 4"):
        elapsed = round(time.perf_counter() - dl_start, 1)
        LOGGER.info("Sync downloaded in %ss via Strategy 4 (Python lib android)", elapsed)
        if start_time > 0 or end_time > 0:
            output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
        return output_path
    strategy_errors.append("4=python_lib_android_failed")

    # ── Strategy 5: Render server /extract-frames (last resort) ──
    if RENDER_SERVER_URL:
        try:
            LOGGER.info("Strategy 5: render server /extract-frames (last resort)")
            with httpx.Client(timeout=httpx.Timeout(120)) as client:
                payload: dict = {"youtubeUrl": url}
                if start_time > 0:
                    payload["startTime"] = start_time
                if end_time > 0:
                    payload["endTime"] = end_time
                resp = client.post(f"{RENDER_SERVER_URL}/extract-frames", json=payload)
                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    if "video" in content_type or "octet-stream" in content_type:
                        if output_path.exists():
                            output_path.unlink()
                        output_path.write_bytes(resp.content)
                        elapsed = round(time.perf_counter() - dl_start, 1)
                        LOGGER.info("Sync downloaded in %ss via Strategy 5 (extract-frames)", elapsed)
                        if start_time > 0 or end_time > 0:
                            output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                        return output_path
                    ct = resp.headers.get("content-type", "")
                    if "json" in ct or "text" in ct:
                        data = resp.json()
                        download_url = data.get("cloudinaryUrl") or data.get("downloadUrl") or data.get("url") or data.get("videoUrl")
                        if download_url:
                            dl_resp = client.get(download_url, follow_redirects=True)
                            if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                                if output_path.exists():
                                    output_path.unlink()
                                output_path.write_bytes(dl_resp.content)
                                elapsed = round(time.perf_counter() - dl_start, 1)
                                LOGGER.info("Sync downloaded in %ss via Strategy 5 (extract-frames URL)", elapsed)
                                if start_time > 0 or end_time > 0:
                                    output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                                return output_path
                LOGGER.warning("Strategy 5 failed: %d", resp.status_code)
        except Exception as exc:
            LOGGER.warning("Strategy 5 failed: %s", exc)
    strategy_errors.append("5=extract_frames_failed")

    raise RuntimeError(
        f"All 5 YouTube download strategies failed (sync) for: {url} "
        f"(original: {original_url}). "
        f"Errors: {', '.join(strategy_errors)}. "
        f"Check Railway logs for per-strategy errors."
    )


async def download_youtube(
    url: str,
    *,
    start_time: float = 0,
    end_time: float = 0,
    yt_dlp_binary: str = "yt-dlp",
    ffmpeg_binary: str = "ffmpeg",
    timeout: float = 120,
) -> Path:
    """Async version — delegates to sync via threadpool."""
    from functools import partial
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(
        partial(
            download_youtube_sync,
            url,
            start_time=start_time,
            end_time=end_time,
            yt_dlp_binary=yt_dlp_binary,
            ffmpeg_binary=ffmpeg_binary,
        )
    )


def test_youtube_download_sync(
    url: str,
    yt_dlp_binary: str = "yt-dlp",
    ffmpeg_binary: str = "ffmpeg",
) -> dict:
    """Test YouTube download and report which strategy worked."""
    start = time.perf_counter()
    strategy_results: list[dict] = []

    # Test Strategy 1: Render server
    s1_start = time.perf_counter()
    try:
        with httpx.Client(timeout=httpx.Timeout(60)) as client:
            payload = {"youtubeUrl": url}
            resp = client.post(f"{RENDER_SERVER_URL}/download-youtube", json=payload)
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "")
                data = resp.json() if "json" in ct else {}
                cloud_url = data.get("cloudinaryUrl", "")
                if cloud_url:
                    strategy_results.append({
                        "strategy": 1, "name": "render_server",
                        "status": "success", "cloudinaryUrl": cloud_url,
                        "elapsed_ms": round((time.perf_counter() - s1_start) * 1000),
                    })
                else:
                    strategy_results.append({
                        "strategy": 1, "name": "render_server",
                        "status": "no_url",
                        "elapsed_ms": round((time.perf_counter() - s1_start) * 1000),
                    })
            else:
                strategy_results.append({
                    "strategy": 1, "name": "render_server",
                    "status": f"http_{resp.status_code}",
                    "elapsed_ms": round((time.perf_counter() - s1_start) * 1000),
                })
    except Exception as exc:
        strategy_results.append({
            "strategy": 1, "name": "render_server",
            "status": "error", "error": str(exc)[:200],
            "elapsed_ms": round((time.perf_counter() - s1_start) * 1000),
        })

    # Full download test using sync chain
    try:
        path = download_youtube_sync(
            url,
            yt_dlp_binary=yt_dlp_binary,
            ffmpeg_binary=ffmpeg_binary,
        )
        elapsed = round(time.perf_counter() - start, 1)
        file_size = path.stat().st_size if path.exists() else 0
        return {
            "success": True,
            "file_size": file_size,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "elapsed": elapsed,
            "file_path": str(path),
            "render_server_url": RENDER_SERVER_URL,
            "strategy_results": strategy_results,
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 1)
        return {
            "success": False,
            "error": str(exc),
            "elapsed": elapsed,
            "render_server_url": RENDER_SERVER_URL,
            "strategy_results": strategy_results,
        }


def get_video_resolution(video_path: Path, ffprobe_binary: str = "ffprobe") -> tuple[int, int]:
    """Get video resolution (width, height) using ffprobe."""
    try:
        result = subprocess.run(
            [ffprobe_binary, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=s=x:p=0", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        parts = result.stdout.strip().split("x")
        if len(parts) == 2:
            w, h = int(parts[0]), int(parts[1])
            LOGGER.info("Video resolution: %dx%d", w, h)
            return w, h
    except Exception:
        pass
    return 0, 0


def get_video_duration(video_path: Path, ffprobe_binary: str = "ffprobe") -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [ffprobe_binary, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def extract_audio(video_path: Path, ffmpeg_binary: str = "ffmpeg") -> Path | None:
    """Extract mono 16kHz WAV from video for audio analysis."""
    audio_path = video_path.parent / "audio.wav"
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-i", str(video_path), "-vn",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             "-y", str(audio_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and audio_path.exists():
            return audio_path
    except Exception as exc:
        LOGGER.warning("extract_audio failed: %s", exc)
    return None
