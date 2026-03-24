# YouTube download proxy — 5-strategy robust download chain
# Strategies tried in order until one succeeds:
# 1. Render server proxy
# 2. yt-dlp with browser spoofing
# 3. yt-dlp with android client
# 4. Cobalt API
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
    r"(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})"
)


def is_youtube_url(url: str) -> bool:
    return bool(_YT_PATTERN.search(url))


def extract_video_id(url: str) -> str | None:
    m = _YT_PATTERN.search(url)
    return m.group(1) if m else None


async def download_youtube(
    url: str,
    *,
    start_time: float = 0,
    end_time: float = 0,
    yt_dlp_binary: str = "yt-dlp",
    ffmpeg_binary: str = "ffmpeg",
    timeout: float = 120,
) -> Path:
    """Download a YouTube video using a 5-strategy chain.

    Returns path to a local .mp4 file.
    Raises RuntimeError only if ALL 5 strategies fail.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="clipt_yt_"))
    output_path = tmp_dir / "video.mp4"
    strategy_used = None

    # ── Strategy 1: Render server proxy (most reliable) ───────────────
    if RENDER_SERVER_URL:
        try:
            LOGGER.info("Strategy 1: render server proxy at %s", RENDER_SERVER_URL)
            async with httpx.AsyncClient(timeout=httpx.Timeout(90)) as client:
                payload: dict = {"url": url}
                if start_time > 0:
                    payload["startTime"] = start_time
                if end_time > 0:
                    payload["endTime"] = end_time

                resp = await client.post(
                    f"{RENDER_SERVER_URL}/download-youtube",
                    json=payload,
                )

                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    if "video" in content_type or "octet-stream" in content_type:
                        output_path.write_bytes(resp.content)
                        LOGGER.info(
                            "YouTube download succeeded via Strategy 1 (render server): %d bytes",
                            len(resp.content),
                        )
                        return output_path

                    # JSON response with download URL
                    data = resp.json()
                    download_url = data.get("downloadUrl") or data.get("url") or data.get("videoUrl")
                    if download_url:
                        dl_resp = await client.get(download_url)
                        if dl_resp.status_code == 200:
                            output_path.write_bytes(dl_resp.content)
                            LOGGER.info(
                                "YouTube download succeeded via Strategy 1 (render server URL): %d bytes",
                                len(dl_resp.content),
                            )
                            return output_path

                LOGGER.warning("Strategy 1 failed: render server returned %d", resp.status_code)
        except Exception as exc:
            LOGGER.warning("Strategy 1 failed: %s — trying next", exc)

    # ── Strategy 2: yt-dlp with cookies and browser spoofing ──────────
    try:
        LOGGER.info("Strategy 2: yt-dlp with browser spoofing")
        cmd = [
            yt_dlp_binary,
            "--no-check-certificate",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "--add-header", "Accept-Language:en-US,en;q=0.9",
            "--format", "best[height<=720]/best",
            "--no-playlist",
            "-o", str(output_path),
        ]
        if start_time > 0 or end_time > 0:
            section = f"*{start_time}-{end_time}" if end_time > 0 else f"*{start_time}-inf"
            cmd.extend(["--download-sections", section])
            cmd.extend(["--force-keyframes-at-cuts"])
        cmd.append(url)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            LOGGER.info(
                "YouTube download succeeded via Strategy 2 (yt-dlp browser spoof): %d bytes",
                output_path.stat().st_size,
            )
            return output_path
        LOGGER.warning("Strategy 2 failed: %s — trying next", result.stderr[:300])
    except Exception as exc:
        LOGGER.warning("Strategy 2 failed: %s — trying next", exc)

    # ── Strategy 3: yt-dlp with android client ────────────────────────
    try:
        LOGGER.info("Strategy 3: yt-dlp with android client extractor")
        # Remove any partial file from strategy 2
        if output_path.exists():
            output_path.unlink()
        cmd = [
            yt_dlp_binary,
            "--no-check-certificate",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "--add-header", "Accept-Language:en-US,en;q=0.9",
            "--extractor-args", "youtube:player_client=android",
            "--format", "worst",
            "--no-playlist",
            "-o", str(output_path),
        ]
        if start_time > 0 or end_time > 0:
            section = f"*{start_time}-{end_time}" if end_time > 0 else f"*{start_time}-inf"
            cmd.extend(["--download-sections", section])
            cmd.extend(["--force-keyframes-at-cuts"])
        cmd.append(url)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            LOGGER.info(
                "YouTube download succeeded via Strategy 3 (yt-dlp android): %d bytes",
                output_path.stat().st_size,
            )
            return output_path
        LOGGER.warning("Strategy 3 failed: %s — trying next", result.stderr[:300])
    except Exception as exc:
        LOGGER.warning("Strategy 3 failed: %s — trying next", exc)

    # ── Strategy 4: Cobalt API ────────────────────────────────────────
    try:
        LOGGER.info("Strategy 4: Cobalt API")
        async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
            cobalt_resp = await client.post(
                "https://co.wuk.sh/api/json",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={"url": url, "vQuality": "480"},
            )
            if cobalt_resp.status_code == 200:
                cobalt_data = cobalt_resp.json()
                cobalt_url = cobalt_data.get("url")
                if cobalt_url:
                    # Download the video from cobalt's URL
                    dl_resp = await client.get(cobalt_url, follow_redirects=True)
                    if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                        if output_path.exists():
                            output_path.unlink()
                        output_path.write_bytes(dl_resp.content)
                        LOGGER.info(
                            "YouTube download succeeded via Strategy 4 (Cobalt): %d bytes",
                            len(dl_resp.content),
                        )
                        return output_path
            LOGGER.warning("Strategy 4 failed: Cobalt returned %d — trying next", cobalt_resp.status_code)
    except Exception as exc:
        LOGGER.warning("Strategy 4 failed: %s — trying next", exc)

    # ── Strategy 5: Render server /extract-frames (last resort) ───────
    if RENDER_SERVER_URL:
        try:
            LOGGER.info("Strategy 5: render server /extract-frames (last resort)")
            async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
                payload = {"url": url}
                if start_time > 0:
                    payload["startTime"] = start_time
                if end_time > 0:
                    payload["endTime"] = end_time

                resp = await client.post(
                    f"{RENDER_SERVER_URL}/extract-frames",
                    json=payload,
                )
                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    if "video" in content_type or "octet-stream" in content_type:
                        if output_path.exists():
                            output_path.unlink()
                        output_path.write_bytes(resp.content)
                        LOGGER.info(
                            "YouTube download succeeded via Strategy 5 (extract-frames): %d bytes",
                            len(resp.content),
                        )
                        return output_path

                    # May return JSON with a video URL
                    data = resp.json()
                    download_url = data.get("downloadUrl") or data.get("url") or data.get("videoUrl")
                    if download_url:
                        dl_resp = await client.get(download_url)
                        if dl_resp.status_code == 200:
                            if output_path.exists():
                                output_path.unlink()
                            output_path.write_bytes(dl_resp.content)
                            LOGGER.info(
                                "YouTube download succeeded via Strategy 5 (extract-frames URL): %d bytes",
                                len(dl_resp.content),
                            )
                            return output_path

                LOGGER.warning("Strategy 5 failed: render server returned %d", resp.status_code)
        except Exception as exc:
            LOGGER.warning("Strategy 5 failed: %s", exc)

    raise RuntimeError(
        f"All 5 YouTube download strategies failed for: {url}"
    )


async def test_youtube_download(
    url: str,
    yt_dlp_binary: str = "yt-dlp",
    ffmpeg_binary: str = "ffmpeg",
) -> dict:
    """Test YouTube download and report which strategy worked."""
    start = time.perf_counter()
    try:
        path = await download_youtube(
            url,
            yt_dlp_binary=yt_dlp_binary,
            ffmpeg_binary=ffmpeg_binary,
        )
        elapsed = round(time.perf_counter() - start, 1)
        file_size = path.stat().st_size if path.exists() else 0
        # Parse strategy from logs (read last log entry)
        return {
            "success": True,
            "file_size": file_size,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "elapsed": elapsed,
            "file_path": str(path),
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 1)
        return {
            "success": False,
            "error": str(exc),
            "elapsed": elapsed,
        }


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
