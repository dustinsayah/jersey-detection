# YouTube download proxy — routes through render server when local yt-dlp fails

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)

# Render server has working YouTube download with cobalt/pytubefix/tor fallbacks
RENDER_SERVER_URL = os.getenv("RENDER_SERVER_URL", "").rstrip("/")

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
    """Download a YouTube video, returning path to a local .mp4 file.

    Strategy:
    1. Try render server proxy (most reliable for Railway)
    2. Fall back to local yt-dlp
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="clipt_yt_"))
    output_path = tmp_dir / "video.mp4"

    # ── Strategy 1: Render server proxy ──────────────────────────────────
    if RENDER_SERVER_URL:
        try:
            LOGGER.info("youtube_proxy: trying render server at %s", RENDER_SERVER_URL)
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
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
                        # Binary video data returned directly
                        output_path.write_bytes(resp.content)
                        LOGGER.info("youtube_proxy: render server returned %d bytes", len(resp.content))
                        return output_path

                    # JSON response with download URL
                    data = resp.json()
                    download_url = data.get("downloadUrl") or data.get("url") or data.get("videoUrl")
                    if download_url:
                        LOGGER.info("youtube_proxy: downloading from render server URL")
                        dl_resp = await client.get(download_url)
                        if dl_resp.status_code == 200:
                            output_path.write_bytes(dl_resp.content)
                            return output_path

                LOGGER.warning("youtube_proxy: render server returned %d", resp.status_code)
        except Exception as exc:
            LOGGER.warning("youtube_proxy: render server failed: %s", exc)

    # ── Strategy 2: Local yt-dlp ─────────────────────────────────────────
    try:
        LOGGER.info("youtube_proxy: trying local yt-dlp")
        cmd = [
            yt_dlp_binary,
            "-f", "best[ext=mp4]/best",
            "--no-playlist",
            "-o", str(output_path),
        ]
        if start_time > 0 or end_time > 0:
            section = f"*{start_time}-{end_time}" if end_time > 0 else f"*{start_time}-inf"
            cmd.extend(["--download-sections", section])
            cmd.extend(["--force-keyframes-at-cuts"])
        cmd.append(url)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and output_path.exists():
            LOGGER.info("youtube_proxy: local yt-dlp succeeded, %d bytes", output_path.stat().st_size)
            return output_path

        LOGGER.warning("youtube_proxy: local yt-dlp failed: %s", result.stderr[:300])
    except Exception as exc:
        LOGGER.warning("youtube_proxy: local yt-dlp error: %s", exc)

    raise RuntimeError(f"Failed to download YouTube video: {url}")


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
