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

# Matches &t=36s, &t=36, ?t=1m30s, &time_continue=36, etc.
_YT_TIMESTAMP_RE = re.compile(
    r"[?&](?:t|time_continue)=(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?",
    re.IGNORECASE,
)


def is_youtube_url(url: str) -> bool:
    return bool(_YT_PATTERN.search(url))


def extract_video_id(url: str) -> str | None:
    m = _YT_PATTERN.search(url)
    return m.group(1) if m else None


def normalize_youtube_url(url: str) -> tuple[str, float]:
    """Normalize a YouTube URL to a clean canonical form.

    Strips tracking/timestamp params (&t=, &list=, &index=, &feature=,
    &time_continue=, &si=, etc.) that can confuse download strategies.

    Returns:
        (clean_url, extracted_start_seconds)
        - clean_url: ``https://www.youtube.com/watch?v=<ID>``
        - extracted_start_seconds: seconds parsed from ``&t=`` / ``&time_continue=``,
          or 0.0 if not present.
    """
    video_id = extract_video_id(url)
    if not video_id:
        # Not a recognized YouTube URL — return as-is
        return url, 0.0

    # Parse timestamp from URL (e.g. &t=36s, &t=1m30s, &time_continue=90)
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
            # Replace original with trimmed
            input_path.unlink()
            trimmed_path.rename(input_path)
            return input_path
        LOGGER.warning("ffmpeg trim failed (returncode=%d), using untrimmed: %s", result.returncode, result.stderr[:200])
    except Exception as exc:
        LOGGER.warning("ffmpeg trim failed, using untrimmed: %s", exc)
    return input_path


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

    .. deprecated::
        This async function calls blocking subprocess.run() on the event loop
        thread. Use ``download_youtube_sync`` via ``run_in_threadpool`` instead.

    Returns path to a local .mp4 file.
    Raises RuntimeError only if ALL 5 strategies fail.
    """
    LOGGER.info("youtube_proxy called with URL: %s", url)

    # ── Normalize URL: strip &t=, &list=, &feature=, etc. ──
    original_url = url
    url, url_start_seconds = normalize_youtube_url(url)
    if url != original_url:
        LOGGER.info("youtube_proxy: normalized URL %s → %s (t=%.0fs)", original_url, url, url_start_seconds)
    # If caller didn't specify start_time but URL had &t=, use the URL timestamp
    if url_start_seconds > 0 and start_time == 0:
        start_time = url_start_seconds
        LOGGER.info("youtube_proxy: using URL timestamp as start_time=%.0f", start_time)

    LOGGER.info("youtube_proxy: RENDER_SERVER_URL=%s, time_range=%s-%s", RENDER_SERVER_URL, start_time, end_time)
    dl_start = time.perf_counter()
    tmp_dir = Path(tempfile.mkdtemp(prefix="clipt_yt_"))
    output_path = tmp_dir / "video.mp4"
    strategy_used = None

    # ── Strategy 1: Render server proxy (most reliable) ───────────────
    # Render server expects {"youtubeUrl": "..."} and returns {"cloudinaryUrl": "...", "duration": N}
    if RENDER_SERVER_URL:
        try:
            LOGGER.info("youtube_proxy: Strategy 1 — render server at %s", RENDER_SERVER_URL)
            async with httpx.AsyncClient(timeout=httpx.Timeout(90)) as client:
                payload: dict = {"youtubeUrl": url}
                if start_time > 0:
                    payload["startTime"] = start_time
                if end_time > 0:
                    payload["endTime"] = end_time

                LOGGER.info("youtube_proxy: POST %s/download-youtube payload=%s", RENDER_SERVER_URL, payload)
                resp = await client.post(
                    f"{RENDER_SERVER_URL}/download-youtube",
                    json=payload,
                )

                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    if "video" in content_type or "octet-stream" in content_type:
                        output_path.write_bytes(resp.content)
                        elapsed = round(time.perf_counter() - dl_start, 1)
                        file_mb = round(len(resp.content) / 1024 / 1024, 1)
                        LOGGER.info("Downloaded: %sMB in %ss via Strategy 1 (render server bytes)", file_mb, elapsed)
                        if start_time > 0 or end_time > 0:
                            output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                        return output_path

                    # JSON response — render server returns cloudinaryUrl
                    ct = resp.headers.get("content-type", "")
                    if "json" not in ct and "text" not in ct:
                        raise ValueError(f"Strategy 1: unexpected content-type '{ct}', skipping JSON parse")
                    data = resp.json()
                    LOGGER.info("youtube_proxy: Strategy 1 response: %s", data)
                    download_url = (
                        data.get("cloudinaryUrl")
                        or data.get("downloadUrl")
                        or data.get("url")
                        or data.get("videoUrl")
                    )
                    if download_url:
                        LOGGER.info("youtube_proxy: downloading from %s", download_url[:80])
                        dl_resp = await client.get(download_url, follow_redirects=True)
                        if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                            output_path.write_bytes(dl_resp.content)
                            elapsed = round(time.perf_counter() - dl_start, 1)
                            file_mb = round(len(dl_resp.content) / 1024 / 1024, 1)
                            LOGGER.info("Downloaded: %sMB in %ss via Strategy 1 (render→cloudinary)", file_mb, elapsed)
                            if start_time > 0 or end_time > 0:
                                output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                            return output_path
                        LOGGER.warning("youtube_proxy: cloudinary download failed: %d, %d bytes",
                                       dl_resp.status_code, len(dl_resp.content))

                LOGGER.warning("Strategy 1 failed: render server returned %d: %s",
                               resp.status_code, resp.text[:200])
        except Exception as exc:
            LOGGER.warning("Strategy 1 failed: %s — trying next", exc)

    # ── Strategy 2: yt-dlp with cookies and browser spoofing ──────────
    try:
        LOGGER.info("Strategy 2: yt-dlp with browser spoofing + remote-components")
        cmd = [
            yt_dlp_binary,
            "--no-check-certificate",
            "--remote-components", "ejs:github",
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

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            elapsed = round(time.perf_counter() - dl_start, 1)
            file_mb = round(output_path.stat().st_size / 1024 / 1024, 1)
            LOGGER.info("Downloaded: %sMB in %ss via Strategy 2 (yt-dlp browser spoof)", file_mb, elapsed)
            if start_time > 0 or end_time > 0:
                output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
            return output_path
        LOGGER.warning("Strategy 2 failed: %s — trying next", result.stderr[:300])
    except Exception as exc:
        LOGGER.warning("Strategy 2 failed: %s — trying next", exc)

    # ── Strategy 3: yt-dlp with android client ────────────────────────
    try:
        LOGGER.info("Strategy 3: yt-dlp with android client + remote-components")
        # Remove any partial file from strategy 2
        if output_path.exists():
            output_path.unlink()
        cmd = [
            yt_dlp_binary,
            "--no-check-certificate",
            "--remote-components", "ejs:github",
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

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            elapsed = round(time.perf_counter() - dl_start, 1)
            file_mb = round(output_path.stat().st_size / 1024 / 1024, 1)
            LOGGER.info("Downloaded: %sMB in %ss via Strategy 3 (yt-dlp android)", file_mb, elapsed)
            if start_time > 0 or end_time > 0:
                output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
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
                        elapsed = round(time.perf_counter() - dl_start, 1)
                        file_mb = round(len(dl_resp.content) / 1024 / 1024, 1)
                        LOGGER.info("Downloaded: %sMB in %ss via Strategy 4 (Cobalt)", file_mb, elapsed)
                        if start_time > 0 or end_time > 0:
                            output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                        return output_path
            LOGGER.warning("Strategy 4 failed: Cobalt returned %d — trying next", cobalt_resp.status_code)
    except Exception as exc:
        LOGGER.warning("Strategy 4 failed: %s — trying next", exc)

    # ── Strategy 5: Render server /extract-frames (last resort) ───────
    if RENDER_SERVER_URL:
        try:
            LOGGER.info("youtube_proxy: Strategy 5 — render server /extract-frames (last resort)")
            async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
                payload = {"youtubeUrl": url}
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
                        elapsed = round(time.perf_counter() - dl_start, 1)
                        file_mb = round(len(resp.content) / 1024 / 1024, 1)
                        LOGGER.info("Downloaded: %sMB in %ss via Strategy 5 (extract-frames)", file_mb, elapsed)
                        if start_time > 0 or end_time > 0:
                            output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                        return output_path

                    # May return JSON with a video URL
                    ct = resp.headers.get("content-type", "")
                    if "json" not in ct and "text" not in ct:
                        raise ValueError(f"Strategy 5: unexpected content-type '{ct}', skipping JSON parse")
                    data = resp.json()
                    download_url = (
                        data.get("cloudinaryUrl")
                        or data.get("downloadUrl")
                        or data.get("url")
                        or data.get("videoUrl")
                    )
                    if download_url:
                        dl_resp = await client.get(download_url, follow_redirects=True)
                        if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                            if output_path.exists():
                                output_path.unlink()
                            output_path.write_bytes(dl_resp.content)
                            elapsed = round(time.perf_counter() - dl_start, 1)
                            file_mb = round(len(dl_resp.content) / 1024 / 1024, 1)
                            LOGGER.info("Downloaded: %sMB in %ss via Strategy 5 (extract-frames URL)", file_mb, elapsed)
                            if start_time > 0 or end_time > 0:
                                output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                            return output_path

                LOGGER.warning("Strategy 5 failed: render server returned %d", resp.status_code)
        except Exception as exc:
            LOGGER.warning("Strategy 5 failed: %s", exc)

    raise RuntimeError(
        f"All 5 YouTube download strategies failed for: {url} "
        f"(original: {original_url}). "
        f"Strategies: 1=render_server({RENDER_SERVER_URL or 'not_set'}), "
        f"2=yt-dlp_browser, 3=yt-dlp_android, 4=cobalt, 5=render_extract_frames. "
        f"Check Railway logs for per-strategy errors."
    )


def download_youtube_sync(
    url: str,
    *,
    start_time: float = 0,
    end_time: float = 0,
    yt_dlp_binary: str = "yt-dlp",
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Synchronous version of download_youtube for use from threadpool workers.

    Same 5-strategy chain but uses httpx.Client (sync) instead of AsyncClient.
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

    # ── Strategy 1: Render server proxy ──
    if RENDER_SERVER_URL:
        try:
            LOGGER.info("youtube_proxy_sync: Strategy 1 — render server")
            with httpx.Client(timeout=httpx.Timeout(90)) as client:
                payload: dict = {"youtubeUrl": url}
                if start_time > 0:
                    payload["startTime"] = start_time
                if end_time > 0:
                    payload["endTime"] = end_time
                resp = client.post(f"{RENDER_SERVER_URL}/download-youtube", json=payload)
                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    if "video" in content_type or "octet-stream" in content_type:
                        output_path.write_bytes(resp.content)
                        elapsed = round(time.perf_counter() - dl_start, 1)
                        LOGGER.info("Sync downloaded: %sMB in %ss via Strategy 1 (bytes)", round(len(resp.content)/1024/1024, 1), elapsed)
                        if start_time > 0 or end_time > 0:
                            output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                        return output_path
                    ct = resp.headers.get("content-type", "")
                    if "json" not in ct and "text" not in ct:
                        raise ValueError(f"Sync Strategy 1: unexpected content-type '{ct}', skipping JSON parse")
                    data = resp.json()
                    download_url = data.get("cloudinaryUrl") or data.get("downloadUrl") or data.get("url") or data.get("videoUrl")
                    if download_url:
                        dl_resp = client.get(download_url, follow_redirects=True)
                        if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                            output_path.write_bytes(dl_resp.content)
                            elapsed = round(time.perf_counter() - dl_start, 1)
                            LOGGER.info("Sync downloaded: %sMB in %ss via Strategy 1 (cloudinary)", round(len(dl_resp.content)/1024/1024, 1), elapsed)
                            if start_time > 0 or end_time > 0:
                                output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                            return output_path
                LOGGER.warning("Sync Strategy 1 failed: %d: %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            LOGGER.warning("Sync Strategy 1 failed: %s", exc)

    # ── Strategy 2: yt-dlp with browser spoofing ──
    try:
        LOGGER.info("youtube_proxy_sync: Strategy 2 — yt-dlp browser spoof + remote-components")
        cmd = [
            yt_dlp_binary, "--no-check-certificate",
            "--remote-components", "ejs:github",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "--add-header", "Accept-Language:en-US,en;q=0.9",
            "--format", "best[height<=720]/best",
            "--no-playlist", "-o", str(output_path),
        ]
        if start_time > 0 or end_time > 0:
            section = f"*{start_time}-{end_time}" if end_time > 0 else f"*{start_time}-inf"
            cmd.extend(["--download-sections", section, "--force-keyframes-at-cuts"])
        cmd.append(url)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            elapsed = round(time.perf_counter() - dl_start, 1)
            LOGGER.info("Sync downloaded: %sMB in %ss via Strategy 2", round(output_path.stat().st_size/1024/1024, 1), elapsed)
            if start_time > 0 or end_time > 0:
                output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
            return output_path
        LOGGER.warning("Sync Strategy 2 failed: %s", result.stderr[:300])
    except Exception as exc:
        LOGGER.warning("Sync Strategy 2 failed: %s", exc)

    # ── Strategy 3: yt-dlp with android client ──
    try:
        LOGGER.info("youtube_proxy_sync: Strategy 3 — yt-dlp android + remote-components")
        if output_path.exists():
            output_path.unlink()
        cmd = [
            yt_dlp_binary, "--no-check-certificate",
            "--remote-components", "ejs:github",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "--add-header", "Accept-Language:en-US,en;q=0.9",
            "--extractor-args", "youtube:player_client=android",
            "--format", "worst", "--no-playlist", "-o", str(output_path),
        ]
        if start_time > 0 or end_time > 0:
            section = f"*{start_time}-{end_time}" if end_time > 0 else f"*{start_time}-inf"
            cmd.extend(["--download-sections", section, "--force-keyframes-at-cuts"])
        cmd.append(url)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            elapsed = round(time.perf_counter() - dl_start, 1)
            LOGGER.info("Sync downloaded: %sMB in %ss via Strategy 3", round(output_path.stat().st_size/1024/1024, 1), elapsed)
            if start_time > 0 or end_time > 0:
                output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
            return output_path
        LOGGER.warning("Sync Strategy 3 failed: %s", result.stderr[:300])
    except Exception as exc:
        LOGGER.warning("Sync Strategy 3 failed: %s", exc)

    # ── Strategy 4: Cobalt API ──
    try:
        LOGGER.info("youtube_proxy_sync: Strategy 4 — Cobalt API")
        with httpx.Client(timeout=httpx.Timeout(30)) as client:
            cobalt_resp = client.post(
                "https://co.wuk.sh/api/json",
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                json={"url": url, "vQuality": "480"},
            )
            if cobalt_resp.status_code == 200:
                cobalt_data = cobalt_resp.json()
                cobalt_url = cobalt_data.get("url")
                if cobalt_url:
                    dl_resp = client.get(cobalt_url, follow_redirects=True)
                    if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                        if output_path.exists():
                            output_path.unlink()
                        output_path.write_bytes(dl_resp.content)
                        elapsed = round(time.perf_counter() - dl_start, 1)
                        LOGGER.info("Sync downloaded: %sMB in %ss via Strategy 4", round(len(dl_resp.content)/1024/1024, 1), elapsed)
                        if start_time > 0 or end_time > 0:
                            output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                        return output_path
            LOGGER.warning("Sync Strategy 4 failed: Cobalt %d", cobalt_resp.status_code)
    except Exception as exc:
        LOGGER.warning("Sync Strategy 4 failed: %s", exc)

    # ── Strategy 5: Render server /extract-frames ──
    if RENDER_SERVER_URL:
        try:
            LOGGER.info("youtube_proxy_sync: Strategy 5 — extract-frames")
            with httpx.Client(timeout=httpx.Timeout(120)) as client:
                payload = {"youtubeUrl": url}
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
                        LOGGER.info("Sync downloaded: %sMB in %ss via Strategy 5", round(len(resp.content)/1024/1024, 1), elapsed)
                        if start_time > 0 or end_time > 0:
                            output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                        return output_path
                    ct = resp.headers.get("content-type", "")
                    if "json" not in ct and "text" not in ct:
                        raise ValueError(f"Sync Strategy 5: unexpected content-type '{ct}', skipping JSON parse")
                    data = resp.json()
                    download_url = data.get("cloudinaryUrl") or data.get("downloadUrl") or data.get("url") or data.get("videoUrl")
                    if download_url:
                        dl_resp = client.get(download_url, follow_redirects=True)
                        if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                            if output_path.exists():
                                output_path.unlink()
                            output_path.write_bytes(dl_resp.content)
                            elapsed = round(time.perf_counter() - dl_start, 1)
                            LOGGER.info("Sync downloaded: %sMB in %ss via Strategy 5 (URL)", round(len(dl_resp.content)/1024/1024, 1), elapsed)
                            if start_time > 0 or end_time > 0:
                                output_path = _trim_video(output_path, start_time, end_time, ffmpeg_binary)
                            return output_path
                LOGGER.warning("Sync Strategy 5 failed: %d", resp.status_code)
        except Exception as exc:
            LOGGER.warning("Sync Strategy 5 failed: %s", exc)

    raise RuntimeError(
        f"All 5 YouTube download strategies failed (sync) for: {url} "
        f"(original: {original_url}). "
        f"Strategies: 1=render_server({RENDER_SERVER_URL or 'not_set'}), "
        f"2=yt-dlp_browser, 3=yt-dlp_android, 4=cobalt, 5=render_extract_frames. "
        f"Check Railway logs for per-strategy errors."
    )


async def test_youtube_download(
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
        async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
            payload = {"youtubeUrl": url}
            resp = await client.post(f"{RENDER_SERVER_URL}/download-youtube", json=payload)
            if resp.status_code == 200:
                data = resp.json() if "json" in resp.headers.get("content-type", "") else {}
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
                        "status": "no_url", "response": str(resp.text[:200]),
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

    # Full download test
    try:
        path = await download_youtube(
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


def test_youtube_download_sync(
    url: str,
    yt_dlp_binary: str = "yt-dlp",
    ffmpeg_binary: str = "ffmpeg",
) -> dict:
    """Synchronous version of test_youtube_download for use from threadpool."""
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
                        "status": "no_url", "response": str(resp.text[:200]),
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
