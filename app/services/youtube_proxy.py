# YouTube download proxy — 9-strategy robust download chain
#
# Key insight (Apr 2026): YouTube blocks datacenter IPs at the network level,
# limiting them to itag 18 (360p muxed). The n-challenge solver (EJS/deno) is
# also REQUIRED to unlock any DASH formats. Strategy order:
#   0. Decodo residential proxy (DECODO_USERNAME + DECODO_PASSWORD env vars)
#   0b. Decodo residential proxy muxed fallback
#   1. yt-dlp android_vr + DASH H.264 + EJS (720p+ if IP not blocked)
#   2. yt-dlp android_vr + DASH H.264 + EJS + proxy (if YT_DLP_PROXY set)
#   3. Render server proxy
#   4. yt-dlp android muxed (360p fallback)
#   5. yt-dlp Python lib android_vr + DASH
#   6. yt-dlp android muxed no-EJS (last resort)
#   7. Render server /extract-frames

from __future__ import annotations

import dataclasses
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)

@dataclasses.dataclass
class DownloadResult:
    """Result from download_youtube_sync with trimming metadata."""
    path: Path
    was_sectioned: bool = False  # True when --download-sections was used (timestamps start at 0)
    requested_start: float = 0
    requested_end: float = 0


# Render server URL
RENDER_SERVER_URL = os.getenv(
    "RENDER_SERVER_URL",
    "https://clipt-render-server-production.up.railway.app",
).rstrip("/")

# Decodo residential proxy — bypasses YouTube datacenter IP blocks
# Set DECODO_USERNAME + DECODO_PASSWORD env vars in Railway to enable
def _get_decodo_proxy() -> str:
    """Build Decodo residential proxy URL from env vars.

    Returns proxy URL like http://USERNAME:PASSWORD@gate.decodo.com:10001
    or empty string if not configured.
    """
    user = os.getenv("DECODO_USERNAME", "").strip()
    passwd = os.getenv("DECODO_PASSWORD", "").strip()
    if user and passwd:
        return f"http://{user}:{passwd}@gate.decodo.com:10001"
    return ""

# Optional proxy for yt-dlp — set to residential proxy or Cloudflare WARP SOCKS5
# e.g. socks5://127.0.0.1:40000 (WARP) or socks5://user:pass@proxy.example.com:1080
# Read at runtime via function (start.sh sets it dynamically after wireproxy starts)
def _get_cookie_file() -> str:
    """Return path to YouTube cookies file if it exists."""
    for path in ["/app/youtube_cookies.txt", "/app/app/youtube_cookies.txt",
                 "/data/youtube_cookies.txt", os.getenv("YOUTUBE_COOKIES_FILE", "")]:
        if path and os.path.isfile(path):
            LOGGER.info("youtube_proxy: using cookies from %s", path)
            return path
    return ""


def _get_proxy() -> str:
    return os.getenv("YT_DLP_PROXY", "").strip()

_YT_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/live/)([A-Za-z0-9_-]{11})"
)

# Matches &t=36s, &t=36, ?t=1m30s, &time_continue=36, etc.
_YT_TIMESTAMP_RE = re.compile(
    r"[?&](?:t|time_continue)=(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?",
    re.IGNORECASE,
)

# DASH H.264 format — merges video+audio into MP4, OpenCV-readable
# Explicitly filter for avc1 (H.264) to avoid VP9/AV1 which OpenCV can't decode
# Fallback chain: 720p+ H.264 DASH → any 720p+ MP4 DASH → muxed MP4 → itag 18
_DASH_H264_FORMAT = (
    "bestvideo[height=720][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
    "bestvideo[height>=720][height<=720][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
    "bestvideo[height>=720][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
    "bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/"
    "best[height>=720][ext=mp4]/"
    "best[ext=mp4]/18/best"
)
# Muxed-only format (no DASH merge needed) — safe fallback, usually 360p
_MUXED_FORMAT = "best[height<=1080][ext=mp4]/best[height<=720][ext=mp4]/best[ext=mp4]/18/best"
# Fallback for android client (muxed only, usually 360p)
_ANDROID_FORMAT = "best[height<=720][ext=mp4]/best[ext=mp4]/18/best"


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
    format_override: str | None = None,
    use_ejs: bool = True,
    proxy: str = "",
) -> bool:
    """Run yt-dlp subprocess with given client. Returns True on success."""
    if output_path.exists():
        output_path.unlink()

    fmt = format_override or _ANDROID_FORMAT
    cmd = [
        yt_dlp_binary,
        "--no-check-certificate",
        "--extractor-args", f"youtube:player_client={client}",
        "--downloader-args", "ffmpeg_i:-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
        "--format", fmt,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--socket-timeout", "60",
        "-o", str(output_path),
    ]

    # Cookie support: use YouTube cookies to bypass login/geo restrictions
    cookie_file = _get_cookie_file()
    if cookie_file:
        cmd.extend(["--cookies", cookie_file])

    # EJS: required for YouTube n-challenge solving (unlocks DASH 720p+ formats)
    # Needs deno runtime installed (see Dockerfile). Without this, only 360p muxed available.
    if use_ejs:
        cmd.extend(["--remote-components", "ejs:github"])

    # Proxy support: route through residential proxy or Cloudflare WARP
    if proxy:
        cmd.extend(["--proxy", proxy])

    if start_time > 0 or end_time > 0:
        section = f"*{start_time}-{end_time}" if end_time > 0 else f"*{start_time}-inf"
        cmd.extend(["--download-sections", section, "--force-keyframes-at-cuts"])

    cmd.append(url)

    LOGGER.info("%s: running yt-dlp client=%s format=%s proxy=%s ejs=%s",
                strategy_name, client, fmt[:60], bool(proxy), use_ejs)
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
    strategy_name: str = "Strategy 5",
    format_override: str | None = None,
    proxy: str = "",
) -> bool:
    """Use yt-dlp as Python library (no subprocess). Returns True on success."""
    try:
        import yt_dlp

        if output_path.exists():
            output_path.unlink()

        fmt = format_override or _ANDROID_FORMAT
        ydl_opts = {
            "format": fmt,
            "merge_output_format": "mp4",
            "outtmpl": str(output_path),
            "extractor_args": {"youtube": {"player_client": [client]}},
            "no_check_certificate": True,
            "socket_timeout": 30,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }

        if proxy:
            ydl_opts["proxy"] = proxy

        cookie_file = _get_cookie_file()
        if cookie_file:
            ydl_opts["cookiefile"] = cookie_file

        if (start_time > 0 or end_time > 0):
            ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
                None, [(start_time, end_time if end_time > 0 else float("inf"))]
            )

        LOGGER.info("%s: yt-dlp Python library (client=%s, proxy=%s)", strategy_name, client, bool(proxy))
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
) -> DownloadResult:
    """Synchronous 9-strategy YouTube download chain.

    Strategy order optimized for quality (720p+ when possible):
      0. Decodo residential proxy (DASH, best success rate)
      0b. Decodo residential proxy (muxed fallback)
      1. android_vr + DASH H.264 + EJS (720p+ from non-blocked IPs)
      2. android_vr + DASH H.264 + EJS + proxy (if YT_DLP_PROXY configured)
      3. Render server proxy
      4. android muxed + EJS (360p, solves n-challenge)
      5. Python lib android_vr + DASH
      6. android muxed no-EJS (bare fallback)
      7. Render server /extract-frames
    """
    LOGGER.info("youtube_proxy_sync called with URL: %s", url)

    original_url = url
    url, url_start_seconds = normalize_youtube_url(url)
    if url != original_url:
        LOGGER.info("youtube_proxy_sync: normalized %s → %s (t=%.0fs)", original_url, url, url_start_seconds)
    if url_start_seconds > 0 and start_time == 0:
        start_time = url_start_seconds

    has_time_range = start_time > 0 or end_time > 0
    LOGGER.info("youtube_proxy_sync: RENDER_SERVER_URL=%s, time=%s-%s, proxy=%s",
                RENDER_SERVER_URL, start_time, end_time, bool(_get_proxy()))
    dl_start = time.perf_counter()
    tmp_dir = Path(tempfile.mkdtemp(prefix="clipt_yt_sync_"))
    output_path = tmp_dir / "video.mp4"

    strategy_errors: list[str] = []

    def _make_result(path: Path, sectioned: bool) -> DownloadResult:
        return DownloadResult(
            path=path,
            was_sectioned=sectioned and has_time_range,
            requested_start=start_time,
            requested_end=end_time,
        )

    # Detect full game (long video) — increase timeout for strategies 1-2
    _is_long_video = (end_time - start_time > 1800) if end_time > 0 else False
    _dl_timeout = 600 if _is_long_video else 90  # 90s per-strategy for short clips

    # Overall timeout: don't let the entire chain exceed this (prevents 1400s hangs)
    # Increased for long videos: Decodo residential proxy adds latency, and
    # merging DASH video+audio for 2hr+ videos takes significant time.
    _TOTAL_TIMEOUT = 900 if _is_long_video else 300

    def _total_expired() -> bool:
        elapsed = time.perf_counter() - dl_start
        if elapsed > _TOTAL_TIMEOUT:
            LOGGER.warning("youtube_proxy_sync: total timeout %.0fs > %ds, aborting remaining strategies", elapsed, _TOTAL_TIMEOUT)
            return True
        return False

    # ── Strategy 0: Decodo residential proxy + EJS (best success rate) ──────
    # Residential proxy routes through real home IPs — YouTube never blocks these.
    # Uses subprocess yt-dlp with EJS (n-challenge solver) for 720p DASH.
    # NOTE: Must use subprocess (_yt_dlp_download) NOT Python library because
    # the Python library doesn't support --remote-components ejs:github.
    decodo_proxy = _get_decodo_proxy()
    _decodo_timeout = max(_dl_timeout, 300)  # At least 5min for Decodo (residential proxy can be slower)
    if decodo_proxy and not _total_expired():
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_decodo_timeout,
                            strategy_name="Strategy 0 (Decodo DASH+EJS)",
                            format_override=_DASH_H264_FORMAT,
                            use_ejs=True, proxy=decodo_proxy):
            elapsed = round(time.perf_counter() - dl_start, 1)
            LOGGER.info("Sync downloaded in %ss via Strategy 0 (Decodo DASH+EJS)", elapsed)
            return _make_result(output_path, sectioned=has_time_range)
        strategy_errors.append("0=decodo_dash_ejs_failed")
        # 0b: Decodo muxed fallback (no EJS needed, 360p) via Python lib
        if not _total_expired() and _yt_dlp_python_download(
                url, output_path, client="android",
                start_time=start_time, end_time=end_time,
                strategy_name="Strategy 0b (Decodo muxed)",
                format_override=_MUXED_FORMAT,
                proxy=decodo_proxy):
            elapsed = round(time.perf_counter() - dl_start, 1)
            LOGGER.info("Sync downloaded in %ss via Strategy 0b (Decodo muxed)", elapsed)
            return _make_result(output_path, sectioned=has_time_range)
        strategy_errors.append("0b=decodo_muxed_failed")
        # 0c: Decodo subprocess muxed (different yt-dlp code path)
        if not _total_expired() and _yt_dlp_download(
                url, output_path, yt_dlp_binary, ffmpeg_binary,
                client="android", start_time=start_time, end_time=end_time,
                timeout=_decodo_timeout,
                strategy_name="Strategy 0c (Decodo subprocess muxed)",
                format_override=_MUXED_FORMAT,
                use_ejs=False, proxy=decodo_proxy):
            elapsed = round(time.perf_counter() - dl_start, 1)
            LOGGER.info("Sync downloaded in %ss via Strategy 0c (Decodo subprocess muxed)", elapsed)
            return _make_result(output_path, sectioned=has_time_range)
        strategy_errors.append("0c=decodo_subprocess_muxed_failed")
    else:
        if not decodo_proxy:
            strategy_errors.append("0=no_decodo_configured")

    # ── Strategy 1: yt-dlp android_vr + DASH H.264 + EJS (best quality) ──
    # android_vr client can list 720p/1080p DASH formats when EJS solver works.
    # DASH H.264 format ensures OpenCV compatibility (no VP9/AV1).
    if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                        client="android_vr", start_time=start_time, end_time=end_time,
                        timeout=_dl_timeout, strategy_name="Strategy 1",
                        format_override=_DASH_H264_FORMAT, use_ejs=True):
        elapsed = round(time.perf_counter() - dl_start, 1)
        LOGGER.info("Sync downloaded in %ss via Strategy 1 (android_vr DASH+EJS)", elapsed)
        return _make_result(output_path, sectioned=has_time_range)
    strategy_errors.append("1=android_vr_dash_ejs_failed")

    # ── Strategy 2: Same as 1 but with proxy (if configured) ──
    # Residential proxy or Cloudflare WARP bypasses YouTube datacenter IP block.
    proxy = _get_proxy()
    if proxy and not _total_expired():
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_dl_timeout, strategy_name="Strategy 2",
                            format_override=_DASH_H264_FORMAT, use_ejs=True,
                            proxy=proxy):
            elapsed = round(time.perf_counter() - dl_start, 1)
            LOGGER.info("Sync downloaded in %ss via Strategy 2 (android_vr DASH+EJS+proxy)", elapsed)
            return _make_result(output_path, sectioned=has_time_range)
        strategy_errors.append("2=android_vr_dash_proxy_failed")
    else:
        strategy_errors.append("2=no_proxy_configured")

    # ── Strategy 3: Render server proxy ──
    # Render server receives startTime/endTime and returns pre-trimmed video.
    # Do NOT call _trim_video — that would seek to e.g. 120s in a 60s file.
    if _total_expired():
        strategy_errors.append("3=total_timeout_skip")
        strategy_errors.append("4=total_timeout_skip")
        strategy_errors.append("5=total_timeout_skip")
        strategy_errors.append("6=total_timeout_skip")
        strategy_errors.append("7=total_timeout_skip")
        raise RuntimeError(
            f"All 7 YouTube download strategies failed (sync) for: {url} "
            f"(original: {original_url}). "
            f"Errors: {', '.join(strategy_errors)}. "
            f"Check Railway logs for per-strategy errors."
        )
    with httpx.Client(timeout=httpx.Timeout(300 if _is_long_video else 60)) as client:
        if _render_server_download(url, output_path, start_time, end_time, ffmpeg_binary, client, "Strategy 3"):
            elapsed = round(time.perf_counter() - dl_start, 1)
            LOGGER.info("Sync downloaded in %ss via Strategy 3 (render server)", elapsed)
            return _make_result(output_path, sectioned=has_time_range)
    strategy_errors.append("3=render_server_failed")

    # ── Strategy 4: yt-dlp android muxed + EJS (360p but reliable) ──
    if _total_expired():
        strategy_errors.append("4=total_timeout_skip")
        strategy_errors.append("5=total_timeout_skip")
        strategy_errors.append("6=total_timeout_skip")
        strategy_errors.append("7=total_timeout_skip")
        raise RuntimeError(
            f"All 7 YouTube download strategies failed (sync) for: {url} "
            f"(original: {original_url}). "
            f"Errors: {', '.join(strategy_errors)}. "
            f"Check Railway logs for per-strategy errors."
        )
    if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                        client="android", start_time=start_time, end_time=end_time,
                        strategy_name="Strategy 4", format_override=_MUXED_FORMAT,
                        use_ejs=True):
        elapsed = round(time.perf_counter() - dl_start, 1)
        LOGGER.info("Sync downloaded in %ss via Strategy 4 (android muxed+EJS)", elapsed)
        return _make_result(output_path, sectioned=has_time_range)
    strategy_errors.append("4=android_muxed_ejs_failed")

    # ── Strategy 5: yt-dlp Python library with android_vr + DASH ──
    if _total_expired():
        strategy_errors.extend(["5=timeout_skip", "6=timeout_skip", "7=timeout_skip"])
        raise RuntimeError(
            f"All 7 YouTube download strategies failed (sync) for: {url} "
            f"(original: {original_url}). Errors: {', '.join(strategy_errors)}. "
            f"Check Railway logs for per-strategy errors."
        )
    if _yt_dlp_python_download(url, output_path, client="android_vr",
                               start_time=start_time, end_time=end_time,
                               strategy_name="Strategy 5",
                               format_override=_DASH_H264_FORMAT):
        elapsed = round(time.perf_counter() - dl_start, 1)
        LOGGER.info("Sync downloaded in %ss via Strategy 5 (Python lib android_vr)", elapsed)
        return _make_result(output_path, sectioned=has_time_range)
    strategy_errors.append("5=python_lib_android_vr_failed")

    # ── Strategy 6: yt-dlp android muxed no-EJS (bare minimum) ──
    if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                        client="android", start_time=start_time, end_time=end_time,
                        strategy_name="Strategy 6", format_override=_ANDROID_FORMAT,
                        use_ejs=False):
        elapsed = round(time.perf_counter() - dl_start, 1)
        LOGGER.info("Sync downloaded in %ss via Strategy 6 (android muxed bare)", elapsed)
        return _make_result(output_path, sectioned=has_time_range)
    strategy_errors.append("6=android_muxed_bare_failed")

    # ── Strategy 7: Render server /extract-frames (last resort) ──
    if RENDER_SERVER_URL:
        try:
            LOGGER.info("Strategy 7: render server /extract-frames (last resort)")
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
                        LOGGER.info("Sync downloaded in %ss via Strategy 7 (extract-frames)", elapsed)
                        return _make_result(output_path, sectioned=has_time_range)
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
                                LOGGER.info("Sync downloaded in %ss via Strategy 7 (extract-frames URL)", elapsed)
                                return _make_result(output_path, sectioned=has_time_range)
                LOGGER.warning("Strategy 7 failed: %d", resp.status_code)
        except Exception as exc:
            LOGGER.warning("Strategy 7 failed: %s", exc)
    strategy_errors.append("7=extract_frames_failed")

    raise RuntimeError(
        f"All 7 YouTube download strategies failed (sync) for: {url} "
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
) -> DownloadResult:
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

    # Test render server separately for diagnostics
    s_start = time.perf_counter()
    try:
        with httpx.Client(timeout=httpx.Timeout(60)) as client:
            payload = {"youtubeUrl": url}
            resp = client.post(f"{RENDER_SERVER_URL}/download-youtube", json=payload)
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "")
                data = resp.json() if "json" in ct else {}
                cloud_url = data.get("cloudinaryUrl", "")
                strategy_results.append({
                    "name": "render_server_probe",
                    "status": "success" if cloud_url else "no_url",
                    "cloudinaryUrl": cloud_url,
                    "elapsed_ms": round((time.perf_counter() - s_start) * 1000),
                })
            else:
                strategy_results.append({
                    "name": "render_server_probe",
                    "status": f"http_{resp.status_code}",
                    "elapsed_ms": round((time.perf_counter() - s_start) * 1000),
                })
    except Exception as exc:
        strategy_results.append({
            "name": "render_server_probe",
            "status": "error", "error": str(exc)[:200],
            "elapsed_ms": round((time.perf_counter() - s_start) * 1000),
        })

    # Full download test using 7-strategy chain
    try:
        dl_result = download_youtube_sync(
            url,
            yt_dlp_binary=yt_dlp_binary,
            ffmpeg_binary=ffmpeg_binary,
        )
        elapsed = round(time.perf_counter() - start, 1)
        file_size = dl_result.path.stat().st_size if dl_result.path.exists() else 0
        # Get resolution of downloaded file
        w, h = get_video_resolution(dl_result.path)
        return {
            "success": True,
            "file_size": file_size,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "resolution": f"{w}x{h}",
            "width": w,
            "height": h,
            "elapsed": elapsed,
            "file_path": str(dl_result.path),
            "was_sectioned": dl_result.was_sectioned,
            "proxy_configured": bool(_get_proxy()),
            "decodo_configured": bool(_get_decodo_proxy()),
            "render_server_url": RENDER_SERVER_URL,
            "strategy_results": strategy_results,
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 1)
        return {
            "success": False,
            "error": str(exc),
            "elapsed": elapsed,
            "proxy_configured": bool(_get_proxy()),
            "decodo_configured": bool(_get_decodo_proxy()),
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
