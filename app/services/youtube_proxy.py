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
import random
import re
import string
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
    strategy_used: str = "unknown"  # Which download strategy succeeded


# Render server URL
RENDER_SERVER_URL = os.getenv(
    "RENDER_SERVER_URL",
    "https://clipt-render-server-production.up.railway.app",
).rstrip("/")

# Decodo residential proxy — bypasses YouTube datacenter IP blocks
# Set DECODO_USERNAME + DECODO_PASSWORD env vars in Railway to enable
#
# Decodo has two proxy endpoints:
#   - Port 10001: Basic rotating proxy (raw username:password)
#   - Port 7000: Advanced endpoint with parameters (session, country, etc.)
#     Requires "user-" prefix: user-USERNAME-session-ID-sessionduration-MIN
#
# See: https://help.decodo.com/docs/residential-proxy-custom-sticky-sessions
def _get_decodo_proxy(sticky: bool = False) -> str:
    """Build Decodo residential proxy URL from env vars.

    Args:
        sticky: If True, use port 7000 with a random session ID so that
                all requests through this proxy URL hit the SAME residential IP.
                Critical for DASH downloads where yt-dlp extracts stream URLs
                bound to one IP and ffmpeg must download from the same IP.
                Session duration: 10 minutes (enough for any single download).

    Returns proxy URL or empty string if not configured.
    """
    user = os.getenv("DECODO_USERNAME", "").strip()
    passwd = os.getenv("DECODO_PASSWORD", "").strip()
    if user and passwd:
        if sticky:
            session_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
            # Decodo advanced format: user-USERNAME-session-ID-sessionduration-MIN
            # Port 7000 with "user-" prefix — supports session parameters
            return f"http://user-{user}-session-{session_id}-sessionduration-10:{passwd}@gate.decodo.com:7000"
        # Port 10001: basic rotating proxy (no session parameters)
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
    skip_sections: bool = False,
    errors_detail: dict | None = None,
) -> bool:
    """Run yt-dlp subprocess with given client. Returns True on success.

    Args:
        skip_sections: If True, download the full video without --download-sections.
                       This is needed when DASH+proxy fails with sectioning.
                       The caller is responsible for trimming afterward.
    """
    if output_path.exists():
        output_path.unlink()

    fmt = format_override or _ANDROID_FORMAT
    cmd = [
        yt_dlp_binary,
        "--no-check-certificate",
        "--extractor-args", f"youtube:player_client={client}",
        "--downloader-args", "ffmpeg_i:-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 10 -reconnect_on_network_error 1",
        "--format", fmt,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--socket-timeout", "60",
        "--retries", "3",
        "--fragment-retries", "5",
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

    if not skip_sections and (start_time > 0 or end_time > 0):
        section = f"*{start_time}-{end_time}" if end_time > 0 else f"*{start_time}-inf"
        cmd.extend(["--download-sections", section, "--force-keyframes-at-cuts"])

    cmd.append(url)

    LOGGER.info("%s: running yt-dlp client=%s format=%s proxy=%s ejs=%s skip_sections=%s",
                strategy_name, client, fmt[:60], bool(proxy), use_ejs, skip_sections)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        LOGGER.warning("%s: TIMED OUT after %ds (client=%s)", strategy_name, timeout, client)
        if errors_detail is not None:
            errors_detail[strategy_name] = f"TIMEOUT after {timeout}s"
        return False

    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
        file_mb = round(output_path.stat().st_size / 1024 / 1024, 1)
        LOGGER.info("%s: SUCCESS — %sMB downloaded (client=%s)", strategy_name, file_mb, client)
        # If we skipped sections but had a time range, trim with ffmpeg -c copy
        if skip_sections and (start_time > 0 or end_time > 0):
            LOGGER.info("%s: post-download trim %.0fs-%.0fs", strategy_name, start_time, end_time)
            _trim_video(output_path, start_time, end_time, ffmpeg_binary)
        return True

    err = result.stderr[:300] if result.stderr else "no stderr"
    LOGGER.warning("%s: FAILED (client=%s): %s", strategy_name, client, err)
    if errors_detail is not None:
        errors_detail[strategy_name] = err[:120]
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
    timeout: int = 90,
    errors_detail: dict | None = None,
) -> bool:
    """Use yt-dlp as Python library (no subprocess). Returns True on success.

    Args:
        timeout: Hard timeout in seconds (default 90). Uses a background thread
                 to enforce the limit since the Python library has no built-in timeout.
    """
    import concurrent.futures
    import threading

    def _do_download() -> bool:
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

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_download)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        LOGGER.warning("%s: TIMED OUT after %ds (Python lib, client=%s)", strategy_name, timeout, client)
        if errors_detail is not None:
            errors_detail[strategy_name] = f"TIMEOUT after {timeout}s"
        return False
    except Exception as exc:
        LOGGER.warning("%s: FAILED (Python lib, client=%s): %s", strategy_name, client, str(exc)[:200])
        if errors_detail is not None:
            errors_detail[strategy_name] = str(exc)[:120]
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

        LOGGER.warning("%s: FAILED — render server returned %d: %s (content-type=%s)",
                       strategy_name, resp.status_code, resp.text[:300], resp.headers.get("content-type", ""))
    except httpx.TimeoutException as exc:
        LOGGER.warning("%s: TIMEOUT — %s", strategy_name, str(exc)[:200])
    except Exception as exc:
        LOGGER.warning("%s: FAILED — %s: %s", strategy_name, type(exc).__name__, str(exc)[:200])

    return False


def download_youtube_sync(
    url: str,
    *,
    start_time: float = 0,
    end_time: float = 0,
    yt_dlp_binary: str = "yt-dlp",
    ffmpeg_binary: str = "ffmpeg",
) -> DownloadResult:
    """Synchronous YouTube download chain — bulletproof strategy cascade.

    Every strategy has its own try/except. Failed strategies NEVER crash the chain.
    The chain ALWAYS tries all strategies before giving up.
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
    _MIN_FILE_SIZE = 100_000  # 0.1MB — reject tiny/corrupt downloads

    def _make_result(path: Path, sectioned: bool, strategy: str = "unknown") -> DownloadResult:
        return DownloadResult(
            path=path,
            was_sectioned=sectioned and has_time_range,
            requested_start=start_time,
            requested_end=end_time,
            strategy_used=strategy,
        )

    # Detect full game (long video) — increase timeouts
    _is_long_video = (end_time - start_time > 1800) if end_time > 0 else False

    # Per-strategy timeout: 120s short clips, 900s full games
    _strategy_timeout = 900 if _is_long_video else 120

    # Decodo sectioned timeout: only downloads the requested range
    _decodo_sections_timeout = 900 if _is_long_video else 180

    # Decodo FULL download timeout: downloads the ENTIRE video then trims.
    # Must be generous even for short time ranges since the source video
    # may be 2+ hours. 600s = enough for ~2GB at 3MB/s residential speed.
    _decodo_full_timeout = 900 if _is_long_video else 600

    # Overall timeout: 1500s full games, 900s short clips (allows ~7 strategies at 120s each)
    _TOTAL_TIMEOUT = 1500 if _is_long_video else 900

    def _total_expired() -> bool:
        elapsed = time.perf_counter() - dl_start
        if elapsed > _TOTAL_TIMEOUT:
            LOGGER.warning("youtube_proxy_sync: total timeout %.0fs > %ds", elapsed, _TOTAL_TIMEOUT)
            return True
        return False

    def _file_valid() -> bool:
        """Check output file exists and is large enough."""
        return output_path.exists() and output_path.stat().st_size > _MIN_FILE_SIZE

    # ── Decodo proxy setup + health check ────────────────────────────────
    # Use non-sticky proxy for health check, sticky for actual downloads
    decodo_proxy = _get_decodo_proxy(sticky=False)
    _decodo_healthy = False
    if decodo_proxy:
        try:
            _probe_resp = httpx.head(
                "https://www.youtube.com/",
                proxy=decodo_proxy,
                timeout=httpx.Timeout(15),
                follow_redirects=True,
            )
            _decodo_healthy = _probe_resp.status_code < 400
            LOGGER.info("Decodo YouTube probe: %s (status=%d)", "OK" if _decodo_healthy else "FAIL", _probe_resp.status_code)
        except Exception as exc:
            LOGGER.warning("Decodo YouTube probe failed: %s", str(exc)[:100])
            _decodo_healthy = False

    proxy = _get_proxy()

    # Shared error detail dict — populated by helpers for diagnostics
    _errors_detail: dict[str, str] = {}

    # ── Strategy functions — each returns DownloadResult | None ───────────

    def _s0_decodo_sections() -> DownloadResult | None:
        """Decodo DASH+EJS with --download-sections + sticky session (ideal: fast + 720p).

        Sticky session ensures yt-dlp extraction and ffmpeg DASH merge use the
        same residential IP. Without this, IP rotation causes 'End of file' errors
        when ffmpeg tries to download DASH segments from a different IP than the
        one yt-dlp used to extract the stream URLs.
        """
        if not (decodo_proxy and _decodo_healthy):
            return None
        sticky_proxy = _get_decodo_proxy(sticky=True)
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_decodo_sections_timeout,
                            strategy_name="Strategy 0 (Decodo DASH+EJS+sections+sticky)",
                            format_override=_DASH_H264_FORMAT,
                            use_ejs=True, proxy=sticky_proxy, errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="decodo_dash_ejs_sectioned")
        return None

    def _s0a_decodo_full_trim() -> DownloadResult | None:
        """Decodo DASH+EJS full download + ffmpeg trim + sticky session.

        Downloads the ENTIRE video (may be 2+ hours) then trims with ffmpeg -c copy.
        Uses _decodo_full_timeout (600s) to allow enough time for large videos.
        """
        if not (decodo_proxy and _decodo_healthy):
            return None
        sticky_proxy = _get_decodo_proxy(sticky=True)
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_decodo_full_timeout,
                            strategy_name="Strategy 0a (Decodo DASH+EJS full+trim+sticky)",
                            format_override=_DASH_H264_FORMAT,
                            use_ejs=True, proxy=sticky_proxy,
                            skip_sections=True, errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=False, strategy="decodo_dash_ejs_full_trim")
        return None

    def _s0b_decodo_muxed() -> DownloadResult | None:
        """Decodo muxed fallback (360p) via Python lib, full download + trim.

        Uses _decodo_full_timeout since this downloads the entire video.
        Muxed format avoids DASH merge issues but is limited to 360p.
        """
        if not (decodo_proxy and _decodo_healthy):
            return None
        sticky_proxy = _get_decodo_proxy(sticky=True)
        if _yt_dlp_python_download(url, output_path, client="android",
                                   start_time=0, end_time=0,
                                   strategy_name="Strategy 0b (Decodo muxed full+trim+sticky)",
                                   format_override=_MUXED_FORMAT,
                                   proxy=sticky_proxy,
                                   timeout=_decodo_full_timeout, errors_detail=_errors_detail):
            if has_time_range:
                _trim_video(output_path, start_time, end_time, ffmpeg_binary)
            if _file_valid():
                return _make_result(output_path, sectioned=False, strategy="decodo_muxed_pylib")
        return None

    def _s_render_early() -> DownloadResult | None:
        """Render server early — used when Decodo is unreachable."""
        if not (decodo_proxy and not _decodo_healthy):
            return None
        if not RENDER_SERVER_URL:
            return None
        _render_timeout_early = 300 if _is_long_video else 120
        with httpx.Client(timeout=httpx.Timeout(_render_timeout_early)) as _rs_client:
            if _render_server_download(url, output_path, start_time, end_time, ffmpeg_binary,
                                       _rs_client, "Strategy 3-early (Decodo fallback)"):
                if _file_valid():
                    return _make_result(output_path, sectioned=has_time_range, strategy="render_server_early")
        return None

    def _s1_android_vr_dash() -> DownloadResult | None:
        """android_vr + DASH H.264 + EJS (720p+ from non-blocked IPs)."""
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_strategy_timeout, strategy_name="Strategy 1",
                            format_override=_DASH_H264_FORMAT, use_ejs=True,
                            errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="android_vr_dash_ejs")
        return None

    def _s2_android_vr_proxy() -> DownloadResult | None:
        """android_vr + DASH H.264 + EJS + YT_DLP_PROXY."""
        if not proxy:
            return None
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_strategy_timeout, strategy_name="Strategy 2",
                            format_override=_DASH_H264_FORMAT, use_ejs=True,
                            proxy=proxy, errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="android_vr_dash_ejs_warp")
        return None

    def _s3_render_server() -> DownloadResult | None:
        """Render server proxy — pre-trimmed video."""
        if not RENDER_SERVER_URL:
            return None
        _render_timeout = 300 if _is_long_video else 120
        with httpx.Client(timeout=httpx.Timeout(_render_timeout)) as client:
            if _render_server_download(url, output_path, start_time, end_time, ffmpeg_binary, client, "Strategy 3"):
                if _file_valid():
                    return _make_result(output_path, sectioned=has_time_range, strategy="render_server")
        return None

    def _s4_android_muxed_ejs() -> DownloadResult | None:
        """android muxed + EJS (360p but reliable)."""
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android", start_time=start_time, end_time=end_time,
                            strategy_name="Strategy 4", format_override=_MUXED_FORMAT,
                            use_ejs=True, timeout=_strategy_timeout,
                            errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="android_muxed_ejs")
        return None

    def _s5_python_lib() -> DownloadResult | None:
        """Python lib android_vr + DASH, full download + trim."""
        if _yt_dlp_python_download(url, output_path, client="android_vr",
                                   start_time=0, end_time=0,
                                   strategy_name="Strategy 5 (full+trim)",
                                   format_override=_DASH_H264_FORMAT,
                                   timeout=_strategy_timeout,
                                   errors_detail=_errors_detail):
            if has_time_range:
                _trim_video(output_path, start_time, end_time, ffmpeg_binary)
            if _file_valid():
                return _make_result(output_path, sectioned=False, strategy="python_lib_android_vr")
        return None

    def _s6_android_bare() -> DownloadResult | None:
        """android muxed no-EJS (bare minimum fallback)."""
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android", start_time=start_time, end_time=end_time,
                            strategy_name="Strategy 6", format_override=_ANDROID_FORMAT,
                            use_ejs=False, timeout=_strategy_timeout,
                            errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="android_muxed_bare")
        return None

    def _s7_render_extract() -> DownloadResult | None:
        """Render server /extract-frames (last resort)."""
        if not RENDER_SERVER_URL:
            return None
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
                    if _file_valid():
                        return _make_result(output_path, sectioned=has_time_range, strategy="render_extract_frames")
                if "json" in content_type or "text" in content_type:
                    data = resp.json()
                    download_url = (data.get("cloudinaryUrl") or data.get("downloadUrl")
                                    or data.get("url") or data.get("videoUrl"))
                    if download_url:
                        dl_resp = client.get(download_url, follow_redirects=True)
                        if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                            if output_path.exists():
                                output_path.unlink()
                            output_path.write_bytes(dl_resp.content)
                            if _file_valid():
                                return _make_result(output_path, sectioned=has_time_range, strategy="render_extract_frames_url")
        return None

    # ── Build strategy chain as (name, function) tuples ──────────────────
    strategies: list[tuple[str, callable]] = [
        ("decodo_dash_sections", _s0_decodo_sections),
        ("decodo_dash_full_trim", _s0a_decodo_full_trim),
        ("decodo_muxed_full_trim", _s0b_decodo_muxed),
        ("render_server_early", _s_render_early),
        ("android_vr_dash_ejs", _s1_android_vr_dash),
        ("android_vr_dash_proxy", _s2_android_vr_proxy),
        ("render_server", _s3_render_server),
        ("android_muxed_ejs", _s4_android_muxed_ejs),
        ("python_lib_full_trim", _s5_python_lib),
        ("android_muxed_bare", _s6_android_bare),
        ("render_extract_frames", _s7_render_extract),
    ]

    # ── Run strategies — each wrapped in try/except, NEVER crashes chain ─
    for name, fn in strategies:
        if _total_expired():
            strategy_errors.append(f"{name}=total_timeout_skip")
            continue  # Skip but NEVER raise — always try remaining strategies

        try:
            result = fn()
            if result is not None:
                elapsed = round(time.perf_counter() - dl_start, 1)
                LOGGER.info("Downloaded in %ss via %s", elapsed, name)
                return result
            # fn returned None — strategy didn't apply or failed gracefully
            strategy_errors.append(f"{name}=failed")
        except Exception as exc:
            strategy_errors.append(f"{name}={type(exc).__name__}")
            LOGGER.warning("Strategy %s EXCEPTION: %s: %s", name, type(exc).__name__, str(exc)[:200])
            continue

    # Build detailed error message with yt-dlp stderr excerpts
    _detail_str = ""
    if _errors_detail:
        _detail_parts = [f"{k}: {v}" for k, v in _errors_detail.items()]
        _detail_str = f" Detail: {'; '.join(_detail_parts[:5])}"
    raise RuntimeError(
        f"All download strategies failed for: {url} "
        f"(original: {original_url}). "
        f"Decodo={('healthy' if _decodo_healthy else 'unhealthy')}. "
        f"Errors: {', '.join(strategy_errors)}.{_detail_str}"
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
            "strategy_used": dl_result.strategy_used,
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
