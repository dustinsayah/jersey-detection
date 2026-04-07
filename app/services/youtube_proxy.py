# YouTube download proxy — 16-strategy robust download chain
#
# Key insight (Apr 2026): YouTube blocks datacenter IPs at the network level,
# limiting them to itag 18 (360p muxed). The n-challenge solver (EJS/deno) is
# also REQUIRED to unlock any DASH formats. Strategy order:
#   C0. Cookie-authenticated DASH (FREE, no proxy — if youtube_cookies.txt has auth)
#   C1. Cookie-authenticated muxed fallback
#   W0. WARP SOCKS5 + DASH H.264 + EJS (FREE — wireproxy on port 40000)
#   W1. WARP SOCKS5 + muxed fallback (FREE)
#   0.  Decodo residential proxy DASH (PAID)
#   0a-0c. Decodo muxed/range/full fallbacks
#   1.  yt-dlp android_vr + DASH H.264 + EJS (720p+ if IP not blocked)
#   2.  yt-dlp android_vr + DASH H.264 + EJS + proxy (if YT_DLP_PROXY set)
#   3.  Render server proxy
#   4.  yt-dlp android muxed (360p fallback)
#   5.  yt-dlp Python lib android_vr + DASH
#   6.  yt-dlp android muxed no-EJS (last resort)
#   7.  Render server /extract-frames
#
# Cookie notes (researched Apr 2026):
#   - Chrome 127+ has app-bound encryption — can't extract Chrome cookies
#   - Use Firefox for cookie export (plain SQLite, no encryption)
#   - Cookies expire in 3-5 days (must refresh from Firefox)
#   - Piped/Invidious/Cobalt are all dead as of 2026
#   - Decodo residential proxy remains the most reliable approach

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
    strategy_used: str = "unknown"  # Which download strategy succeeded


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

    Credentials are URL-encoded to handle special characters (@ # $ : etc.)
    that would otherwise break the proxy URL parsing.
    """
    from urllib.parse import quote
    user = os.getenv("DECODO_USERNAME", "").strip()
    passwd = os.getenv("DECODO_PASSWORD", "").strip()
    if user and passwd:
        encoded_user = quote(user, safe="")
        encoded_passwd = quote(passwd, safe="")
        proxy_url = f"http://{encoded_user}:{encoded_passwd}@gate.decodo.com:10001"
        LOGGER.info("Decodo proxy: user=%s (encoded=%s), url_len=%d", user[:4] + "...", encoded_user[:4] + "...", len(proxy_url))
        return proxy_url
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

# Cloudflare WARP SOCKS5 proxy — wireproxy listens on port 40000
_WARP_PROXY = "socks5://127.0.0.1:40000"

def _is_warp_running() -> bool:
    """Check if wireproxy WARP SOCKS5 proxy is listening on port 40000."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", 40000), timeout=2):
            return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False

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

    # CRITICAL: Set HTTP_PROXY/HTTPS_PROXY env vars for the subprocess.
    # yt-dlp's --proxy only applies to yt-dlp's own HTTP requests.
    # When merging DASH video+audio, yt-dlp invokes ffmpeg which makes its
    # OWN HTTP requests to googlevideo.com URLs. Without the proxy env var,
    # ffmpeg hits the datacenter IP and gets "End of file" errors.
    env = os.environ.copy()
    if proxy:
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
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

    # Per-strategy timeout: 60s for quick-try strategies (DASH, sections)
    _strategy_timeout = 900 if _is_long_video else 60

    # Decodo sectioned timeout: only downloads the requested range
    # Keep short since sections always invoke ffmpeg which fails through proxy
    _decodo_sections_timeout = 900 if _is_long_video else 60

    # Decodo FULL download timeout: downloads the ENTIRE video then trims.
    # Source video may be 2+ hours. At ~1MB/s residential proxy speed,
    # a 555MB 360p muxed video takes ~555s. Allow 1200s for safety.
    _decodo_full_timeout = 1200

    # Overall timeout: generous for all videos since full download may be needed
    _TOTAL_TIMEOUT = 1500

    def _total_expired() -> bool:
        elapsed = time.perf_counter() - dl_start
        if elapsed > _TOTAL_TIMEOUT:
            LOGGER.warning("youtube_proxy_sync: total timeout %.0fs > %ds", elapsed, _TOTAL_TIMEOUT)
            return True
        return False

    def _file_valid() -> bool:
        """Check output file exists and is large enough."""
        return output_path.exists() and output_path.stat().st_size > _MIN_FILE_SIZE

    # ── Cookie file detection ─────────────────────────────────────────
    cookie_file = _get_cookie_file()
    _has_cookies = bool(cookie_file)
    if _has_cookies:
        LOGGER.info("YouTube cookies found at %s — will try cookie-first strategies", cookie_file)

    # ── Decodo proxy setup ──────────────────────────────────────────────
    # NOTE: No health probe. The probe was too fragile — a single transient
    # failure (IP rotation, timeout, rate limit) would skip ALL Decodo strategies.
    # Instead, always try Decodo if configured. Let the actual download attempt
    # be the test. Each strategy has its own timeout and error handling.
    decodo_proxy = _get_decodo_proxy()
    _decodo_healthy = bool(decodo_proxy)  # Trust config, skip fragile probe
    if decodo_proxy:
        LOGGER.info("Decodo proxy configured — will attempt download strategies")

    proxy = _get_proxy()

    # Shared error detail dict — populated by helpers for diagnostics
    _errors_detail: dict[str, str] = {}

    # ── Strategy functions — each returns DownloadResult | None ───────────

    def _s_cookies_dash() -> DownloadResult | None:
        """Cookie-authenticated DASH H.264 (720p+ without proxy).

        If valid YouTube Premium/login cookies are present, this bypasses
        datacenter IP blocks entirely. FREE — no Decodo bandwidth used.
        Cookies expire every 3-5 days and must be refreshed from Firefox.
        """
        if not _has_cookies:
            return None
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_strategy_timeout,
                            strategy_name="Strategy C0 (Cookies DASH+EJS)",
                            format_override=_DASH_H264_FORMAT,
                            use_ejs=True, proxy="", errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="cookies_dash_ejs")
        return None

    def _s_cookies_muxed() -> DownloadResult | None:
        """Cookie-authenticated muxed fallback (360p, no proxy needed)."""
        if not _has_cookies:
            return None
        if _yt_dlp_python_download(url, output_path, client="android_vr",
                                   start_time=start_time, end_time=end_time,
                                   strategy_name="Strategy C1 (Cookies muxed pylib)",
                                   format_override=_MUXED_FORMAT,
                                   proxy="",
                                   timeout=_strategy_timeout,
                                   errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="cookies_muxed_pylib")
        return None

    def _s0_decodo_sections() -> DownloadResult | None:
        """Decodo DASH+EJS with --download-sections (ideal: fast + 720p).

        HTTP_PROXY env var is set in _yt_dlp_download so that ffmpeg also uses
        the Decodo proxy when merging DASH video+audio streams. Without this,
        ffmpeg hits the datacenter IP and gets 'End of file' on googlevideo URLs.
        """
        if not (decodo_proxy and _decodo_healthy):
            return None
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_decodo_sections_timeout,
                            strategy_name="Strategy 0 (Decodo DASH+EJS+sections)",
                            format_override=_DASH_H264_FORMAT,
                            use_ejs=True, proxy=decodo_proxy, errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="decodo_dash_ejs_sectioned")
        return None

    def _s0a_decodo_muxed_pylib_range() -> DownloadResult | None:
        """Decodo muxed via Python lib + download_ranges (most reliable).

        Uses yt-dlp's OWN HTTP client (not ffmpeg) for downloading.
        download_ranges cuts the video without ffmpeg during download.
        No ffmpeg connection issues, no DASH merge, pure yt-dlp HTTP.
        """
        if not (decodo_proxy and _decodo_healthy):
            return None
        if _yt_dlp_python_download(url, output_path, client="android_vr",
                                   start_time=start_time, end_time=end_time,
                                   strategy_name="Strategy 0a (Decodo muxed pylib+range)",
                                   format_override=_MUXED_FORMAT,
                                   proxy=decodo_proxy,
                                   timeout=_decodo_sections_timeout,
                                   errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="decodo_muxed_pylib_range")
        return None

    def _s0a2_decodo_muxed_sections() -> DownloadResult | None:
        """Decodo muxed+EJS with --download-sections (subprocess fallback)."""
        if not (decodo_proxy and _decodo_healthy):
            return None
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_decodo_sections_timeout,
                            strategy_name="Strategy 0a2 (Decodo muxed+EJS+sections)",
                            format_override=_MUXED_FORMAT,
                            use_ejs=True, proxy=decodo_proxy, errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="decodo_muxed_ejs_sectioned")
        return None

    def _s0b_decodo_full_trim() -> DownloadResult | None:
        """Decodo DASH+EJS full download + ffmpeg trim.

        Downloads the ENTIRE video (may be 2+ hours) then trims with ffmpeg -c copy.
        SKIP for short time ranges — makes no sense to download 2 hours for 60 seconds.
        Only used for full-game requests where sectioned download fails.
        """
        if not (decodo_proxy and _decodo_healthy):
            return None
        # Skip full-download for short clips — wasteful of bandwidth + time
        _requested_range = (end_time - start_time) if end_time > start_time else 0
        if _requested_range > 0 and _requested_range < 1800:
            LOGGER.info("Strategy 0b: SKIP — short range (%.0fs), full download wasteful", _requested_range)
            return None
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_decodo_full_timeout,
                            strategy_name="Strategy 0b (Decodo DASH+EJS full+trim)",
                            format_override=_DASH_H264_FORMAT,
                            use_ejs=True, proxy=decodo_proxy,
                            skip_sections=True, errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=False, strategy="decodo_dash_ejs_full_trim")
        return None

    def _s0c_decodo_muxed_pylib() -> DownloadResult | None:
        """Decodo muxed via Python lib — full download + local trim.

        KEY INSIGHT: This is the ONLY strategy that avoids ffmpeg during download.
        - Muxed format = single pre-merged file, no DASH video+audio merge
        - No download_ranges = no ffmpeg section cutting during download
        - yt-dlp Python lib = uses its own HTTP client, not ffmpeg subprocess
        - After download completes, trim with ffmpeg on LOCAL file (no proxy)

        This downloads the ENTIRE video (may be 2+ hours) even for short clips,
        but it's the only reliable path when Decodo proxy drops ffmpeg connections.
        360p muxed at ~500MB/2hr takes ~170s at 3MB/s residential.
        """
        if not (decodo_proxy and _decodo_healthy):
            return None
        if _yt_dlp_python_download(url, output_path, client="android_vr",
                                   start_time=0, end_time=0,
                                   strategy_name="Strategy 0c (Decodo muxed pylib full+trim)",
                                   format_override=_MUXED_FORMAT,
                                   proxy=decodo_proxy,
                                   timeout=_decodo_full_timeout, errors_detail=_errors_detail):
            if has_time_range:
                _trim_video(output_path, start_time, end_time, ffmpeg_binary)
            if _file_valid():
                return _make_result(output_path, sectioned=False, strategy="decodo_muxed_pylib_full_trim")
        return None

    # ── Cloudflare WARP strategies ─────────────────────────────────────
    # WARP routes through Cloudflare's consumer VPN network (millions of users),
    # so YouTube treats WARP IPs as residential — FREE 720p from datacenter.
    _warp_available = _is_warp_running()
    if _warp_available:
        LOGGER.info("WARP proxy detected on %s — will try WARP strategies", _WARP_PROXY)

    def _s_warp_dash() -> DownloadResult | None:
        """WARP SOCKS5 + DASH H.264 + EJS — FREE 720p via Cloudflare network."""
        if not _warp_available:
            return None
        if _yt_dlp_download(url, output_path, yt_dlp_binary, ffmpeg_binary,
                            client="android_vr", start_time=start_time, end_time=end_time,
                            timeout=_strategy_timeout,
                            strategy_name="Strategy W0 (WARP DASH+EJS)",
                            format_override=_DASH_H264_FORMAT, use_ejs=True,
                            proxy=_WARP_PROXY, errors_detail=_errors_detail):
            if _file_valid():
                return _make_result(output_path, sectioned=has_time_range, strategy="warp_dash_ejs")
        return None

    def _s_warp_muxed_pylib() -> DownloadResult | None:
        """WARP SOCKS5 + muxed via Python lib — full download + trim."""
        if not _warp_available:
            return None
        if _yt_dlp_python_download(url, output_path, client="android_vr",
                                   start_time=0, end_time=0,
                                   strategy_name="Strategy W1 (WARP muxed pylib)",
                                   format_override=_MUXED_FORMAT,
                                   proxy=_WARP_PROXY,
                                   timeout=_strategy_timeout, errors_detail=_errors_detail):
            if has_time_range:
                _trim_video(output_path, start_time, end_time, ffmpeg_binary)
            if _file_valid():
                return _make_result(output_path, sectioned=False, strategy="warp_muxed_pylib")
        return None

    def _s_render_early() -> DownloadResult | None:
        """Render server early — used when Decodo is not configured."""
        if decodo_proxy:
            return None  # Decodo is configured, skip early render server
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
    # Order (Apr 2026):
    #   C0-C1. Cookie-authenticated (FREE — if cookies file has YouTube auth)
    #   W0-W1. Cloudflare WARP SOCKS5 (FREE — wireproxy on port 40000)
    #   0-0c.  Decodo residential proxy (PAID — reliable 720p)
    #   1-7.   Direct datacenter + render server fallbacks
    strategies: list[tuple[str, callable]] = [
        # Cookie-first: FREE, no proxy — try if cookies file has valid YouTube auth
        ("cookies_dash_ejs", _s_cookies_dash),
        ("cookies_muxed_pylib", _s_cookies_muxed),
        # WARP: FREE — routes through Cloudflare's consumer VPN, YouTube treats as residential
        ("warp_dash_ejs", _s_warp_dash),
        ("warp_muxed_pylib", _s_warp_muxed_pylib),
        # Decodo residential proxy — paid but reliable
        ("decodo_dash_sections", _s0_decodo_sections),
        ("decodo_muxed_pylib_full_trim", _s0c_decodo_muxed_pylib),
        ("decodo_muxed_pylib_range", _s0a_decodo_muxed_pylib_range),
        ("decodo_muxed_sections", _s0a2_decodo_muxed_sections),
        ("decodo_dash_full_trim", _s0b_decodo_full_trim),
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
