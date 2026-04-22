#!/usr/bin/env python3
"""
Reads cloudflared stderr output, finds the tunnel URL,
and POSTs it to the local Flask server at /set-tunnel-url.
Flask then updates Railway's HOME_PROXY_URL automatically.

Usage (piped from cloudflared):
    cloudflared tunnel --url http://localhost:PORT 2>&1 | python update_railway.py
"""
import os
import re
import sys
import tempfile
import time


def _read_proxy_port() -> int:
    """Read port from temp file written by home_proxy.py."""
    port_file = os.path.join(tempfile.gettempdir(), "clipt_proxy_port.txt")
    try:
        with open(port_file) as f:
            return int(f.read().strip())
    except Exception:
        return 5050  # fallback


def main():
    print("[update_railway] Watching for tunnel URL...")
    port = _read_proxy_port()
    print(f"[update_railway] Using Flask port: {port}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # Print cloudflared output for visibility
        print(f"[cloudflared] {line}")

        # Look for trycloudflare.com URL
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            print(f"\n[update_railway] Tunnel URL found: {url}")

            # Re-read port in case it changed
            port = _read_proxy_port()

            # Wait for Flask to be ready
            time.sleep(2)

            try:
                import urllib.request
                import json
                data = json.dumps({"url": url}).encode()
                req = urllib.request.Request(
                    f"http://localhost:{port}/set-tunnel-url",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    print(f"[update_railway] Flask notified: {resp.read().decode()}")
            except Exception as e:
                print(f"[update_railway] Failed to notify Flask: {e}")

            # Don't break - keep reading to keep the pipe alive
            # (cloudflared exits if stdout/stderr pipe breaks)
            continue

    print("[update_railway] cloudflared stream ended")


if __name__ == "__main__":
    main()
