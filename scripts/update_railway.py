#!/usr/bin/env python3
"""
Reads cloudflared stderr output, finds the tunnel URL,
and POSTs it to the local Flask server at /set-tunnel-url.
Flask then updates Railway's HOME_PROXY_URL automatically.

Usage (piped from cloudflared):
    cloudflared tunnel --url http://localhost:5050 2>&1 | python update_railway.py
"""
import re
import sys
import time

def main():
    print("[update_railway] Watching for tunnel URL...")
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

            # Wait for Flask to be ready
            time.sleep(2)

            try:
                import requests
                resp = requests.post(
                    "http://localhost:5050/set-tunnel-url",
                    json={"url": url},
                    timeout=10,
                )
                result = resp.json()
                print(f"[update_railway] Flask notified: {result}")
            except Exception as e:
                print(f"[update_railway] Failed to notify Flask: {e}")
                # Try with urllib as fallback (no requests dependency needed)
                try:
                    import urllib.request
                    import json
                    data = json.dumps({"url": url}).encode()
                    req = urllib.request.Request(
                        "http://localhost:5050/set-tunnel-url",
                        data=data,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        print(f"[update_railway] Flask notified (urllib): {resp.read().decode()}")
                except Exception as e2:
                    print(f"[update_railway] urllib fallback also failed: {e2}")

            # Don't break - keep reading to keep the pipe alive
            # (cloudflared exits if stdout/stderr pipe breaks)
            continue

    print("[update_railway] cloudflared stream ended")


if __name__ == "__main__":
    main()
