#!/bin/bash
set -e

# ── Link volume models ──
if [ -d "/data/models" ]; then
  for f in /data/models/*.pt /data/models/*.pth; do
    [ -f "$f" ] || continue
    bn=$(basename "$f")
    [ -f "/app/app/model/$bn" ] || ln -sf "$f" "/app/app/model/$bn"
  done
  echo "Volume models linked"
fi

# ── Download critical football models if missing (6MB each, non-fatal) ──
GITHUB_RAW="https://raw.githubusercontent.com/dustinsayah/jersey-detection/main/app/model"
for model in football_player_detector.pt football_digit_detector.pt football_jersey_tracker.pt; do
  if [ ! -f "/app/app/model/$model" ]; then
    echo "Downloading missing model: $model"
    curl -fsSL "$GITHUB_RAW/$model" -o "/app/app/model/$model" 2>/dev/null \
      && echo "Downloaded $model" \
      || echo "Failed to download $model (non-fatal)"
  fi
done
echo "Models: $(ls /app/app/model/*.pt /app/app/model/*.pth 2>/dev/null | wc -l)"

# ── Reassemble split v7 models (>100MB, split for GitHub) ──
for base in navy_jersey_specialist_v7 football_player_crop_v7; do
  if [ ! -f "/app/app/model/${base}.pt" ] && [ -f "/app/app/model/${base}.pt.part_aa" ]; then
    echo "Reassembling ${base}.pt from parts..."
    cat /app/app/model/${base}.pt.part_* > /app/app/model/${base}.pt
    echo "Reassembled ${base}.pt ($(du -h /app/app/model/${base}.pt | cut -f1))"
  fi
done

# ── Update yt-dlp at runtime (YouTube changes API frequently) ──
echo "Updating yt-dlp..."
pip install --upgrade --pre yt-dlp 2>/dev/null \
  && echo "yt-dlp updated: $(yt-dlp --version)" \
  || echo "yt-dlp update failed (using build version: $(yt-dlp --version))"

# ── Start Cloudflare WARP proxy (non-fatal) ──
WARP_OK=false
if [ -n "$WARP_WG_CONFIG" ]; then
  echo "Starting Cloudflare WARP proxy..."
  echo "$WARP_WG_CONFIG" | base64 -d > /tmp/warp-raw.conf

  # Patch the config for better reliability on Railway:
  # 1. Lower MTU from 1280 to 1200 (avoid silent packet drops)
  # 2. Use direct IP endpoint instead of DNS (faster, no DNS resolution issues)
  # 3. Add PersistentKeepalive to keep NAT mappings alive
  # 4. Remove IPv6 (simplify, Railway may not support IPv6 WireGuard)
  sed -i 's/MTU = 1280/MTU = 1200/' /tmp/warp-raw.conf
  sed -i 's/Endpoint = engage.cloudflareclient.com:2408/Endpoint = 162.159.192.1:2408/' /tmp/warp-raw.conf
  # Remove IPv6 address line
  sed -i '/^Address = 2606:/d' /tmp/warp-raw.conf
  # Remove IPv6 AllowedIPs line
  sed -i '/^AllowedIPs = ::\/0/d' /tmp/warp-raw.conf
  # Remove IPv6 DNS entries
  sed -i 's/, 2606:4700:4700::1111, 2606:4700:4700::1001//' /tmp/warp-raw.conf
  # Add PersistentKeepalive if not present
  if ! grep -q 'PersistentKeepalive' /tmp/warp-raw.conf; then
    echo "PersistentKeepalive = 5" >> /tmp/warp-raw.conf
  fi

  # Append wireproxy SOCKS5 + HTTP proxy config
  # SOCKS5 on :40000 for yt-dlp, HTTP on :40001 for ffmpeg (ffmpeg can't use SOCKS5)
  printf "\n[Socks5]\nBindAddress = 127.0.0.1:40000\n\n[http]\nBindAddress = 127.0.0.1:40001\n" >> /tmp/warp-raw.conf

  echo "=== wireproxy config ==="
  cat /tmp/warp-raw.conf | grep -v PrivateKey
  echo "========================"

  # Start wireproxy with stderr redirected to suppress DEBUG spam
  wireproxy -c /tmp/warp-raw.conf 2>/tmp/wireproxy.log &
  WARP_PID=$!
  sleep 3

  if kill -0 $WARP_PID 2>/dev/null; then
    echo "wireproxy process alive (PID=$WARP_PID)"

    # REAL connectivity test: actually try to reach the internet through the proxy
    echo "Testing WARP connectivity..."
    for attempt in 1 2 3; do
      WARP_IP=$(curl -s --max-time 10 --proxy socks5h://127.0.0.1:40000 https://httpbin.org/ip 2>/dev/null || true)
      if [ -n "$WARP_IP" ] && echo "$WARP_IP" | grep -q "origin"; then
        echo "WARP connectivity CONFIRMED (attempt $attempt): $WARP_IP"
        WARP_OK=true
        break
      fi
      echo "WARP test attempt $attempt failed, waiting 3s..."
      sleep 3
    done

    if [ "$WARP_OK" = true ]; then
      echo "WARP SOCKS5 WORKING on socks5://127.0.0.1:40000"
      # Also test HTTP proxy on port 40001
      HTTP_TEST=$(curl -s --max-time 10 --proxy http://127.0.0.1:40001 https://httpbin.org/ip 2>/dev/null || true)
      if [ -n "$HTTP_TEST" ] && echo "$HTTP_TEST" | grep -q "origin"; then
        echo "WARP HTTP proxy WORKING on http://127.0.0.1:40001: $HTTP_TEST"
      else
        echo "WARP HTTP proxy on port 40001 NOT responding (SOCKS5 still works)"
      fi
      export YT_DLP_PROXY="socks5://127.0.0.1:40000"
    else
      echo "WARP proxy STARTED but connectivity test FAILED"
      echo "wireproxy logs:"
      tail -20 /tmp/wireproxy.log 2>/dev/null || true
      # Kill the non-functional wireproxy
      kill $WARP_PID 2>/dev/null || true
    fi
  else
    echo "wireproxy process DIED immediately"
    echo "wireproxy logs:"
    cat /tmp/wireproxy.log 2>/dev/null || true
  fi
else
  echo "WARP_WG_CONFIG not set, skipping WARP proxy"
fi

# Report proxy status
if [ "$WARP_OK" = true ]; then
  echo "=== YouTube proxy: WARP SOCKS5 ==="
elif [ -n "$YT_DLP_PROXY" ]; then
  echo "=== YouTube proxy: $YT_DLP_PROXY ==="
else
  echo "=== YouTube proxy: NONE (will use direct/cookies/Decodo) ==="
fi

exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout ${GUNICORN_TIMEOUT:-1800} asgi:app
