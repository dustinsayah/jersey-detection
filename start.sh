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

# ── Start bgutil PO Token server (required for web/mweb clients) ──
if [ -f "/app/bgutil-pot/server/build/main.js" ] && command -v node &>/dev/null; then
  echo "Starting bgutil PO Token server on port 4416..."
  node /app/bgutil-pot/server/build/main.js -p 4416 &
  POT_PID=$!
  sleep 3
  if kill -0 $POT_PID 2>/dev/null; then
    echo "PO Token server running (PID=$POT_PID)"
    export YT_DLP_POT_PROVIDER_PORT=4416
    # Verify it responds
    POT_CHECK=$(curl -s --max-time 5 http://127.0.0.1:4416/ 2>/dev/null || true)
    if [ -n "$POT_CHECK" ]; then
      echo "PO Token server verified: responding on port 4416"
    else
      echo "PO Token server started but not responding (may need warm-up)"
    fi
  else
    echo "PO Token server failed to start"
    cat /tmp/bgutil-pot.log 2>/dev/null || true
  fi
else
  echo "bgutil PO Token server not found (node=$(command -v node 2>/dev/null || echo 'missing'), build=$(ls /app/bgutil-pot/server/build/main.js 2>/dev/null || echo 'missing'))"
fi

# ── Start Cloudflare WARP proxy pool (up to 3 instances, non-fatal) ──
# Each WARP account gets its own wireproxy on a unique port pair:
#   WARP_WG_CONFIG   → SOCKS5 :40000 / HTTP :40001
#   WARP_WG_CONFIG_2 → SOCKS5 :40002 / HTTP :40003
#   WARP_WG_CONFIG_3 → SOCKS5 :40004 / HTTP :40005
WARP_OK=false
WARP_POOL_SIZE=0

start_warp_instance() {
  local CONFIG_B64="$1"
  local SOCKS_PORT="$2"
  local HTTP_PORT="$3"
  local LABEL="$4"
  local CONF_FILE="/tmp/warp-${LABEL}.conf"
  local LOG_FILE="/tmp/wireproxy-${LABEL}.log"

  echo "Starting WARP instance ${LABEL} (SOCKS5:${SOCKS_PORT}, HTTP:${HTTP_PORT})..."
  echo "$CONFIG_B64" | base64 -d > "$CONF_FILE"

  # Patch config for Railway reliability
  sed -i 's/MTU = 1280/MTU = 1200/' "$CONF_FILE"
  sed -i 's/Endpoint = engage.cloudflareclient.com:2408/Endpoint = 162.159.192.1:2408/' "$CONF_FILE"
  sed -i '/^Address = 2606:/d' "$CONF_FILE"
  sed -i '/^AllowedIPs = ::\/0/d' "$CONF_FILE"
  sed -i 's/, 2606:4700:4700::1111, 2606:4700:4700::1001//' "$CONF_FILE"
  if ! grep -q 'PersistentKeepalive' "$CONF_FILE"; then
    echo "PersistentKeepalive = 5" >> "$CONF_FILE"
  fi

  # Append SOCKS5 + HTTP proxy config on the specified ports
  printf "\n[Socks5]\nBindAddress = 127.0.0.1:${SOCKS_PORT}\n\n[http]\nBindAddress = 127.0.0.1:${HTTP_PORT}\n" >> "$CONF_FILE"

  echo "=== wireproxy config ${LABEL} ==="
  cat "$CONF_FILE" | grep -v PrivateKey
  echo "========================"

  wireproxy -c "$CONF_FILE" 2>"$LOG_FILE" &
  local PID=$!
  sleep 3

  if kill -0 $PID 2>/dev/null; then
    echo "wireproxy ${LABEL} alive (PID=$PID)"
    # Connectivity test
    for attempt in 1 2 3; do
      WARP_IP=$(curl -s --max-time 10 --proxy "socks5h://127.0.0.1:${SOCKS_PORT}" https://httpbin.org/ip 2>/dev/null || true)
      if [ -n "$WARP_IP" ] && echo "$WARP_IP" | grep -q "origin"; then
        echo "WARP ${LABEL} CONFIRMED (attempt $attempt): $WARP_IP"
        WARP_OK=true
        WARP_POOL_SIZE=$((WARP_POOL_SIZE + 1))
        return 0
      fi
      echo "WARP ${LABEL} test attempt $attempt failed, waiting 3s..."
      sleep 3
    done
    echo "WARP ${LABEL} STARTED but connectivity test FAILED"
    tail -20 "$LOG_FILE" 2>/dev/null || true
    kill $PID 2>/dev/null || true
  else
    echo "wireproxy ${LABEL} DIED immediately"
    cat "$LOG_FILE" 2>/dev/null || true
  fi
  return 1
}

# Start up to 3 WARP instances
if [ -n "$WARP_WG_CONFIG" ]; then
  start_warp_instance "$WARP_WG_CONFIG" 40000 40001 "warp1"
else
  echo "WARP_WG_CONFIG not set, skipping WARP proxy"
fi

if [ -n "$WARP_WG_CONFIG_2" ]; then
  start_warp_instance "$WARP_WG_CONFIG_2" 40002 40003 "warp2"
else
  echo "WARP_WG_CONFIG_2 not set, skipping WARP instance 2"
fi

if [ -n "$WARP_WG_CONFIG_3" ]; then
  start_warp_instance "$WARP_WG_CONFIG_3" 40004 40005 "warp3"
else
  echo "WARP_WG_CONFIG_3 not set, skipping WARP instance 3"
fi

# Export pool size for Python to discover
export WARP_POOL_SIZE=$WARP_POOL_SIZE

# Report proxy status
if [ "$WARP_OK" = true ]; then
  echo "=== YouTube proxy: WARP POOL (${WARP_POOL_SIZE} instances) ==="
  export YT_DLP_PROXY="socks5://127.0.0.1:40000"
elif [ -n "$YT_DLP_PROXY" ]; then
  echo "=== YouTube proxy: $YT_DLP_PROXY ==="
else
  echo "=== YouTube proxy: NONE (will use direct/cookies/Decodo) ==="
fi

exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout ${GUNICORN_TIMEOUT:-1800} asgi:app
