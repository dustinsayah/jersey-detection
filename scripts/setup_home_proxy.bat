@echo off
echo ====================================================
echo   CLIPT HOME PROXY - Setup Guide
echo ====================================================
echo.

:: Step 1: Install Python dependencies
echo [Step 1/4] Installing Python dependencies...
pip install flask yt-dlp
echo.

:: Step 2: Check for cloudflared
echo [Step 2/4] Checking for cloudflared...
where cloudflared >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo cloudflared NOT found. Installing...
    echo.
    echo Option A: Download from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
    echo Option B: Run: winget install cloudflare.cloudflared
    echo.
    winget install cloudflare.cloudflared
    echo.
) else (
    echo cloudflared found!
)

:: Step 3: Login to Cloudflare (one time)
echo [Step 3/4] Logging into Cloudflare...
echo This will open your browser. Log in with your Cloudflare account.
echo.
cloudflared tunnel login

:: Step 4: Create named tunnel
echo [Step 4/4] Creating named tunnel 'clipt-proxy'...
cloudflared tunnel create clipt-proxy

echo.
echo ====================================================
echo   SETUP COMPLETE! Now do these manual steps:
echo ====================================================
echo.
echo 1. Add DNS route (run this command):
echo    cloudflared tunnel route dns clipt-proxy proxy.cliptapp.com
echo.
echo 2. Create config file at %%USERPROFILE%%\.cloudflared\config.yml:
echo.
echo    tunnel: ^<TUNNEL_UUID from above^>
echo    credentials-file: %%USERPROFILE%%\.cloudflared\^<TUNNEL_UUID^>.json
echo    ingress:
echo      - hostname: proxy.cliptapp.com
echo        service: http://localhost:5000
echo      - service: http_status:404
echo.
echo 3. Add Railway env vars:
echo    HOME_PROXY_URL = https://proxy.cliptapp.com
echo    HOME_PROXY_SECRET = clipt-home-proxy-2026
echo.
echo 4. To run (2 terminals):
echo    Terminal 1: python scripts/home_proxy.py
echo    Terminal 2: cloudflared tunnel run clipt-proxy
echo.
echo Or run both at once:
echo    start "Home Proxy" python scripts/home_proxy.py
echo    cloudflared tunnel run clipt-proxy
echo.
pause
