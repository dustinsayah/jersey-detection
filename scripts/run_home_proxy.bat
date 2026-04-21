@echo off
echo ====================================================
echo   CLIPT HOME PROXY - Starting...
echo ====================================================
echo.

:: Check dependencies
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Install Python 3.11+ first.
    pause
    exit /b 1
)

python -c "import flask" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing Flask...
    pip install flask
)

python -c "import yt_dlp" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing yt-dlp...
    pip install yt-dlp
)

:: Check for cloudflared
set CLOUDFLARED=cloudflared
where cloudflared >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    if exist "%LOCALAPPDATA%\cloudflared.exe" (
        set CLOUDFLARED=%LOCALAPPDATA%\cloudflared.exe
    ) else (
        echo cloudflared not found. Downloading...
        curl -fsSL -o "%LOCALAPPDATA%\cloudflared.exe" "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        set CLOUDFLARED=%LOCALAPPDATA%\cloudflared.exe
    )
)

:: Start Flask server in background
echo Starting Home Proxy on port 5050...
start "Clipt Home Proxy" /min python "%~dp0home_proxy.py"

:: Give Flask a moment to start
timeout /t 3 /nobreak >nul

:: Quick health check
curl -s http://localhost:5050/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Home proxy failed to start on port 5050
    pause
    exit /b 1
)
echo Home proxy is running on port 5050
echo.

:: ============================================================
:: PERMANENT TUNNEL (preferred) vs QUICK TUNNEL (fallback)
:: ============================================================
:: If CLIPT_TUNNEL_TOKEN is set, use permanent named tunnel.
:: This gives a fixed URL like proxy.cliptapp.com that never changes.
:: Set the token once: set CLIPT_TUNNEL_TOKEN=eyJ...
:: Or save it in: %~dp0.tunnel-token
:: ============================================================

set TUNNEL_TOKEN=
if defined CLIPT_TUNNEL_TOKEN (
    set TUNNEL_TOKEN=%CLIPT_TUNNEL_TOKEN%
)
if not defined TUNNEL_TOKEN (
    if exist "%~dp0.tunnel-token" (
        set /p TUNNEL_TOKEN=<"%~dp0.tunnel-token"
    )
)

:: ============================================================
:: Load Railway token for auto-update (reads from .railway-token file)
:: ============================================================
set RAILWAY_TOKEN=
if exist "%~dp0.railway-token" (
    set /p RAILWAY_TOKEN=<"%~dp0.railway-token"
    echo Railway auto-update: ENABLED
) else (
    echo Railway auto-update: DISABLED (no .railway-token file)
    echo To enable: save your Railway API token to %~dp0.railway-token
)

if defined TUNNEL_TOKEN (
    echo Using PERMANENT Cloudflare tunnel...
    echo URL: proxy.cliptapp.com (never changes)
    echo.

    :: Tell Flask to update Railway with the permanent URL
    curl -s -X POST http://localhost:5050/set-tunnel-url -H "Content-Type: application/json" -d "{\"url\":\"https://proxy.cliptapp.com\"}" >nul 2>&1

    echo ====================================================
    echo   HOME PROXY IS RUNNING (Permanent Tunnel)
    echo ====================================================
    echo   URL: https://proxy.cliptapp.com
    echo   Railway auto-updated with this URL.
    echo   Keep this window open while using AI Highlights.
    echo   Close to stop the proxy.
    echo ====================================================
    echo.
    :: Trim any trailing whitespace/newlines from token
    for /f "tokens=*" %%t in ("%TUNNEL_TOKEN%") do set TUNNEL_TOKEN=%%t
    "%CLOUDFLARED%" --no-autoupdate tunnel run --token %TUNNEL_TOKEN%
) else (
    echo Using quick tunnel (URL changes each restart, auto-updates Railway)...
    echo.

    :: Start tunnel and pipe output through update_railway.py
    :: This script reads cloudflared output, finds the URL, and POSTs it to Flask
    :: Flask then updates Railway's HOME_PROXY_URL automatically
    echo Starting Cloudflare tunnel with auto-update...
    echo.
    echo ====================================================
    echo   HOME PROXY IS RUNNING (Quick Tunnel + Auto-Update)
    echo ====================================================
    echo   Tunnel URL will be detected and sent to Railway
    echo   automatically. No manual steps needed.
    echo.
    echo   Keep this window open while using AI Highlights.
    echo   Close to stop the proxy.
    echo ====================================================
    echo.

    :: Run cloudflared and pipe stderr through update_railway.py
    :: The 2>&1 merges stderr to stdout so Python can read the URL
    "%CLOUDFLARED%" tunnel --url http://localhost:5050 --no-autoupdate 2>&1 | python "%~dp0update_railway.py"
)

:: Cleanup
taskkill /FI "WINDOWTITLE eq Clipt*" /F >nul 2>&1
echo Proxy stopped.
