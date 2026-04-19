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

:: Start Cloudflare Quick Tunnel and capture URL
echo Starting Cloudflare Tunnel...
echo.

:: Create a temp file to capture the tunnel URL
set TUNNEL_LOG=%TEMP%\clipt_tunnel.log

:: Start tunnel in background, log output
start "Clipt Tunnel" /min cmd /c "%CLOUDFLARED% tunnel --url http://localhost:5050 --no-autoupdate 2>&1 | tee %TUNNEL_LOG%"

:: Wait for tunnel URL to appear
echo Waiting for tunnel URL...
:wait_loop
timeout /t 2 /nobreak >nul
findstr /C:"trycloudflare.com" "%TUNNEL_LOG%" >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto wait_loop

:: Extract tunnel URL
for /f "tokens=*" %%a in ('findstr /C:"trycloudflare.com" "%TUNNEL_LOG%"') do set TUNNEL_LINE=%%a
for /f "tokens=2 delims= " %%u in ('echo %TUNNEL_LINE% ^| findstr "https://"') do set TUNNEL_URL=%%u

echo.
echo ====================================================
echo   HOME PROXY IS RUNNING
echo ====================================================
echo   Tunnel URL: %TUNNEL_URL%
echo.
echo   To connect Railway, set these env vars:
echo     HOME_PROXY_URL = %TUNNEL_URL%
echo     HOME_PROXY_SECRET = clipt-home-proxy-2026
echo.
echo   Or run: railway variables --set "HOME_PROXY_URL=%TUNNEL_URL%"
echo.
echo   Keep this window open while using AI Highlights.
echo   Close to stop the proxy.
echo ====================================================
echo.

:: Try to auto-update Railway if CLI is available
where railway >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Updating Railway env vars automatically...
    railway variables --set "HOME_PROXY_URL=%TUNNEL_URL%"
    echo Railway updated! New deploy will start automatically.
)

echo.
echo Press any key to stop the proxy...
pause >nul

:: Cleanup
taskkill /FI "WINDOWTITLE eq Clipt*" /F >nul 2>&1
echo Proxy stopped.
