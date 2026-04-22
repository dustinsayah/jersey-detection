@echo off
setlocal EnableDelayedExpansion
title Clipt Home Proxy

echo ====================================================
echo   CLIPT HOME PROXY - Starting...
echo ====================================================
echo.

:: ── Step 0: Stop cloudflared Windows service if running (conflicts with our tunnel) ──
sc query cloudflared >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [0/6] Checking cloudflared Windows service...
    net stop cloudflared >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        sc config cloudflared start= disabled >nul 2>&1
        echo       Service stopped and disabled.
    ) else (
        echo       Service running as SYSTEM - needs admin to stop.
        echo       Using dynamic port to avoid conflict.
    )
    echo.
)

:: ── Step 1: Kill any existing proxy/tunnel processes ──
echo [1/6] Cleaning up existing processes...
taskkill /F /FI "WINDOWTITLE eq Clipt-Proxy*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Clipt-Tunnel*" >nul 2>&1
:: Kill anything listening on ports 5050-5069
for /l %%p in (5050,1,5069) do (
    for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":%%p " ^| findstr "LISTENING"') do (
        if not "%%a"=="0" taskkill /F /PID %%a >nul 2>&1
    )
)
timeout /t 2 /nobreak >nul
echo       Done.
echo.

:: ── Step 2: Check dependencies ──
echo [2/6] Checking dependencies...
where python >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Python not found. Install Python 3.11+ first.
    pause
    exit /b 1
)

python -c "import flask" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo       Installing Flask...
    pip install flask >nul 2>&1
)

python -c "import yt_dlp" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo       Installing yt-dlp...
    pip install yt-dlp >nul 2>&1
)

python -c "import requests" 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo       Installing requests...
    pip install requests >nul 2>&1
)

:: Check for cloudflared
set CLOUDFLARED=cloudflared
where cloudflared >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    if exist "%LOCALAPPDATA%\cloudflared.exe" (
        set "CLOUDFLARED=%LOCALAPPDATA%\cloudflared.exe"
    ) else (
        echo       Downloading cloudflared...
        curl -fsSL -o "%LOCALAPPDATA%\cloudflared.exe" "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        set "CLOUDFLARED=%LOCALAPPDATA%\cloudflared.exe"
    )
)
echo       All dependencies OK.
echo.

:: ── Load tokens ──
set RAILWAY_TOKEN=
if exist "%~dp0.railway-token" (
    set /p RAILWAY_TOKEN=<"%~dp0.railway-token"
)

set TUNNEL_TOKEN=
if defined CLIPT_TUNNEL_TOKEN (
    set "TUNNEL_TOKEN=%CLIPT_TUNNEL_TOKEN%"
)
if not defined TUNNEL_TOKEN (
    if exist "%~dp0.tunnel-token" (
        set /p TUNNEL_TOKEN=<"%~dp0.tunnel-token"
    )
)

:: ── Step 3: Start Flask server ──
echo [3/6] Starting Flask server...
set "LOG_FILE=%TEMP%\clipt_proxy.log"
del "%LOG_FILE%" >nul 2>&1
del "%TEMP%\clipt_proxy_port.txt" >nul 2>&1

start "Clipt-Proxy" /min cmd /c "python "%~dp0home_proxy.py" > "%LOG_FILE%" 2>&1"

:: Wait for Flask to write its port file (up to 30 seconds)
set PROXY_PORT=
set /a WAIT_COUNT=0
:wait_for_port
timeout /t 1 /nobreak >nul
set /a WAIT_COUNT+=1
if !WAIT_COUNT! GTR 30 (
    echo       ERROR: Flask failed to start after 30 seconds.
    echo       Log:
    type "%LOG_FILE%" 2>nul
    echo.
    pause
    exit /b 1
)

:: Read port from temp file
set "PORT_FILE=%TEMP%\clipt_proxy_port.txt"
if exist "!PORT_FILE!" (
    set /p PROXY_PORT=<"!PORT_FILE!"
)
if "!PROXY_PORT!"=="" goto wait_for_port
echo       Flask started on port !PROXY_PORT! (%WAIT_COUNT%s)
echo.

:: ── Step 4: Health check ──
echo [4/6] Verifying proxy health...
set HEALTH_OK=0
for /l %%i in (1,1,15) do (
    if !HEALTH_OK!==0 (
        curl -s --max-time 2 http://localhost:!PROXY_PORT!/health >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            set HEALTH_OK=1
        ) else (
            timeout /t 1 /nobreak >nul
        )
    )
)
if "!HEALTH_OK!"=="0" (
    echo       ERROR: Proxy not responding on port !PROXY_PORT!
    echo       Log:
    type "%LOG_FILE%" 2>nul
    echo.
    pause
    exit /b 1
)
echo       Proxy healthy on port !PROXY_PORT!
echo.

:: ── Step 5: Start Cloudflare tunnel ──
echo [5/6] Starting Cloudflare tunnel...
set "TUNNEL_LOG=%TEMP%\clipt_tunnel.log"
del "!TUNNEL_LOG!" >nul 2>&1

if defined TUNNEL_TOKEN (
    :: Trim whitespace from token
    for /f "tokens=*" %%t in ("!TUNNEL_TOKEN!") do set "TUNNEL_TOKEN=%%t"

    echo       Using permanent tunnel: proxy.cliptapp.com
    set "TUNNEL_URL=https://proxy.cliptapp.com"

    :: Tell Flask to update Railway with the permanent URL
    curl -s -X POST http://localhost:!PROXY_PORT!/set-tunnel-url -H "Content-Type: application/json" -d "{\"url\":\"https://proxy.cliptapp.com\"}" >nul 2>&1

    echo.
    echo [6/6] Verifying Railway connection...
    timeout /t 3 /nobreak >nul

    echo.
    echo ====================================================
    echo   CLIPT HOME PROXY READY
    echo ====================================================
    echo   Port:    !PROXY_PORT!
    echo   Tunnel:  https://proxy.cliptapp.com
    echo   Railway: auto-update queued
    echo.
    echo   STATUS: ONLINE - YouTube downloads ENABLED
    echo.
    echo   Keep this window open while using Clipt.
    echo   Close to stop all proxy services.
    echo ====================================================
    echo.

    :: Run cloudflared in foreground (keeps bat file alive)
    "!CLOUDFLARED!" --no-autoupdate tunnel run --token !TUNNEL_TOKEN!

) else (
    echo       Using quick tunnel (URL changes each restart)...
    echo.

    :: Start cloudflared and pipe through update_railway.py to auto-detect URL
    echo [6/6] Detecting tunnel URL...
    echo.
    echo ====================================================
    echo   CLIPT HOME PROXY READY
    echo ====================================================
    echo   Port:    !PROXY_PORT!
    echo   Tunnel:  detecting URL...
    echo   Railway: will auto-update when URL found
    echo.
    echo   STATUS: ONLINE - YouTube downloads ENABLED
    echo.
    echo   Keep this window open while using Clipt.
    echo   Close to stop all proxy services.
    echo ====================================================
    echo.

    :: Run cloudflared piped through update_railway.py (keeps bat file alive)
    "!CLOUDFLARED!" tunnel --url http://localhost:!PROXY_PORT! --no-autoupdate 2>&1 | python "%~dp0update_railway.py"
)

:: ── Cleanup on exit ──
echo.
echo Shutting down...
taskkill /F /FI "WINDOWTITLE eq Clipt-Proxy*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Clipt-Tunnel*" >nul 2>&1
del "%TEMP%\clipt_proxy_port.txt" >nul 2>&1
echo Proxy stopped.
