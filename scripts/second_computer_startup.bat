@echo off
:: ============================================================
:: CLIPT HOME PROXY - Automatic Startup Script
:: ============================================================
:: Place this in your Startup folder to auto-start on boot:
::   Win+R → shell:startup → copy this file there
:: ============================================================

echo ====================================================
echo   CLIPT HOME PROXY - Auto-starting...
echo ====================================================
echo.

:: Wait for network to be ready
echo Waiting for network...
timeout /t 10 /nobreak >nul

:: Check network connectivity
ping -n 1 google.com >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Waiting for network...
    timeout /t 10 /nobreak >nul
    ping -n 1 google.com >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: No network detected. Will try anyway.
    )
)

:: Change to scripts directory
cd /d "%~dp0"

:: Load Railway token if available
set RAILWAY_TOKEN=
if exist "%~dp0.railway-token" (
    set /p RAILWAY_TOKEN=<"%~dp0.railway-token"
)

:: Start Flask server in background
echo Starting Home Proxy on port 5050...
start "Clipt Home Proxy" /min python "%~dp0home_proxy.py"

:: Give Flask a moment to start
timeout /t 5 /nobreak >nul

:: Quick health check
curl -s http://localhost:5050/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Flask not ready, waiting longer...
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:5050/health >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Home proxy failed to start
        pause
        exit /b 1
    )
)
echo Home proxy is running on port 5050

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

:: Check for permanent tunnel token
set TUNNEL_TOKEN=
if exist "%~dp0.tunnel-token" (
    set /p TUNNEL_TOKEN=<"%~dp0.tunnel-token"
)

if defined TUNNEL_TOKEN (
    echo Using PERMANENT tunnel (proxy.cliptapp.com)
    :: Tell Flask to update Railway with the permanent URL
    curl -s -X POST http://localhost:5050/set-tunnel-url -H "Content-Type: application/json" -d "{\"url\":\"https://proxy.cliptapp.com\"}" >nul 2>&1
    for /f "tokens=*" %%t in ("%TUNNEL_TOKEN%") do set TUNNEL_TOKEN=%%t
    "%CLOUDFLARED%" --no-autoupdate tunnel run --token %TUNNEL_TOKEN%
) else (
    echo Using quick tunnel (auto-updates Railway)
    echo.
    :: Start cloudflared and pipe through update_railway.py
    :: This detects the URL and auto-updates Railway
    "%CLOUDFLARED%" tunnel --url http://localhost:5050 --no-autoupdate 2>&1 | python "%~dp0update_railway.py"
)
