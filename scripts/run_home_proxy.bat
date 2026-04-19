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

if defined TUNNEL_TOKEN (
    echo Using PERMANENT Cloudflare tunnel...
    echo URL: proxy.cliptapp.com (never changes)
    echo.
    echo ====================================================
    echo   HOME PROXY IS RUNNING (Permanent Tunnel)
    echo ====================================================
    echo   URL: https://proxy.cliptapp.com
    echo   This URL never changes. Railway is pre-configured.
    echo   Keep this window open while using AI Highlights.
    echo   Close to stop the proxy.
    echo ====================================================
    echo.
    "%CLOUDFLARED%" tunnel --no-autoupdate run --token %TUNNEL_TOKEN%
) else (
    echo No permanent tunnel configured. Using quick tunnel...
    echo (URL will change each restart - see setup guide for permanent tunnel)
    echo.

    :: Create a temp file to capture the tunnel URL
    set TUNNEL_LOG=%TEMP%\clipt_tunnel.log
    del "%TUNNEL_LOG%" 2>nul

    :: Start tunnel in background, log output
    start "Clipt Tunnel" /min cmd /c ""%CLOUDFLARED%" tunnel --url http://localhost:5050 --no-autoupdate 2>"%TUNNEL_LOG%""

    :: Wait for tunnel URL to appear
    echo Waiting for tunnel URL...
    :wait_loop
    timeout /t 2 /nobreak >nul
    findstr /C:"trycloudflare.com" "%TUNNEL_LOG%" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 goto wait_loop

    :: Extract tunnel URL
    for /f "tokens=*" %%a in ('findstr /C:"trycloudflare.com" "%TUNNEL_LOG%"') do set TUNNEL_LINE=%%a

    echo.
    echo ====================================================
    echo   HOME PROXY IS RUNNING (Quick Tunnel)
    echo ====================================================
    echo   Check %TUNNEL_LOG% for your tunnel URL
    echo.
    echo   For a permanent URL, set up a named tunnel:
    echo   1. Go to https://one.dash.cloudflare.com
    echo   2. Networks ^> Tunnels ^> Create
    echo   3. Save the token to: %~dp0.tunnel-token
    echo.
    echo   Keep this window open while using AI Highlights.
    echo   Close to stop the proxy.
    echo ====================================================
    echo.

    :: Try to auto-update Railway if CLI is available
    where railway >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo Updating Railway env vars with quick tunnel URL...
        for /f "tokens=*" %%u in ('findstr /R "https://.*trycloudflare.com" "%TUNNEL_LOG%"') do (
            for %%w in (%%u) do (
                echo %%w | findstr "https://" >nul && (
                    railway variables --set "HOME_PROXY_URL=%%w"
                    echo Railway updated with: %%w
                )
            )
        )
    )

    echo.
    echo Press any key to stop the proxy...
    pause >nul
)

:: Cleanup
taskkill /FI "WINDOWTITLE eq Clipt*" /F >nul 2>&1
echo Proxy stopped.
