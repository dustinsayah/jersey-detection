@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "SKIP_BOOTSTRAP=0"
set "RUN_SMOKE_TEST=0"
set "BOOTSTRAP_ARGS="
set "SKIP_REPOS=0"

:parse_args
if "%~1"=="" goto after_parse
if /I "%~1"=="--help" goto show_help
if /I "%~1"=="/?" goto show_help
if /I "%~1"=="--skip-bootstrap" (
  set "SKIP_BOOTSTRAP=1"
  shift
  goto parse_args
)
if /I "%~1"=="--smoke-test" (
  set "RUN_SMOKE_TEST=1"
  shift
  goto parse_args
)
if /I "%~1"=="--skip-repos" set "SKIP_REPOS=1"
set "BOOTSTRAP_ARGS=%BOOTSTRAP_ARGS% %~1"
shift
goto parse_args

:after_parse
call :find_python
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment in .venv...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto fail
) else (
  echo [INFO] Reusing existing virtual environment.
)

set "VENV_PY=.venv\Scripts\python.exe"

echo [INFO] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto fail

echo [INFO] Installing Python dependencies...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail

if "%SKIP_BOOTSTRAP%"=="1" (
  echo [INFO] Skipping public-reader bootstrap.
) else (
  if "%SKIP_REPOS%"=="0" (
    where git >nul 2>&1
    if errorlevel 1 (
      echo [ERROR] Git is required for bootstrap. Install Git or rerun with --skip-repos.
      exit /b 1
    )
  )
  echo [INFO] Bootstrapping public-reader repos and weights...
  "%VENV_PY%" scripts\bootstrap_public_reader.py %BOOTSTRAP_ARGS%
  if errorlevel 1 goto fail
)

if "%RUN_SMOKE_TEST%"=="1" (
  echo [INFO] Running smoke tests...
  "%VENV_PY%" -m pytest -q tests/test_runtime_behavior.py tests/test_jersey_reader_backend.py tests/test_public_reader_pipeline.py
  if errorlevel 1 goto fail
)

echo.
echo [OK] Setup complete.
echo.
echo Next steps:
echo   .venv\Scripts\python.exe -m uvicorn asgi:app --host 0.0.0.0 --port 8000
echo   open http://127.0.0.1:8000/docs
exit /b 0

:find_python
set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 (
  py -3.11 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PYTHON_CMD=py -3.11"
)
if not defined PYTHON_CMD (
  where python >nul 2>&1
  if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
  )
)
if defined PYTHON_CMD (
  echo [INFO] Using Python command: %PYTHON_CMD%
  exit /b 0
)
echo [ERROR] Python 3.11 was not found. Install Python 3.11 and rerun setup.bat.
exit /b 1

:show_help
echo Usage: setup.bat [setup options] [bootstrap_public_reader.py options]
echo.
echo Quick start:
echo   setup.bat
echo   setup.bat --skip-models
echo   setup.bat --skip-repos --skip-models
echo   setup.bat --smoke-test
echo.
echo Setup options:
echo   --skip-bootstrap   Skip scripts\bootstrap_public_reader.py completely.
echo   --smoke-test       Run focused smoke tests after setup.
echo.
echo Any other options are forwarded to scripts\bootstrap_public_reader.py
echo For example:
echo   setup.bat --skip-models --no-update
exit /b 0

:fail
echo.
echo [ERROR] Setup failed.
exit /b 1
