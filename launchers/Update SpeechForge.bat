@echo off
setlocal EnableExtensions
title SpeechForge - update

REM These launchers live in <release>\launchers\. In the installed layout
REM <root>\current -> <root>\releases\<tag>, the shared data is <root>\data.
for %%I in ("%~dp0..") do set "REPO=%%~fI"
for %%I in ("%REPO%\..") do set "PARENT=%%~fI"
cd /d "%REPO%"
if exist "%REPO%\local-env.bat" call "%REPO%\local-env.bat"
if not defined SPEECHFORGE_ROOT if exist "%PARENT%\data" set "SPEECHFORGE_ROOT=%PARENT%"
if not defined SPEECHFORGE_DATA_DIR if defined SPEECHFORGE_ROOT set "SPEECHFORGE_DATA_DIR=%SPEECHFORGE_ROOT%\data"
if not defined SPEECHFORGE_DATA_DIR set "SPEECHFORGE_DATA_DIR=%REPO%\data"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "PY=%REPO%\.venv\Scripts\python.exe"

REM Blue/green (SPEC 10.4): builds releases\<tag> beside the running one, runs
REM the canary self-test on 127.0.0.1:1237, drains, switches "current",
REM restarts, and switches back by itself if the new release fails within 2 min.
REM Usage: Update SpeechForge.bat C:\path\speechforge.bundle [tag]
if "%~1"=="" (
  echo   Usage: "Update SpeechForge.bat" C:\path\to\speechforge.bundle [tag]
  echo          "Update SpeechForge.bat" rollback
  pause
  exit /b 1
)
if /I "%~1"=="rollback" (
  "%PY%" -m speechforge update --rollback
  pause
  exit /b %ERRORLEVEL%
)
"%PY%" -m speechforge update --check --bundle "%~1"
if "%~2"=="" (
  "%PY%" -m speechforge update --apply --bundle "%~1"
) else (
  "%PY%" -m speechforge update --apply --bundle "%~1" --tag "%~2"
)
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" (echo   Update applied.) else (echo   Update NOT applied ^(current release unchanged or rolled back^).)
pause
exit /b %RC%
