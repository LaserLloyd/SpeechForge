@echo off
setlocal EnableExtensions
title SpeechForge

REM These launchers live in <release>\launchers\. In the installed layout
REM <root>\current -> <root>\releases\<tag>, the shared data is <root>\data
REM (or what <root>\local-env.bat says, when install.ps1 ran with -DataDir).
REM In a plain checkout (developer layout) the data is <repo>\data unless
REM <repo>\local-env.bat or the environment sets SPEECHFORGE_DATA_DIR.
for %%I in ("%~dp0..") do set "REPO=%%~fI"
for %%I in ("%REPO%\..") do set "PARENT=%%~fI"
cd /d "%REPO%"
if exist "%REPO%\local-env.bat" call "%REPO%\local-env.bat"
for %%I in ("%PARENT%") do if /I "%%~nxI"=="releases" if exist "%PARENT%\..\local-env.bat" call "%PARENT%\..\local-env.bat"
if not defined SPEECHFORGE_ROOT if exist "%PARENT%\data" set "SPEECHFORGE_ROOT=%PARENT%"
if not defined SPEECHFORGE_DATA_DIR if defined SPEECHFORGE_ROOT set "SPEECHFORGE_DATA_DIR=%SPEECHFORGE_ROOT%\data"
if not defined SPEECHFORGE_DATA_DIR set "SPEECHFORGE_DATA_DIR=%REPO%\data"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "PY=%REPO%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo   The virtual environment is missing: run install.ps1 ^(first install^) or
  echo   "speechforge update --apply" ^(a new release^).
  pause
  exit /b 1
)
echo.
echo   SpeechForge  (data: %SPEECHFORGE_DATA_DIR%)
echo   Close this window or press Ctrl+C to stop. The supervisor restarts the
echo   server if it exits or stops answering, stops after 5 crashes in 10 minutes,
echo   and stops it cleanly on exit (the GPU lease is released on the way out).
echo.
"%PY%" -m speechforge doctor
"%PY%" -m speechforge supervise %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo   SpeechForge exited with code %RC%. 78 = configuration refused; see above.
  pause
)
exit /b %RC%
