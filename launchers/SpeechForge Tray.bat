@echo off
setlocal EnableExtensions
title SpeechForge Tray

REM The tray controller (DECISIONS S23): an "SF" icon in the notification area
REM that adopts a server already answering on the port or launches
REM "speechforge supervise" itself, shows its state as a coloured dot, and
REM offers restart / stop / logs / start-at-logon from a menu.
REM
REM Same layout discovery as "Start SpeechForge.bat": in the installed layout
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
set "PYW=%REPO%\.venv\Scripts\pythonw.exe"

if not exist "%PY%" (
  echo   The virtual environment is missing: run install.ps1 ^(first install^) or
  echo   "speechforge update --apply" ^(a new release^).
  pause
  exit /b 1
)
REM pythonw keeps no console window behind the icon. The tray logs to
REM <data>\logs\tray.log; the supervisor it launches writes to
REM <data>\logs\supervisor.out.log and the server to logs\speechforge.jsonl.
REM Needs the tray extra (uv sync --extra gpu --extra tray); without it the
REM tray shows a message box saying so and exits.
if not exist "%PYW%" set "PYW=%PY%"
start "SpeechForge Tray" "%PYW%" -m speechforge tray %*
exit /b 0
