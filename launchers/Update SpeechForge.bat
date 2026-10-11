@echo off
setlocal EnableExtensions
title SpeechForge - update

REM These launchers live in <release>\launchers\. In the installed layout
REM <root>\current -> <root>\releases\<tag>, the shared data is <root>\data
REM (or what <root>\local-env.bat says, when install.ps1 ran with -DataDir).
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

REM Blue/green (SPEC 10.4): builds releases\<tag> beside the running one, runs
REM the canary self-test on 127.0.0.1:1237, drains, switches "current",
REM restarts, and switches back by itself if the new release fails within 2 min.
REM Usage: Update SpeechForge.bat [source] [tag]
REM   source = a git URL or a bundle/clone path; empty or "" = the release's
REM            origin (the public repository after a default install)
REM   tag    = a release tag (default: the newest v*)
REM   "Update SpeechForge.bat" rollback
if /I "%~1"=="help" goto :usage
if /I "%~1"=="/?" goto :usage
if /I "%~1"=="rollback" (
  "%PY%" -m speechforge update --rollback
  pause
  exit /b %ERRORLEVEL%
)
set "SRC="
if not "%~1"=="" set "SRC=--source "%~1""
set "TAG="
if not "%~2"=="" set "TAG=--tag "%~2""
"%PY%" -m speechforge update --check %SRC%
"%PY%" -m speechforge update --apply %SRC% %TAG%
goto :done

:usage
echo   Usage: "Update SpeechForge.bat"                      newest release from the release's origin
echo          "Update SpeechForge.bat" "" v0.3.0            that tag, same source
echo          "Update SpeechForge.bat" https://... [tag]    another git URL
echo          "Update SpeechForge.bat" C:\path\x.bundle [tag]   offline, from a bundle
echo          "Update SpeechForge.bat" rollback
pause
exit /b 1

:done
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" (echo   Update applied.) else (echo   Update NOT applied ^(current release unchanged or rolled back^).)
pause
exit /b %RC%
