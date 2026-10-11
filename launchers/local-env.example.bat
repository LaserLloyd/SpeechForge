@echo off
REM Optional machine-specific overrides for the launchers. Two places are read:
REM   <release>\local-env.bat        this release only (a checkout: copy this file
REM                                  next to pyproject.toml as local-env.bat)
REM   <root>\local-env.bat           every release under <root>\releases (install.ps1
REM                                  writes it when the data folder is not <root>\data)
REM Both are git-ignored. install.ps1's default layout needs neither.
REM
REM Run from a checkout (developer layout): a plain git clone, the data folder
REM outside the repo, no releases\<tag> + current junction. Set the data dir here
REM (or in the environment), then from the repo:
REM     uv sync --extra gpu
REM     uv run speechforge doctor
REM     uv run speechforge supervise        (or launchers\Start SpeechForge.bat)
REM `speechforge update` stays for the installed layout; a checkout updates with git.
REM set "SPEECHFORGE_DATA_DIR=D:\SpeechForge\data"
REM
REM Installed layout with the data somewhere else (both lines):
REM set "SPEECHFORGE_ROOT=C:\SpeechForge"
REM set "SPEECHFORGE_DATA_DIR=D:\SpeechForge\data"
