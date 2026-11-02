@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
title Video Enhancer 4K (AnimaScale)
cd /d "%~dp0"

echo ========================================================
echo   Video Enhancer 4K - Environment Setup
echo ========================================================

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo Creating local Python virtual environment...
    set "PY_CMD="
    for %%V in (3.12 3.11 3.10 3) do (
        if not defined PY_CMD (
            py -%%V -V >nul 2>nul
            if not errorlevel 1 (
                set "PY_CMD=py -%%V"
            )
        )
    )
    if defined PY_CMD (
        echo Using: !PY_CMD!
        !PY_CMD! -m venv "%~dp0.venv"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo ERROR: Python 3.10 or newer is required.
            echo Please install Python from https://www.python.org/downloads/
            pause
            exit /b 1
        )
        python -m venv "%~dp0.venv"
    )
)

if not exist "%VENV_PY%" (
    echo ERROR: Unable to create Python virtual environment.
    pause
    exit /b 1
)

"%VENV_PY%" -c "from PyQt6.QtWidgets import QApplication; from PIL import Image" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies for first-time setup...
    "%VENV_PY%" -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
)

"%VENV_PY%" "%~dp0main.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

rem Pause if an error occurred or if informational CLI command was executed
if "%EXIT_CODE%" neq "0" (
    echo.
    echo An error occurred. Run launch.bat --doctor for diagnostic.
    pause
) else (
    echo "%*" | findstr /i "\-\-doctor \-\-help \-h \-\-list\-presets" >nul
    if not errorlevel 1 (
        echo.
        pause
    )
)

exit /b %EXIT_CODE%
