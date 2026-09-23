@echo off
chcp 65001 > nul
setlocal
title Video Enhancer 4K
cd /d "%~dp0"

echo ========================================================
echo   Video Enhancer 4K - Environment Setup
echo ========================================================

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo Creating local Python virtual environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%~dp0.venv"
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

"%VENV_PY%" -c "import PyQt6" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies (one-time setup)...
    "%VENV_PY%" -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
)

"%VENV_PY%" "%~dp0main.py" %*
if errorlevel 1 (
    echo.
    echo An error occurred. Run launch.bat --doctor for diagnostic.
    pause
)

endlocal
