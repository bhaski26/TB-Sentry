@echo off
title TB-Sentry Demo
cd /d "%~dp0"

echo ==========================================
echo           TB-Sentry Demo Launcher
echo ==========================================
echo.

REM Check for Python environment
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else if exist ".venv_test\Scripts\python.exe" (
    set "PYTHON=.venv_test\Scripts\python.exe"
) else (
    echo Creating virtual environment...
    py -3 -m venv .venv

    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        pause
        exit /b 1
    )

    set "PYTHON=.venv\Scripts\python.exe"
)

echo.
echo Checking Python...
"%PYTHON%" --version

if errorlevel 1 (
    echo ERROR: Python could not start.
    pause
    exit /b 1
)

REM Install dependencies
echo.
echo Installing/checking dependencies...
"%PYTHON%" -m pip install -r requirements-demo.txt

if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

REM Download and verify model
echo.
echo Checking TB-Sentry model checkpoint...
"%PYTHON%" download_model.py

if errorlevel 1 (
    echo ERROR: Model setup failed.
    pause
    exit /b 1
)

REM Launch application
echo.
echo Starting TB-Sentry...
echo Your browser should open automatically.
echo Press Ctrl+C to stop the application.
echo.

"%PYTHON%" -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo The application exited with an error.
    pause
)