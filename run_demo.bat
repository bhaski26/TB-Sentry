@echo off
title TB-Sentry Demo
cd /d "%~dp0"

echo ==========================================
echo           TB-Sentry Demo Launcher
echo ==========================================
echo.

if exist ".venv_test\Scripts\python.exe" (
set "PYTHON=.venv_test\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
set "PYTHON=.venv\Scripts\python.exe"
) else (
echo ERROR: No existing virtual environment was found.
echo Expected .venv_test or .venv in:
echo %CD%
echo.
echo Please check your Python environment before continuing.
pause
exit /b 1
)

echo Checking Python environment...
"%PYTHON%" --version
if errorlevel 1 (
echo ERROR: Python could not start.
pause
exit /b 1
)

echo.
echo Starting TB-Sentry...
echo Your browser should open automatically.
echo Press Ctrl+C in this window to stop the application.
echo.

"%PYTHON%" -m streamlit run app.py

if errorlevel 1 (
echo.
echo The application exited with an error.
echo Please copy the error messages and share them with me.
pause
)
