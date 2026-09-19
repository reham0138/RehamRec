@echo off
title RehamRec - Launcher
cd /d "%~dp0"

echo ==========================================
echo        RehamRec - Project Launcher
echo ==========================================
echo.

if not exist "backend\description_model_project\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please create .venv and install requirements first.
    pause
    exit /b 1
)

echo Starting Backend...
start "RehamRec Backend" cmd /k "cd /d "%~dp0backend\description_model_project" && call .venv\Scripts\activate.bat && python -m uvicorn src.inference.app:app --reload"

echo Starting Frontend...
start "RehamRec Frontend" cmd /k "cd /d "%~dp0frontend" && python -m http.server 5500"

echo Waiting for servers...
timeout /t 3 /nobreak >nul

echo Opening RehamRec...
start "" "http://127.0.0.1:5500"

echo.
echo RehamRec is starting.
echo Keep the two CMD windows open while presenting.
echo.
pause
