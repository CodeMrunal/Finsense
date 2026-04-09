@echo off
REM Startup script for FinSense (Windows)

echo Starting FinSense Financial Intelligence System...

REM Check if .env exists
if not exist .env (
    echo .env file not found. Copying from config/env.example...
    copy config\env.example .env
    echo Please edit .env file with your API keys
    pause
    exit /b 1
)

REM Create necessary directories
if not exist data mkdir data
if not exist models mkdir models
if not exist logs mkdir logs
if not exist chatbot\vector_store mkdir chatbot\vector_store
if not exist chatbot\knowledge_base mkdir chatbot\knowledge_base

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist venv\.installed (
    echo Installing dependencies...
    pip install -r requirements.txt
    type nul > venv\.installed
)

REM Start services based on argument
if "%1"=="backend" (
    echo Starting backend API...
    python run_backend.py
) else if "%1"=="dashboard" (
    echo Starting Streamlit dashboard...
    streamlit run dashboard/app.py
) else (
    echo Usage: start.bat [backend^|dashboard]
    echo   backend   - Start backend API only
    echo   dashboard - Start dashboard only
    pause
    exit /b 1
)













