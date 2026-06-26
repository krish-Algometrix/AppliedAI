@echo off
title BFSI Risk Register AI
color 1F

echo.
echo  ==========================================
echo   BFSI Risk Register AI - Local System
echo  ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

REM Check Ollama
ollama --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Ollama not found. Install from https://ollama.com
    pause
    exit /b 1
)

REM Check if index exists
if not exist "risk_index\risk_index.faiss" (
    echo  [First run] Building vector index...
    echo  This takes ~30 seconds and downloads a small embedding model.
    echo.
    python build_index.py
    if errorlevel 1 (
        echo  ERROR: Index build failed. Check error above.
        pause
        exit /b 1
    )
)

REM Check if model is available
echo  Checking LLM model availability...
ollama list | findstr "llama3.2:3b" >nul 2>&1
if errorlevel 1 (
    echo  Pulling llama3.2:3b model (~2GB, one-time download)...
    ollama pull llama3.2:3b
)

echo.
echo  Starting web UI at http://localhost:7860
echo  (Your browser will open automatically)
echo.
echo  Press Ctrl+C to stop the server.
echo.

python app.py --model llama3.2:3b

pause
