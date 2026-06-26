#!/bin/bash
# BFSI Risk Register AI — Mac/Linux launcher

set -e

echo ""
echo "=========================================="
echo "  BFSI Risk Register AI - Local System"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 not found. Install from https://python.org"
    exit 1
fi

# Check Ollama
if ! command -v ollama &>/dev/null; then
    echo "ERROR: Ollama not found. Install from https://ollama.com"
    exit 1
fi

# Start Ollama server in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server in background..."
    ollama serve &>/dev/null &
    sleep 2
fi

# Build index if needed
if [ ! -f "risk_index/risk_index.faiss" ]; then
    echo "[First run] Building vector index (~30 seconds)..."
    python3 build_index.py
fi

# Pull model if not available
if ! ollama list | grep -q "llama3.2:3b"; then
    echo "Pulling llama3.2:3b model (~2GB, one-time)..."
    ollama pull llama3.2:3b
fi

echo ""
echo "Starting web UI at http://localhost:7860"
echo "Press Ctrl+C to stop."
echo ""

python3 app.py --model llama3.2:3b
