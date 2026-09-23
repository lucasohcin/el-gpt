#!/bin/bash
# One-click start script for El GPT 1.0

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "========================================================"
echo "          Starting El GPT 1.0 AI Engine                 "
echo "========================================================"

# Check Python 3.11 virtualenv
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating Python 3.11 virtual environment..."
    /opt/homebrew/bin/python3.11 -m venv .venv
fi

# Ensure dependencies installed
if ! .venv/bin/python -c "import torch, fastapi, uvicorn" 2>/dev/null; then
    echo "[2/4] Installing dependencies into .venv..."
    .venv/bin/pip install -r requirements.txt
fi

# Ensure datasets are generated
if [ ! -f "data/train.jsonl" ]; then
    echo "[3/4] Synthesizing math, coding, and chat training curriculum..."
    .venv/bin/python data/prepare_datasets.py
fi

# Launch server
echo "[4/4] Launching FastAPI backend and ChatGPT interface at http://localhost:8000"
echo "Press Ctrl+C to stop."
.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
