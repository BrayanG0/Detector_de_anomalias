#!/usr/bin/env bash
# Linux/macOS sin conda: crea .venv, instala dependencias y arranca.
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
if [ -f "model.pth" ]; then
  .venv/bin/pip install -r requirements-modelo.txt
fi
.venv/bin/python app.py
