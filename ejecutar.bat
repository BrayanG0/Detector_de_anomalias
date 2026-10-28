@echo off
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
)
if exist model.pth .venv\Scripts\pip install -r requirements-modelo.txt
.venv\Scripts\python app.py
