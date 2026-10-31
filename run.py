"""One-command launcher:  python run.py   -> http://localhost:8000  (trains models first if missing)"""
import subprocess, sys, webbrowser
from pathlib import Path
R = Path(__file__).parent
if not (R / 'artifacts' / 'nli_model.joblib').exists():
    print('No trained models found - training (a few minutes)...')
    subprocess.check_call([sys.executable, 'train.py'], cwd=R / 'backend')
webbrowser.open('http://localhost:8000')
subprocess.call([sys.executable, '-m', 'uvicorn', 'main:app', '--app-dir', 'backend', '--port', '8000'], cwd=R)
