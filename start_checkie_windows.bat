@echo off
cd /d %~dp0
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt
python checkie_backend_app_v7_qr_print.py
pause
