@echo off
rem Abre el dashboard en tu computador. Los datos los actualizan las GitHub
rem Actions; aqui solo se traen los ultimos desde GitHub antes de abrir.
cd /d "%~dp0"
echo Trayendo los datos mas recientes desde GitHub...
git pull --quiet
pip install -q -r requirements.txt
streamlit run app.py
pause
