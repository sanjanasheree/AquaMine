@echo off
echo Installing requirements (first run only)...
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause
