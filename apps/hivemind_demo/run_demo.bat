@echo off
cd /d "%~dp0"
echo Starting HiveMind demo at http://localhost:8501 ...
start "" http://localhost:8501
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
