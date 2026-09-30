@echo off
echo ===================================================
echo Starting Resume Skill-Gap Assistant (GENAI-09)...
echo ===================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
