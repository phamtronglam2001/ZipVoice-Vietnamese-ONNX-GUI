@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" cli_tts.py %*
) else (
    python cli_tts.py %*
)
exit /b %ERRORLEVEL%
