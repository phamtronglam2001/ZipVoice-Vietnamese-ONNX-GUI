@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "%~dp0src\cli_tts.py" %*
) else (
    python "%~dp0src\cli_tts.py" %*
)
exit /b %ERRORLEVEL%
