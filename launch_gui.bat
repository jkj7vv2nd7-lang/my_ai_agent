@echo off
cd /d "%~dp0"
echo Starting Local AI App Builder...

if not exist "%~dp0local_ai_agent.py" (
    echo [ERROR] local_ai_agent.py not found in %~dp0
    pause
    exit /b 1
)

py -3 -c "import tkinter" >nul 2>nul
if %errorlevel%==0 goto :run_py3

python -c "import tkinter" >nul 2>nul
if %errorlevel%==0 goto :run_python

python3 -c "import tkinter" >nul 2>nul
if %errorlevel%==0 goto :run_python3

echo [ERROR] Python 3.9+ with Tkinter was not found.
echo Please install Python from https://www.python.org/ (check tcl/tk during setup).
pause
exit /b 1

:run_py3
py -3 "%~dp0local_ai_agent.py" --gui
if errorlevel 1 pause
exit /b %errorlevel%

:run_python
python "%~dp0local_ai_agent.py" --gui
if errorlevel 1 pause
exit /b %errorlevel%

:run_python3
python3 "%~dp0local_ai_agent.py" --gui
if errorlevel 1 pause
exit /b %errorlevel%
