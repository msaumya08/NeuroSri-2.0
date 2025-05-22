@echo off
echo Starting Integrated NeuroSri System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv\Scripts\activate.bat (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

REM Start the integrated system
echo.
echo Starting NeuroSri system...
python start_integrated_system.py

REM Deactivate virtual environment when done
call venv\Scripts\deactivate.bat

echo.
echo NeuroSri system shut down.
pause 