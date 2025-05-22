@echo off
echo Starting NeuroSri Integrated System...
echo.

rem Check if virtual environment exists
if not exist venv\Scripts\activate.bat (
    echo Creating virtual environment...
    python -m venv venv
)

rem Activate virtual environment
call venv\Scripts\activate.bat

rem Install required packages
echo Installing required packages...
pip install flask flask-cors numpy pandas scipy matplotlib scikit-learn joblib torch pylsl requests bleak asyncio g4f

rem Create empty files if they don't exist
echo Preparing data files...
if not exist eeg_data.csv (
    echo timestamp,ch1,ch2,ch3 > eeg_data.csv
)

if not exist prediction_output.json (
    echo {"mental_state": "Normal/baseline state", "confidence": 0.5, "timestamp": "%date:~10,4%-%date:~4,2%-%date:~7,2%T%time:~0,2%:%time:~3,2%:%time:~6,2%", "eeg_data": [0, 0, 0], "counseling_response": "Initializing system..."} > prediction_output.json
)

rem Ask if user wants to enable BLE
echo.
set /p enable_ble="Do you want to enable BLE connectivity? (y/n, default is n): "
if /i "%enable_ble%"=="y" (
    echo Starting NeuroSri system with BLE ENABLED...
    python start_system.py
) else (
    echo Starting NeuroSri system with BLE DISABLED...
    python start_system.py --disable-ble
)

rem Deactivate virtual environment when done
call venv\Scripts\deactivate.bat

echo.
echo NeuroSri system shut down.
pause 