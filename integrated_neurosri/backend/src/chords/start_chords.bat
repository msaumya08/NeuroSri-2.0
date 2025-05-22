@echo off
title Chords EEG Data Acquisition
cd /d C:\Users\suyas\Desktop\EEG\Chords-Python-main

:: Activate virtual environment
call .\venv\Scripts\activate  

:: Install dependencies
pip install -r chords_requirements.txt

:: Open a new command window and run gui.py (initial window remains open)
start cmd /k "cd /d C:\Users\suyas\Desktop\EEG\Chords-Python-main && python gui.py"

:: Run chords.py in the same command window
python chords.py -p COM3 -b 115200 --csv --lsl -v -t 600


:: Keep the initial window open
cmd /k