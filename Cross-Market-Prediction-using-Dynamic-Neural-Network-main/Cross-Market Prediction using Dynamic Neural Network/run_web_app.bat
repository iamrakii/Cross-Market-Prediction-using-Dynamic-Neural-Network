@echo off
REM Launch Flask Web App
setlocal enabledelayedexpansion

set "CONDA_ROOT=C:\Users\AIML\anaconda3"
set "PROJECT_DIR=C:\Users\AIML\Desktop\Cross-Market Prediction using Dynamic Neural Network"

REM Activate base environment (Flask should be available)
call "%CONDA_ROOT%\Scripts\activate.bat" base

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Run Flask app
echo.
echo ========================================================================
echo  Cross-Market Volatility Prediction - Flask Web App
echo ========================================================================
echo.
echo Starting Flask server...
echo.
echo Open your browser and go to: http://localhost:5000
echo.
echo Press CTRL+C to stop the server
echo ========================================================================
echo.

python app.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch Flask app
    echo.
    pause
)
