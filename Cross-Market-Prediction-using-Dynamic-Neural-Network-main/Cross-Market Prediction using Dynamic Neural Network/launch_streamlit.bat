@echo off
REM Launch Streamlit app using ml_fresh environment
setlocal enabledelayedexpansion

REM Disable python alias issue by using explicit paths
REM Set up the path to conda
set "CONDA_ROOT=C:\Users\AIML\anaconda3"
set "PROJECT_DIR=C:\Users\AIML\Desktop\Cross-Market Prediction using Dynamic Neural Network"

REM Activate the ml_fresh environment which adds it to PATH
call "%CONDA_ROOT%\Scripts\activate.bat" ml_fresh

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Use explicit path to python from conda
REM Try multiple possible locations
set "PYTHON_EXE=%CONDA_ROOT%\envs\ml_fresh\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%CONDA_ROOT%\envs\ml_fresh\Lib\venv\scripts\nt\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%CONDA_ROOT%\envs\ml_fresh\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%CONDA_ROOT%\envs\ml_fresh\Library\bin\python.exe"

REM Check if python exists
if not exist "%PYTHON_EXE%" (
    echo Python executable not found at %PYTHON_EXE%
    echo.
    echo Available python in PATH after conda activation:
    where python.exe 2>nul || echo (None found)
    pause
    exit /b 1
)

echo Using Python from: %PYTHON_EXE%
echo.

REM Run Streamlit with explicit python path
"%PYTHON_EXE%" -m streamlit run project_streamlit.py

REM Keep window open if error
if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch Streamlit
    pause
)
