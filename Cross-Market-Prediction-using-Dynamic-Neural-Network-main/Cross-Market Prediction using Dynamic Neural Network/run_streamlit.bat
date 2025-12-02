@echo off
REM Launch Streamlit app using ml_fresh environment
setlocal enabledelayedexpansion

REM Set up the path to conda
set "CONDA_ROOT=C:\Users\AIML\anaconda3"
set "PROJECT_DIR=C:\Users\AIML\Desktop\Cross-Market Prediction using Dynamic Neural Network"

REM Activate the ml_fresh environment which adds it to PATH
call "%CONDA_ROOT%\Scripts\activate.bat" ml_fresh

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Use explicit path to python from conda - try multiple locations
set "PYTHON_EXE=%CONDA_ROOT%\envs\ml_fresh\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%CONDA_ROOT%\envs\ml_fresh\Library\bin\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%CONDA_ROOT%\envs\ml_fresh\python.exe"

REM Check if python exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found in ml_fresh environment
    echo.
    echo Checked locations:
    echo - %CONDA_ROOT%\envs\ml_fresh\Lib\venv\scripts\nt\python.exe
    echo - %CONDA_ROOT%\envs\ml_fresh\python.exe
    echo - %CONDA_ROOT%\envs\ml_fresh\Scripts\python.exe
    echo - %CONDA_ROOT%\envs\ml_fresh\Library\bin\python.exe
    exit /b 1
)

echo Using Python from: %PYTHON_EXE%
echo.

REM Run Streamlit with explicit python path
"%PYTHON_EXE%" -m streamlit run project_streamlit.py

REM Indicate if there was an error
if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch Streamlit (exit code: %errorlevel%)
    exit /b %errorlevel%
)
