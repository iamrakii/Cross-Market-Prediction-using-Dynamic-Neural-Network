@echo off
REM Install Streamlit and Plotly for web interface
REM This script should be run AFTER creating the ml_fresh environment

echo.
echo ============================================================
echo Installing Streamlit and Plotly packages...
echo ============================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Conda not found. Please install Anaconda first.
    pause
    exit /b 1
)

REM Try to activate ml_fresh environment
call conda activate ml_fresh 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Could not activate ml_fresh environment.
    echo Please run setup.bat first to create the environment.
    pause
    exit /b 1
)

echo Environment activated: ml_fresh
echo.

REM Install Streamlit via pip (more reliable than conda)
echo [1/2] Installing Streamlit...
pip install --upgrade streamlit 2>&1

if %errorlevel% neq 0 (
    echo Warning: Streamlit installation may have issues, but continuing...
)

REM Install Plotly
echo.
echo [2/2] Installing Plotly...
pip install --upgrade plotly 2>&1

if %errorlevel% neq 0 (
    echo Warning: Plotly installation may have issues, but continuing...
)

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo To run the Streamlit web app, use:
echo   streamlit run project_streamlit.py
echo.
echo The interface will open in your default browser at:
echo   http://localhost:8501
echo.
pause
