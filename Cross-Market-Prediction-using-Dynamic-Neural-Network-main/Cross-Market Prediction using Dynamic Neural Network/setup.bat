@echo off
REM Automated setup script for Cross-Market Prediction project
REM This script creates a fresh conda environment and installs all dependencies
REM Works on Windows with Anaconda/Miniconda

echo.
echo ============================================================================
echo          Cross-Market Prediction - Automated Setup
echo ============================================================================
echo.
echo This will create a fresh environment and install all dependencies.
echo Time: 5-10 minutes | Disk space: 2-3 GB
echo.
echo ============================================================================
echo.

REM Check if conda is available
echo [Step 0] Checking for conda...
conda --version >nul 2>&1
if errorlevel 1 (
    echo X Conda not found. Install Anaconda/Miniconda first.
    echo   https://www.anaconda.com/download
    pause
    exit /b 1
)
echo OK - Conda found
echo.

REM Step 1: Remove old environment if it exists
echo [Step 1] Removing old ml_fresh environment if it exists...
call conda remove -n ml_fresh -y >nul 2>&1
echo OK
echo.

REM Step 2: Create new environment with all packages including Streamlit
echo [Step 2] Creating fresh conda environment 'ml_fresh'...
echo This may take 2-5 minutes. Please wait...
call conda create -n ml_fresh python=3.9 numpy pandas scipy scikit-learn matplotlib seaborn networkx statsmodels yfinance plotly -c conda-forge -y
if errorlevel 1 (
    echo WARNING: Full install failed. Installing minimal version...
    call conda create -n ml_fresh python=3.9 -y
)
echo OK - Environment created
echo.

REM Step 2b: Install Streamlit and related web packages
echo [Step 2b] Installing web interface packages (Streamlit, Plotly)...
call conda activate ml_fresh
pip install streamlit plotly --quiet
if errorlevel 1 (
    echo WARNING: Web packages install had issues, but continuing...
)
echo OK - Web packages installed
echo.

REM Step 3: Initialize conda for this shell
echo [Step 3] Preparing conda activation...
call conda init cmd.exe >nul 2>&1
echo OK
echo.

REM Step 4: Install PyTorch
echo [Step 4] Installing PyTorch...
call conda activate ml_fresh
call conda install pytorch::pytorch torchvision torchaudio -c pytorch -y 2>nul
if errorlevel 1 (
    echo Trying pip install for PyTorch...
    call python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)
echo OK - PyTorch installed
echo.

REM Step 5: Install PyTorch Geometric
echo [Step 5] Installing PyTorch Geometric...
call python -m pip install torch-geometric 2>nul
if errorlevel 1 (
    echo WARNING: PyTorch Geometric install skipped (optional)
) else (
    echo OK - PyTorch Geometric installed
)
echo.

REM Step 6: Verify
echo [Step 6] Verifying installation...
call python -c "import torch; import numpy; import pandas; print('SUCCESS - All imports work!')" 2>nul
if errorlevel 1 (
    echo WARNING: Some imports failed
) else (
    echo OK - All packages verified
)
echo.

echo ============================================================================
echo                    SETUP COMPLETE!
echo ============================================================================
echo.
echo Your 'ml_fresh' environment is ready.
echo.
echo OPTION 1 - Run Web Dashboard (Recommended):
echo   conda activate ml_fresh
echo   streamlit run project_streamlit.py
echo.
echo OPTION 2 - Run Full Project:
echo   conda activate ml_fresh
echo   python project.py
echo.
echo OPTION 3 - Run Mock Demo:
echo   conda activate ml_fresh
echo   python project_mock.py
echo.
echo ============================================================================
echo Web Dashboard will open in your browser at: http://localhost:8501
echo ============================================================================
echo.
echo   cd "C:\Users\AIML\Desktop\Cross-Market Prediction using Dynamic Neural Network"
echo   python project.py
echo.
echo Or test first:
echo   python project_mock.py
echo.
echo ============================================================================
echo.
pause
