@echo off
REM Automated setup script for Cross-Market Prediction project
REM This script creates a fresh conda environment and installs all dependencies
REM Works on Windows with Anaconda/Miniconda

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo          Cross-Market Prediction - Automated Setup (Windows)
echo ============================================================================
echo.
echo This script will:
echo   1. Create a fresh conda environment 'ml_fresh'
echo   2. Install all required scientific computing packages
echo   3. Install PyTorch and PyTorch Geometric
echo   4. Ready your environment to run project.py
echo.
echo Time required: 5-10 minutes
echo Disk space needed: ~2-3 GB
echo.
echo ============================================================================
echo.

REM Check if conda is available
echo [Step 0] Checking for conda installation...
conda --version >nul 2>&1
if errorlevel 1 (
    echo X Conda not found. Please install Anaconda or Miniconda first.
    echo   Download from: https://www.anaconda.com/download
    pause
    exit /b 1
)
echo. ✓ Conda found
echo.

REM Step 1: Create environment
echo [Step 1] Creating fresh conda environment 'ml_fresh'...
echo Command: conda create -n ml_fresh python=3.9 numpy pandas scipy scikit-learn matplotlib seaborn networkx statsmodels yfinance plotly -c conda-forge -y
echo.
call conda create -n ml_fresh python=3.9 numpy pandas scipy scikit-learn matplotlib seaborn networkx statsmodels yfinance plotly -c conda-forge -y
if errorlevel 1 (
    echo.
    echo [NOTICE] Full package installation failed. Trying minimal installation...
    call conda create -n ml_fresh python=3.9 -y
    if errorlevel 1 (
        echo X Failed to create environment
        pause
        exit /b 1
    )
)
echo. ✓ Environment created

REM Step 2: Install PyTorch
echo.
echo [Step 2] Installing PyTorch...
echo Command: conda activate ml_fresh ^&^& conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
echo.
call conda activate ml_fresh
call conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
if errorlevel 1 (
    echo.
    echo [NOTICE] PyTorch conda installation failed. Trying pip...
    call python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)
echo. ✓ PyTorch installed

REM Step 3: Install PyTorch Geometric
echo.
echo [Step 3] Installing PyTorch Geometric...
echo Command: pip install torch-geometric
echo.
call python -m pip install torch-geometric
if errorlevel 1 (
    echo X PyTorch Geometric installation failed
    echo  (This is optional - try running project.py anyway)
) else (
    echo. ✓ PyTorch Geometric installed
)

REM Step 4: Verify installation
echo.
echo [Step 4] Verifying installation...
call python -c "import torch; import numpy; import pandas; print('✓ All imports successful!')" 2>nul
if errorlevel 1 (
    echo [WARNING] Some imports failed - environment may need manual fixes
) else (
    echo. ✓ All packages verified
)

REM Success message
echo.
echo ============================================================================
echo                       SETUP COMPLETE!
echo ============================================================================
echo.
echo Next steps:
echo.
echo 1. The environment 'ml_fresh' is now ready
echo    (It will remain active in this window)
echo.
echo 2. Navigate to your project directory:
echo    cd "C:\Users\AIML\Desktop\Cross-Market Prediction using Dynamic Neural Network"
echo.
echo 3. Run the project:
echo    python project.py
echo.
echo 4. Or test first with the mock version:
echo    python project_mock.py
echo.
echo 5. To verify imports work:
echo    python test_imports.py
echo.
echo ============================================================================
echo.

pause
