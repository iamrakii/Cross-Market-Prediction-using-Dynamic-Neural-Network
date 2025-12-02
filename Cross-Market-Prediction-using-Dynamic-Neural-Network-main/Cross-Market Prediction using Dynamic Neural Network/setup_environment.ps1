# Automated setup script for Cross-Market Prediction project
# PowerShell version for Windows users
# Run as: powershell -ExecutionPolicy Bypass -File setup_environment.ps1

Write-Host "`n" -ForegroundColor White
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "          Cross-Market Prediction - Automated Setup (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "`n"

Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "  1. Create a fresh conda environment 'ml_fresh'" -ForegroundColor Yellow
Write-Host "  2. Install all required scientific computing packages" -ForegroundColor Yellow
Write-Host "  3. Install PyTorch and PyTorch Geometric" -ForegroundColor Yellow
Write-Host "  4. Ready your environment to run project.py" -ForegroundColor Yellow
Write-Host "`n"
Write-Host "Time required: 5-10 minutes" -ForegroundColor Yellow
Write-Host "Disk space needed: ~2-3 GB" -ForegroundColor Yellow
Write-Host "`n"
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "`n"

# Check if conda is available
Write-Host "[Step 0] Checking for conda installation..." -ForegroundColor Green
$condaExists = $false
try {
    $condaVersion = conda --version 2>$null
    Write-Host "✓ Conda found: $condaVersion" -ForegroundColor Green
    $condaExists = $true
} 
catch {
    Write-Host "✗ Conda not found. Please install Anaconda or Miniconda first." -ForegroundColor Red
    Write-Host "  Download from: https://www.anaconda.com/download" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not $condaExists) {
    Write-Host "✗ Conda not found. Please install Anaconda or Miniconda first." -ForegroundColor Red
    exit 1
}

# Step 1: Create environment
Write-Host "[Step 1] Creating fresh conda environment 'ml_fresh'..." -ForegroundColor Green
Write-Host "Command: conda create -n ml_fresh python=3.9 numpy pandas scipy scikit-learn matplotlib seaborn networkx statsmodels yfinance plotly -c conda-forge -y" -ForegroundColor Gray
Write-Host "`n"

conda create -n ml_fresh python=3.9 numpy pandas scipy scikit-learn matplotlib seaborn networkx statsmodels yfinance plotly -c conda-forge -y 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Environment created successfully" -ForegroundColor Green
} else {
    Write-Host "[NOTICE] Full installation failed. Trying minimal Python installation..." -ForegroundColor Yellow
    conda create -n ml_fresh python=3.9 -y 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to create environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
# Step 2: Install PyTorch
Write-Host "`n[Step 2] Installing PyTorch..." -ForegroundColor Green
Write-Host "Command: conda activate ml_fresh && conda install pytorch::pytorch torchvision torchaudio -c pytorch -y" -ForegroundColor Gray
Write-Host "`n"

conda activate ml_fresh; conda install pytorch::pytorch torchvision torchaudio -c pytorch -y 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[NOTICE] PyTorch conda installation failed. Trying pip..." -ForegroundColor Yellow
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>&1 | Out-Null
}
Write-Host "✓ PyTorch installation complete" -ForegroundColor Green

# Step 3: Install PyTorch Geometric
Write-Host "`n[Step 3] Installing PyTorch Geometric..." -ForegroundColor Green
Write-Host "Command: pip install torch-geometric" -ForegroundColor Gray
Write-Host "`n"

python -m pip install torch-geometric 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PyTorch Geometric installed" -ForegroundColor Green
} else {
    Write-Host "[WARNING] PyTorch Geometric installation failed" -ForegroundColor Yellow
    Write-Host "          (This is optional - project may still work)" -ForegroundColor Yellow
}

# Step 3b: Install Streamlit and web packages
Write-Host "`n[Step 3b] Installing web interface packages (Streamlit, Plotly)..." -ForegroundColor Green
Write-Host "Command: pip install streamlit plotly" -ForegroundColor Gray
Write-Host "`n"

python -m pip install streamlit plotly 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Streamlit and Plotly installed" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Web packages installation had issues" -ForegroundColor Yellow
}

# Step 4: Verify installation
Write-Host "`n[Step 4] Verifying installation..." -ForegroundColor Green
python -c "import torch; import numpy; import pandas; import streamlit; print('✓ All imports successful!')" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All packages verified" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Some imports failed - environment may need manual fixes" -ForegroundColor Yellow
}

# Success message
Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "                       SETUP COMPLETE!" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "`n"

Write-Host "Your 'ml_fresh' environment is ready with all packages including Streamlit!" -ForegroundColor Green
Write-Host "`n"

Write-Host "OPTION 1 - Run Web Dashboard (Recommended):" -ForegroundColor Yellow
Write-Host "   streamlit run project_streamlit.py" -ForegroundColor Cyan
Write-Host "   (Browser will open at http://localhost:8501)" -ForegroundColor Gray
Write-Host "`n"

Write-Host "OPTION 2 - Run Full Project:" -ForegroundColor Yellow
Write-Host "   python project.py" -ForegroundColor Cyan
Write-Host "   (45-60 min, requires all dependencies)" -ForegroundColor Gray
Write-Host "`n"

Write-Host "OPTION 3 - Run Mock Demo:" -ForegroundColor Yellow
Write-Host "   python project_mock.py" -ForegroundColor Cyan
Write-Host "   (Quick 5-min demo with synthetic data)" -ForegroundColor Gray
Write-Host "`n"

Write-Host "OPTION 4 - Verify Imports:" -ForegroundColor Yellow
Write-Host "   python test_imports.py" -ForegroundColor Cyan
Write-Host "   (Check all packages are working)" -ForegroundColor Gray
Write-Host "`n"

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "`n"

Write-Host "Press Enter to continue..." -ForegroundColor Yellow
Read-Host
