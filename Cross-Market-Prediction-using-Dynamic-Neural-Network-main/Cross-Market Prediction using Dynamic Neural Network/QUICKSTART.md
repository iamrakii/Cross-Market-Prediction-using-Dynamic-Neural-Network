# Quick Start Guide

## You Asked For Automated Setup

I've created **3 setup scripts** for you:

### For Windows Users:

#### Option 1: Batch Script (Simplest)
```bash
Double-click: setup.bat
```
Or from PowerShell/CMD:
```bash
cd "C:\Users\AIML\Desktop\Cross-Market Prediction using Dynamic Neural Network"
setup.bat
```

#### Option 2: PowerShell Script
```bash
powershell -ExecutionPolicy Bypass -File setup_environment.ps1
```

#### Option 3: Python Script  
```bash
python setup_environment.py
```

---

## What These Scripts Do

1. ✓ Create fresh conda environment `ml_fresh`
2. ✓ Install NumPy, Pandas, SciPy, Scikit-learn
3. ✓ Install Matplotlib, Seaborn, NetworkX
4. ✓ Install YFinance, Plotly, Statsmodels
5. ✓ Install PyTorch (CPU version)
6. ✓ Install PyTorch Geometric
7. ✓ Verify all imports work

**Total Time**: 5-10 minutes  
**Disk Space**: 2-3 GB

---

## After Setup Completes

### In a New Terminal/Command Prompt:

```bash
# Activate the environment
conda activate ml_fresh

# Navigate to project
cd "C:\Users\AIML\Desktop\Cross-Market Prediction using Dynamic Neural Network"

# Run your project
python project.py
```

---

## Alternative: Manual Steps

If scripts don't work, run these commands manually:

```bash
# Create environment
conda create -n ml_fresh python=3.9 numpy pandas scipy scikit-learn matplotlib seaborn networkx statsmodels yfinance plotly -c conda-forge -y

# Activate
conda activate ml_fresh

# Install PyTorch
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y

# Install PyTorch Geometric
pip install torch-geometric

# Run project
python project.py
```

---

## Test If It Works

Before running the full project, test:

```bash
# Activate environment
conda activate ml_fresh

# Test imports
python test_imports.py

# Or run mock version (no downloads)
python project_mock.py
```

---

## Files Created

| File | Purpose |
|------|---------|
| `setup.bat` | Windows batch setup (recommended) |
| `setup_environment.ps1` | PowerShell setup script |
| `setup_environment.py` | Python setup script |
| `test_imports.py` | Verify packages are installed |
| `project_mock.py` | Demo with synthetic data |
| `project.py` | Your main project (fixed code) |

---

## Troubleshooting

### "Conda not found"
- Install Anaconda: https://www.anaconda.com/download
- Or use Python venv: `python -m venv myenv`

### Setup hangs
- Let it run - large downloads take time
- Check internet connection
- Close other applications using conda

### Import errors remain
- Close and reopen terminal after setup
- Try: `conda activate ml_fresh` again
- Check: `python -c "import numpy; print('OK')"`

---

**Status**: Your code is fixed ✓  
**Next Step**: Run one of the setup scripts above ↑

Good luck!
