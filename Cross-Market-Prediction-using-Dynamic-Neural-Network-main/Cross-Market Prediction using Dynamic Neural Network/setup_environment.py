#!/usr/bin/env python
"""
Automated setup script for Cross-Market Prediction project.
This script creates a fresh conda environment and installs all dependencies.
"""
import subprocess
import sys
import os
import platform

def run_command(cmd, description, shell=True):
    """Execute a command and report status."""
    print(f"\n{'='*70}")
    print(f"[{description}]")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✓ {description} - SUCCESS")
            return True
        else:
            print(f"✗ {description} - FAILED (Return code: {result.returncode})")
            return False
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False

def main():
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║         Cross-Market Prediction - Automated Setup                      ║
    ║                                                                        ║
    ║  This script will:                                                    ║
    ║  1. Create a fresh conda environment 'ml_fresh'                       ║
    ║  2. Install all required scientific computing packages                ║
    ║  3. Install PyTorch and PyTorch Geometric                             ║
    ║  4. Ready your environment to run project.py                          ║
    ║                                                                        ║
    ║  Time required: 5-10 minutes                                          ║
    ║  Disk space needed: ~2-3 GB                                           ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if conda is available
    print("\n[Step 0] Checking for conda installation...")
    try:
        subprocess.run("conda --version", shell=True, capture_output=True, check=True)
        print("✓ Conda found")
    except subprocess.CalledProcessError:
        print("✗ Conda not found. Please install Anaconda or Miniconda first.")
        sys.exit(1)
    
    # Step 1: Create environment
    if not run_command(
        'conda create -n ml_fresh python=3.9 numpy pandas scipy scikit-learn matplotlib seaborn networkx statsmodels yfinance plotly -c conda-forge -y',
        "Creating fresh conda environment 'ml_fresh'"
    ):
        print("\n⚠ Environment creation failed. Trying again with fewer options...")
        run_command(
            'conda create -n ml_fresh python=3.9 -y',
            "Creating minimal Python 3.9 environment"
        )
    
    # Step 2: Determine conda activation script based on OS
    if platform.system() == "Windows":
        activate_cmd = "conda activate ml_fresh && "
        separator = " && "
    else:
        activate_cmd = "source activate ml_fresh && "
        separator = " && "
    
    # Step 3: Install PyTorch with CPU support
    pytorch_cmd = activate_cmd + "conda install pytorch::pytorch torchvision torchaudio -c pytorch -y"
    if not run_command(pytorch_cmd, "Installing PyTorch (CPU version)"):
        print("\n⚠ PyTorch installation failed. Trying alternative approach...")
        pytorch_cmd_alt = activate_cmd + "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        run_command(pytorch_cmd_alt, "Installing PyTorch via pip")
    
    # Step 4: Install PyTorch Geometric
    pyg_cmd = activate_cmd + "pip install torch-geometric"
    run_command(pyg_cmd, "Installing PyTorch Geometric")
    
    # Step 5: Verify installation
    verify_cmd = activate_cmd + "python -c \"import torch; import numpy; import pandas; print('All imports successful!')\""
    print("\n[Step 5] Verifying installation...")
    run_command(verify_cmd, "Verifying all packages can be imported")
    
    # Final summary
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                      SETUP COMPLETE!                                  ║
    ╚════════════════════════════════════════════════════════════════════════╝
    
    Next steps:
    
    1. Activate the new environment:
       conda activate ml_fresh
    
    2. Run the project:
       cd "C:\\Users\\AIML\\Desktop\\Cross-Market Prediction using Dynamic Neural Network"
       python project.py
    
    3. Or test first with the mock version:
       python project_mock.py
    
    4. To verify imports work:
       python test_imports.py
    
    ✓ Your environment is ready!
    """)

if __name__ == "__main__":
    main()
