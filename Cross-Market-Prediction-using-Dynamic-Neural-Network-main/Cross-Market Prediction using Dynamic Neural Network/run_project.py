#!/usr/bin/env python
"""
Wrapper script to run the project with better error handling.
"""
import sys
import os

# Set environment variables to help with library loading
os.environ['MKL_THREADING_LAYER'] = 'GNU'

try:
    # Try to import the main project module
    import project
    print("Project executed successfully!")
except ImportError as e:
    print(f"Import Error: {e}")
    print("\nTroubleshooting steps:")
    print("1. The NumPy DLL issue suggests environment corruption")
    print("2. Try creating a fresh conda environment:")
    print("   conda create -n fresh_env python=3.9 numpy pandas matplotlib scipy scikit-learn seaborn networkx yfinance plotly")
    print("3. Then install PyTorch and PyTorch Geometric:")
    print("   conda activate fresh_env")
    print("   pip install torch torch-geometric")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
