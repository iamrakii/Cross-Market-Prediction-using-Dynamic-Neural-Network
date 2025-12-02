"""
Test script to diagnose the NumPy DLL loading issue
"""
import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("System PATH entries related to conda:")
for path in sys.path:
    if 'anaconda' in path.lower() or 'conda' in path.lower():
        print(f"  {path}")

print("\nAttempting to import pandas (which uses NumPy)...")
try:
    import pandas as pd
    print("✓ Pandas imported successfully")
except Exception as e:
    print(f"✗ Pandas import failed: {e}")

print("\nAttempting to import NumPy directly...")
try:
    import numpy as np
    print("✓ NumPy imported successfully")
    print(f"  NumPy version: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    print("\nSolution: The NumPy binary is corrupted or missing dependencies.")
    print("Try reinstalling with: conda install -c conda-forge numpy")
