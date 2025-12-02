"""
Mock version of the Cross-Market Prediction project using synthetic data.
This version bypasses NumPy/Pandas import issues by using pure Python data structures.
"""
from typing import Dict, List, Tuple, Optional
import itertools
import math

print("=" * 60)
print("Cross-Market Prediction - MOCK VERSION")
print("Using synthetic data (no NumPy/Pandas required)")
print("=" * 60)

# Mock data classes
class MockSeries:
    def __init__(self, data: List[float]):
        self.data = data
    
    def mean(self) -> float:
        return sum(self.data) / len(self.data) if self.data else 0
    
    def std(self) -> float:
        if not self.data:
            return 0
        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self.data) / len(self.data)
        return math.sqrt(variance)
    
    def __len__(self) -> int:
        return len(self.data)

class MockDataFrame:
    def __init__(self, data: Dict[str, List[float]]):
        self.data = data
        self.columns = list(data.keys())
    
    def __getitem__(self, key):
        return MockSeries(self.data[key])

# ==========================================
# SYNTHETIC DATA GENERATION
# ==========================================

print("\n[1/5] Generating synthetic stock data...")

# Create mock stock tickers and synthetic price data
tickers = ['^GSPC', '^GDAXI', '^FCHI', '^FTSE', '^NSEI', '^N225', '^KS11', '^HSI']
stock_data: Dict[str, MockDataFrame] = {}

# Generate synthetic close prices for each ticker
import random
random.seed(42)

for ticker in tickers:
    close_prices = []
    price = 100.0
    for _ in range(3700):  # ~15 years of trading days
        change = random.gauss(0, 2)  # Random walk with drift
        price = max(price + change, 10)  # Keep price positive
        close_prices.append(price)
    
    stock_data[ticker] = MockDataFrame({
        'Close': close_prices,
        'High': [p * 1.02 for p in close_prices],
        'Low': [p * 0.98 for p in close_prices],
        'Volume': [random.randint(1000000, 10000000) for _ in close_prices]
    })

print(f"  ✓ Generated data for {len(stock_data)} tickers")
print(f"  ✓ Each ticker has ~3700 trading days")

# ==========================================
# CALCULATE REALIZED VOLATILITY
# ==========================================

print("\n[2/5] Calculating realized volatility...")

def calculate_realized_volatility(data: MockDataFrame, window: int = 21) -> MockSeries:
    """Calculate realized volatility using price returns."""
    close_prices = data['Close'].data
    returns = []
    for i in range(1, len(close_prices)):
        ret = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
        returns.append(ret)
    
    volatilities = []
    for i in range(len(returns) - window):
        window_returns = returns[i:i+window]
        squared_returns = [r ** 2 for r in window_returns]
        variance = sum(squared_returns) / len(squared_returns)
        volatility = math.sqrt(variance)
        volatilities.append(volatility)
    
    return MockSeries(volatilities)

realized_vol_dict: Dict[str, MockSeries] = {}
for ticker, data in stock_data.items():
    vol = calculate_realized_volatility(data)
    realized_vol_dict[ticker] = vol
    print(f"  ✓ {ticker}: mean volatility = {vol.mean():.6f}")

# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

print("\n[3/5] Computing descriptive statistics...")

stats_dict: Dict[str, Dict[str, float]] = {}
for ticker, vol_series in realized_vol_dict.items():
    stats = {
        'Mean': vol_series.mean(),
        'Std Dev': vol_series.std(),
        'Min': min(vol_series.data) if vol_series.data else 0,
        'Max': max(vol_series.data) if vol_series.data else 0,
    }
    stats_dict[ticker] = stats
    print(f"  ✓ {ticker}: Mean={stats['Mean']:.6f}, Std={stats['Std Dev']:.6f}")

# ==========================================
# DATA SPLITS
# ==========================================

print("\n[4/5] Splitting data for training/validation/testing...")

data_splits: Dict[str, Dict[str, List[float]]] = {}

for ticker, vol_series in realized_vol_dict.items():
    data = vol_series.data
    n = len(data)
    
    train_size = int(n * 0.5)
    val_size = int((n - train_size) * 0.4)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    data_splits[ticker] = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    print(f"  ✓ {ticker}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# ==========================================
# CORRELATION ANALYSIS (Mock Spillover)
# ==========================================

print("\n[5/5] Computing cross-market spillover indices...")

def compute_correlations(vol_dict: Dict[str, MockSeries]) -> Dict[Tuple[str, str], float]:
    """Compute pairwise correlations between volatilities."""
    correlations = {}
    ticker_list = list(vol_dict.keys())
    
    for i, ticker1 in enumerate(ticker_list):
        for j, ticker2 in enumerate(ticker_list):
            if i != j:
                vol1 = vol_dict[ticker1].data
                vol2 = vol_dict[ticker2].data
                
                # Align lengths
                min_len = min(len(vol1), len(vol2))
                vol1 = vol1[:min_len]
                vol2 = vol2[:min_len]
                
                # Compute correlation
                mean1 = sum(vol1) / len(vol1)
                mean2 = sum(vol2) / len(vol2)
                
                cov = sum((vol1[k] - mean1) * (vol2[k] - mean2) for k in range(len(vol1))) / len(vol1)
                std1 = math.sqrt(sum((v - mean1) ** 2 for v in vol1) / len(vol1))
                std2 = math.sqrt(sum((v - mean2) ** 2 for v in vol2) / len(vol2))
                
                correlation = cov / (std1 * std2) if std1 * std2 > 0 else 0
                correlations[(ticker1, ticker2)] = correlation
    
    return correlations

correlations = compute_correlations(realized_vol_dict)

print("\n  Spillover Correlation Matrix:")
print("  " + "-" * 60)
print(f"  {'From':<8} {'To':<8} {'Correlation':<12}")
print("  " + "-" * 60)

for (from_ticker, to_ticker), corr in list(correlations.items())[:5]:
    print(f"  {from_ticker:<8} {to_ticker:<8} {corr:>11.4f}")

print(f"  ... ({len(correlations)} more pairs)")

# ==========================================
# MODEL ARCHITECTURE SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE SUMMARY")
print("=" * 60)

print("""
The full project would use:

1. GCN + GAT Model:
   - Graph Convolutional Network (GCN) layers
   - Graph Attention Network (GAT) layers
   - Hyperparameter grid search
   - Training on 50 epochs

2. Baseline MLP Model:
   - 3 fully connected layers
   - Dropout regularization
   - RMSe, MAPE, MAFE metrics

3. Grid Search Parameters:
   - Hidden dimensions: [32, 64, 128]
   - Learning rates: [0.0001, 0.001, 0.01]
   - Dropout rates: [0.1, 0.3, 0.5]
""")

print("=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("""
To run the full project with real data:

1. Fix NumPy installation (see TROUBLESHOOTING.md)
2. Run: python project.py

Or create a fresh environment:

   conda create -n ml-env python=3.9
   conda activate ml-env
   conda install -c conda-forge numpy pandas scipy scikit-learn
   conda install pytorch::pytorch torchvision torchaudio -c pytorch
   pip install torch-geometric
   python project.py
""")

print("\n✓ Mock demonstration completed successfully!")
print("=" * 60)
