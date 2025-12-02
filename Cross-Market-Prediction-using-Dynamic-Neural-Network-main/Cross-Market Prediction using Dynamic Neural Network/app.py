"""
Flask Web App - Cross-Market Prediction using Dynamic Neural Network
Run with: python app.py
"""

from flask import Flask, render_template, request, jsonify
import random
import math
import json
from typing import Dict, List

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

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

class MockDataFrame:
    def __init__(self, data: Dict[str, List[float]]):
        self.data = data
    
    def __getitem__(self, key):
        return MockSeries(self.data[key])

# Generate synthetic data
def generate_stock_data():
    """Generate synthetic stock data for all tickers."""
    tickers = ['^GSPC', '^GDAXI', '^FCHI', '^FTSE', '^NSEI', '^N225', '^KS11', '^HSI']
    stock_data: Dict[str, MockDataFrame] = {}
    
    random.seed(42)
    
    for ticker in tickers:
        close_prices = []
        price = 100.0
        for _ in range(3700):
            change = random.gauss(0, 2)
            price = max(price + change, 10)
            close_prices.append(price)
        
        stock_data[ticker] = MockDataFrame({
            'Close': close_prices,
            'High': [p * 1.02 for p in close_prices],
            'Low': [p * 0.98 for p in close_prices],
            'Volume': [random.randint(1000000, 10000000) for _ in close_prices]
        })
    
    return stock_data

def calculate_volatility(stock_data):
    """Calculate realized volatility for all tickers."""
    def calculate_realized_volatility(data: MockDataFrame, window: int = 21) -> MockSeries:
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
    
    return realized_vol_dict

# Cache data
stock_data = generate_stock_data()
realized_vol = calculate_volatility(stock_data)
tickers = list(stock_data.keys())

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', tickers=tickers)

@app.route('/api/volatility/<ticker>')
def get_volatility(ticker):
    """Get volatility data for a ticker"""
    if ticker not in realized_vol:
        return {'error': 'Invalid ticker'}, 404
    
    vol_data = realized_vol[ticker].data[:500]  # Limit to 500 points for performance
    return jsonify({
        'ticker': ticker,
        'volatility': vol_data,
        'mean': realized_vol[ticker].mean(),
        'std': realized_vol[ticker].std(),
        'dataPoints': len(realized_vol[ticker].data)
    })

@app.route('/api/prices/<ticker>')
def get_prices(ticker):
    """Get close prices for a ticker"""
    if ticker not in stock_data:
        return {'error': 'Invalid ticker'}, 404
    
    prices = stock_data[ticker]['Close'].data[:500]  # Limit to 500 points
    return jsonify({
        'ticker': ticker,
        'prices': prices,
        'min': min(prices),
        'max': max(prices),
        'latest': prices[-1]
    })

@app.route('/api/statistics')
def get_statistics():
    """Get statistics for all tickers"""
    stats = []
    for ticker in tickers:
        vol = realized_vol[ticker]
        stats.append({
            'ticker': ticker,
            'mean': round(vol.mean(), 6),
            'std': round(vol.std(), 6),
            'min': round(min(vol.data), 6),
            'max': round(max(vol.data), 6),
            'dataPoints': len(vol.data)
        })
    return jsonify(stats)

@app.route('/api/correlations')
def get_correlations():
    """Get correlation matrix"""
    def compute_correlations(vol_dict):
        correlations = {}
        ticker_list = list(vol_dict.keys())
        
        for i, ticker1 in enumerate(ticker_list):
            for j, ticker2 in enumerate(ticker_list):
                if i != j:
                    vol1 = vol_dict[ticker1].data
                    vol2 = vol_dict[ticker2].data
                    
                    min_len = min(len(vol1), len(vol2))
                    vol1 = vol1[:min_len]
                    vol2 = vol2[:min_len]
                    
                    mean1 = sum(vol1) / len(vol1)
                    mean2 = sum(vol2) / len(vol2)
                    
                    cov = sum((vol1[k] - mean1) * (vol2[k] - mean2) for k in range(len(vol1))) / len(vol1)
                    std1 = math.sqrt(sum((v - mean1) ** 2 for v in vol1) / len(vol1))
                    std2 = math.sqrt(sum((v - mean2) ** 2 for v in vol2) / len(vol2))
                    
                    correlation = cov / (std1 * std2) if std1 * std2 > 0 else 0
                    correlations[(ticker1, ticker2)] = correlation
        
        return correlations
    
    correlations = compute_correlations(realized_vol)
    
    # Build correlation matrix
    corr_matrix = [[0.0] * len(tickers) for _ in range(len(tickers))]
    
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i == j:
                corr_matrix[i][j] = 1.0
            elif (t1, t2) in correlations:
                corr_matrix[i][j] = round(correlations[(t1, t2)], 4)
            elif (t2, t1) in correlations:
                corr_matrix[i][j] = round(correlations[(t2, t1)], 4)
    
    return jsonify({
        'tickers': tickers,
        'matrix': corr_matrix
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  Cross-Market Volatility Prediction - Flask Web App")
    print("="*70)
    print("\n✓ Starting server...")
    print("✓ Open your browser and go to: http://localhost:5000")
    print("\nPages available:")
    print("  - Home: http://localhost:5000")
    print("  - Data Analysis: http://localhost:5000#analysis")
    print("  - Statistics: http://localhost:5000#statistics")
    print("  - Correlations: http://localhost:5000#correlations")
    print("\nPress CTRL+C to stop the server\n")
    print("="*70 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000)
