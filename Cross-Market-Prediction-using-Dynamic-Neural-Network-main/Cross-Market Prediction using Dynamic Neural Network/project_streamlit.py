"""
Streamlit Web App - Cross-Market Prediction using Dynamic Neural Network
Run with: streamlit run project_streamlit.py
"""

import streamlit as st  # type: ignore
import random
import math
from typing import Dict, List, Tuple
import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore

# Configure page
st.set_page_config(
    page_title="Cross-Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Cross-Market Volatility Prediction")
st.markdown("""
    Analyzing global stock indices for volatility patterns using Graph Neural Networks
    and Machine Learning. This demo uses synthetic data for instant results.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section:",
    ["🏠 Home", "📈 Data Analysis", "📊 Statistics", "🔗 Correlations", "🤖 Models", "📋 About"]
)

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

# Generate synthetic data
@st.cache_data
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

@st.cache_data
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

# Page: Home
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to the Project Dashboard")
        st.write("""
        This interactive web application demonstrates a **Cross-Market Volatility Prediction System**
        that analyzes global stock indices using advanced neural network architectures.
        
        ### Key Features:
        - 📊 **Real-time Data Analysis** of 8 global stock indices
        - 📈 **Volatility Calculation** using 21-day rolling windows
        - 🔗 **Correlation Analysis** to understand market relationships
        - 🤖 **Neural Network Models** (GCN + GAT architectures)
        - 📋 **Comprehensive Metrics** (MAPE, RMSE, MSE, MAFE)
        """)
    
    with col2:
        st.info("""
        ### Stock Indices Analyzed:
        - 🇺🇸 S&P 500 (^GSPC)
        - 🇩🇪 DAX (^GDAXI)
        - 🇫🇷 CAC 40 (^FCHI)
        - 🇬🇧 FTSE 100 (^FTSE)
        - 🇮🇳 Nifty 50 (^NSEI)
        - 🇯🇵 Nikkei 225 (^N225)
        - 🇰🇷 KOSPI (^KS11)
        - 🇭🇰 Hang Seng (^HSI)
        """)
    
    st.divider()
    
    st.subheader("Quick Links")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📈 View Data Analysis"):
            st.session_state.page = "📈 Data Analysis"
    with col2:
        if st.button("📊 See Statistics"):
            st.session_state.page = "📊 Statistics"
    with col3:
        if st.button("🔗 Check Correlations"):
            st.session_state.page = "🔗 Correlations"
    with col4:
        if st.button("🤖 Explore Models"):
            st.session_state.page = "🤖 Models"

# Page: Data Analysis
elif page == "📈 Data Analysis":
    st.header("Data Analysis")
    
    stock_data = generate_stock_data()
    realized_vol = calculate_volatility(stock_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Ticker")
        selected_ticker = st.selectbox("Choose a stock index:", list(stock_data.keys()))
    
    with col2:
        st.subheader("Data Summary")
        vol_series = realized_vol[selected_ticker]
        st.metric("Mean Volatility", f"{vol_series.mean():.6f}")
        st.metric("Std Deviation", f"{vol_series.std():.6f}")
        st.metric("Data Points", f"{len(vol_series)}")
    
    # Plot volatility
    st.subheader(f"Realized Volatility - {selected_ticker}")
    
    vol_data = realized_vol[selected_ticker].data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=vol_data,
        mode='lines',
        name='Realized Volatility',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        title=f"21-Day Realized Volatility for {selected_ticker}",
        xaxis_title="Time Period",
        yaxis_title="Volatility",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot close prices
    st.subheader(f"Close Prices - {selected_ticker}")
    
    close_prices = stock_data[selected_ticker]['Close'].data
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=close_prices,
        mode='lines',
        name='Close Price',
        line=dict(color='#ff7f0e', width=1.5)
    ))
    fig2.update_layout(
        title=f"Historical Close Prices for {selected_ticker}",
        xaxis_title="Trading Days",
        yaxis_title="Price",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

# Page: Statistics
elif page == "📊 Statistics":
    st.header("Descriptive Statistics")
    
    stock_data = generate_stock_data()
    realized_vol = calculate_volatility(stock_data)
    
    # Create statistics table
    stats_data = []
    for ticker, vol_series in realized_vol.items():
        stats_data.append({
            'Ticker': ticker,
            'Mean': f"{vol_series.mean():.6f}",
            'Std Dev': f"{vol_series.std():.6f}",
            'Min': f"{min(vol_series.data):.6f}",
            'Max': f"{max(vol_series.data):.6f}",
            'Data Points': len(vol_series)
        })
    
    st.subheader("Volatility Statistics Summary")
    st.dataframe(stats_data, use_container_width=True)
    
    # Visualization of statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mean Volatility by Index")
        means = [realized_vol[t].mean() for t in stock_data.keys()]
        tickers = list(stock_data.keys())
        
        fig = px.bar(
            x=tickers,
            y=means,
            title="Average Volatility by Stock Index",
            labels={'y': 'Mean Volatility', 'x': 'Ticker'},
            color=means,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Standard Deviation by Index")
        stds = [realized_vol[t].std() for t in stock_data.keys()]
        
        fig = px.bar(
            x=tickers,
            y=stds,
            title="Volatility Std Dev by Stock Index",
            labels={'y': 'Std Deviation', 'x': 'Ticker'},
            color=stds,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

# Page: Correlations
elif page == "🔗 Correlations":
    st.header("Cross-Market Correlations")
    
    stock_data = generate_stock_data()
    realized_vol = calculate_volatility(stock_data)
    
    def compute_correlations(vol_dict):
        """Compute pairwise correlations."""
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
    
    st.subheader("Volatility Spillover Matrix")
    
    # Create correlation matrix for heatmap
    tickers = list(stock_data.keys())
    corr_matrix = [[0.0] * len(tickers) for _ in range(len(tickers))]
    
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i == j:
                corr_matrix[i][j] = 1.0
            elif (t1, t2) in correlations:
                corr_matrix[i][j] = correlations[(t1, t2)]
            elif (t2, t1) in correlations:
                corr_matrix[i][j] = correlations[(t2, t1)]
    
    # Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    fig.update_layout(
        title="Volatility Spillover Correlation Matrix",
        width=700,
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations
    st.subheader("Strongest Spillover Effects (Top 10)")
    
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    top_corr_data = []
    for (from_t, to_t), corr in sorted_corrs:
        top_corr_data.append({
            'From': from_t,
            'To': to_t,
            'Correlation': f"{corr:.4f}",
            'Strength': 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
        })
    
    st.dataframe(top_corr_data, use_container_width=True)

# Page: Models
elif page == "🤖 Models":
    st.header("Neural Network Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 GCN + GAT Model")
        st.markdown("""
        **Architecture:**
        - Graph Convolutional Network (GCN) layers
        - Graph Attention Network (GAT) layers
        - Hybrid approach for market structure learning
        
        **Features:**
        - Learns market topology from data
        - Attention mechanism for edge importance
        - 50 epochs training
        - Dropout regularization
        
        **Hyperparameter Search:**
        - Hidden dimensions: [32, 64, 128]
        - Learning rates: [0.0001, 0.001, 0.01]
        - Dropout rates: [0.1, 0.3, 0.5]
        """)
        
        st.info("🎯 Best for: Understanding market structure and attention weights")
    
    with col2:
        st.subheader("📈 Baseline MLP Model")
        st.markdown("""
        **Architecture:**
        - 3 Fully Connected Layers
        - ReLU activation functions
        - Dropout regularization
        
        **Features:**
        - Simple benchmark model
        - Fast training
        - Good baseline performance
        - Easy to interpret
        
        **Hyperparameter Search:**
        - Hidden dimensions: [32, 64, 128]
        - Learning rates: [0.0001, 0.001, 0.01]
        - Dropout rates: [0.3, 0.5, 0.7]
        """)
        
        st.info("🎯 Best for: Quick baseline comparison")
    
    st.divider()
    
    st.subheader("Prediction Metrics")
    st.markdown("""
    Models are evaluated at multiple forecast horizons:
    
    | Metric | Definition |
    |--------|-----------|
    | **MAFE** | Mean Absolute Forecast Error |
    | **MSE** | Mean Squared Error |
    | **RMSE** | Root Mean Squared Error |
    | **MAPE** | Mean Absolute Percentage Error |
    
    ### Forecast Horizons
    - **1-day ahead** - Next trading day
    - **5-day ahead** - Weekly forecast
    - **10-day ahead** - Two-week forecast
    - **22-day ahead** - Monthly forecast
    """)
    
    # Mock performance comparison
    st.subheader("Model Performance Comparison (Mock Results)")
    
    model_perf = {
        'Horizon': ['1-day', '5-day', '10-day', '22-day'],
        'GCN+GAT RMSE': [0.0234, 0.0412, 0.0621, 0.0845],
        'MLP RMSE': [0.0285, 0.0502, 0.0745, 0.0921]
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=model_perf['Horizon'],
        y=model_perf['GCN+GAT RMSE'],
        name='GCN + GAT',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    fig.add_trace(go.Scatter(
        x=model_perf['Horizon'],
        y=model_perf['MLP RMSE'],
        name='Baseline MLP',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=10)
    ))
    fig.update_layout(
        title="RMSE Comparison Across Forecast Horizons",
        xaxis_title="Forecast Horizon",
        yaxis_title="RMSE",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Page: About
elif page == "📋 About":
    st.header("About This Project")
    
    st.markdown("""
    ## Cross-Market Prediction Using Dynamic Neural Network
    
    ### Overview
    This project develops a sophisticated forecasting system for global market volatility
    using advanced machine learning techniques, specifically Graph Neural Networks.
    
    ### Research Focus
    - **Volatility Spillover**: Understanding how shocks in one market affect others
    - **Graph-based Learning**: Using network structure to improve predictions
    - **Attention Mechanisms**: Learning which market relationships matter most
    
    ### Data
    - **Time Period**: 2007-2022 (15 years of historical data)
    - **Indices**: 8 major global stock market indices
    - **Frequency**: Daily trading data
    - **Preprocessing**: Forward-fill missing values, aligned to business days
    
    ### Methodology
    
    #### 1. Feature Engineering
    - **Realized Volatility**: 21-day rolling window calculation
    - **Returns**: Log returns from close prices
    - **Spillover Index**: VAR-FEVD based volatility connections
    
    #### 2. Graph Construction
    - Nodes: Stock indices
    - Edges: Volatility spillover relationships
    - Weights: Spillover strength
    
    #### 3. Model Architecture
    
    **GCN + GAT (Primary Model)**
    - Graph Convolutional Network for feature learning
    - Graph Attention Network for edge importance
    - Hybrid architecture to capture market structure
    
    **Baseline MLP**
    - 3-layer fully connected network
    - Standard benchmark for comparison
    
    #### 4. Evaluation
    - Multiple forecast horizons: 1, 5, 10, 22 days
    - Standard metrics: RMSE, MAPE, MAE, MSE
    - Cross-validation on train/val/test splits
    
    ### Results Expected
    - GCN+GAT outperforms baseline on longer horizons
    - Attention weights reveal market topology
    - Spillover effects are time-varying
    - Global synchronization during crises
    
    ### Technologies Used
    - **PyTorch**: Deep learning framework
    - **PyTorch Geometric**: Graph neural networks
    - **Pandas/NumPy**: Data processing
    - **Scikit-learn**: Machine learning utilities
    - **Streamlit**: Interactive web interface
    
    ### Author Notes
    This project demonstrates the application of modern deep learning techniques
    to financial forecasting. The use of GNNs is particularly suited for this problem
    as it naturally captures the interconnected nature of global financial markets.
    """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Data**: 3,700 days × 8 indices")
    with col2:
        st.info("🤖 **Models**: 2 architectures compared")
    with col3:
        st.info("📈 **Horizons**: 4 forecast periods")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("© 2025 Cross-Market Prediction Project")
with col2:
    st.caption("Built with Streamlit & PyTorch")
with col3:
    st.caption("Updated: December 2, 2025")
