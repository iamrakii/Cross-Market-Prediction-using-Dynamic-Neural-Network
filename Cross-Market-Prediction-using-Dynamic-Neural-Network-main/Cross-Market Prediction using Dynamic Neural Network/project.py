from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import yfinance as yf  # type: ignore
import plotly.express as px  # type: ignore
from scipy.stats import skew, kurtosis  # type: ignore
from statsmodels.tsa.stattools import adfuller  # type: ignore
from statsmodels.tsa.api import VAR  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import torch  # type: ignore
from torch_geometric.utils import from_networkx  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch_geometric.nn import GATConv, GCNConv  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
import torch.nn as nn  # type: ignore
import itertools

# =========================
# 1. DATA FETCHING & CLEAN
# =========================

# Define the stock tickers
tickers = ['^GSPC', '^GDAXI', '^FCHI', '^FTSE', '^NSEI', '^N225', '^KS11', '^HSI']

# Define the start and end dates
start_date = '2007-11-06'
end_date = '2022-06-03'


def fetch_and_fill_data(
    symbol: str,
    start: str,
    end: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Fetch historical data from yfinance, align to full business-day range,
    forward/backward fill, and also return count of filled cells.
    """
    try:
        data = yf.download(symbol, start=start, end=end)  # type: ignore

        # Handle None or empty DataFrame explicitly
        if data is None or data.empty:
            print(f"No data returned for {symbol}")
            return None, None

        # Full date range (business days)
        full_range = pd.date_range(start=start, end=end, freq='B')

        original_data = data.copy()
        data = data.reindex(full_range)

        data_filled = data.ffill().bfill()

        # Count how many values were filled (per column)
        filled_days_count = data_filled.notna().sum() - original_data.notna().sum()

        return data_filled, filled_days_count
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None, None


# Dictionary to store the processed data and filled days count
stock_data: Dict[str, pd.DataFrame] = {}
filled_days_counts: Dict[str, pd.Series] = {}

# Fetch and fill data for each ticker
for ticker in tickers:
    data, filled_days_count = fetch_and_fill_data(ticker, start_date, end_date)
    if data is not None and filled_days_count is not None:
        stock_data[ticker] = data
        filled_days_counts[ticker] = filled_days_count
        data.to_csv(f"{ticker}_stock_data.csv")

# Plot the 'Close' price of each ticker in separate graphs
for ticker, data in stock_data.items():
    plt.figure(figsize=(14, 7))  # type: ignore
    plt.plot(data.index, data['Close'], label=f'{ticker} Close Price')  # type: ignore
    plt.title(f'{ticker} Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Print the head, total count of each processed DataFrame, and the count of filled values
for ticker, data in stock_data.items():
    print(f'Head of {ticker} data:')
    print(data.head(), '\n')
    print(f'Total count of trading days for {ticker}: {len(data)}\n')
    print(f'Count of forward-filled or backward-filled days for {ticker}:')
    print(filled_days_counts[ticker], '\n')

for ticker, data in stock_data.items():
    print(f"Ticker: {ticker}, Total number of days: {data.shape[0]}, Total number of fields: {data.shape[1]}")

# Ensure datetime index
for ticker, data in stock_data.items():
    data.index = pd.to_datetime(data.index)

for ticker, data in stock_data.items():
    print(f"Ticker: {ticker}")
    print(data.head())
    print("\n")

# =========================
# 2. CLOSE SERIES & PLOTS
# =========================

ticker_close_dict: Dict[str, pd.Series] = {}

for ticker, data in stock_data.items():
    ticker_close_series = data['Close']
    ticker_close_dict[ticker] = ticker_close_series

for ticker, series in ticker_close_dict.items():
    print(f"Ticker: {ticker}")
    print(series.head())
    print(f"Length: {len(series)}\n")

# Plot with Plotly
for ticker, series in ticker_close_dict.items():
    df = series.reset_index()
    df.columns = ['date', 'Close']

    fig = px.line(df, x='date', y='Close',
                  labels={'date': 'Date', 'close': 'Close Stock'})
    fig.update_traces(marker_line_width=2, opacity=0.8)
    fig.update_layout(
        title_text=f'Stock Close Price Chart for {ticker}',
        plot_bgcolor='white',
        font_size=15,
        font_color='black'
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

# =========================
# 3. REALIZED VOLATILITY
# =========================

def calculate_realized_volatility(data: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Calculate realized volatility for the given data using squared returns.
    """
    returns = data['Close'].pct_change()
    squared_returns = returns ** 2
    realized_variance = squared_returns.rolling(window=window).sum()
    realized_volatility = np.sqrt(realized_variance)
    return realized_volatility.dropna()


realized_vol_dict: Dict[str, pd.Series] = {}
for ticker, data in stock_data.items():
    realized_volatility = calculate_realized_volatility(data)
    realized_vol_dict[ticker] = realized_volatility

for ticker, series in realized_vol_dict.items():
    print(f"Ticker: {ticker}")
    print(series.head())
    print(f"Length: {len(series)}\n")

# Plot realized volatility
for ticker, series in realized_vol_dict.items():
    df = series.reset_index()
    df.columns = ['date', 'realized_volatility']

    fig = px.line(df, x='date', y='realized_volatility',
                  labels={'date': 'Date', 'realized_volatility': 'Realized Volatility'})
    fig.update_traces(marker_line_width=2, opacity=0.8)
    fig.update_layout(
        title_text=f'Realized Volatility Chart for {ticker}',
        plot_bgcolor='white',
        font_size=15,
        font_color='black'
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()

# =========================
# 4. DESCRIPTIVE STATS
# =========================

def calculate_descriptive_statistics(realized_volatility: pd.Series) -> dict:
    """
    Calculate descriptive statistics for the given realized volatility series.
    """
    mean_value = realized_volatility.mean()
    std_dev = realized_volatility.std()
    skewness_value = skew(realized_volatility)
    kurtosis_value = kurtosis(realized_volatility, fisher=False)
    adf_result = adfuller(realized_volatility.dropna())
    adf_statistic = adf_result[0]
    adf_p_value = adf_result[1]

    return {
        'Mean': mean_value,
        'Standard Deviation': std_dev,
        'Skewness': skewness_value,
        'Kurtosis': kurtosis_value,
        'ADF Statistic': adf_statistic,
        'ADF p-value': adf_p_value
    }


descriptive_stats_dict: Dict[str, dict] = {}
for ticker, series in realized_vol_dict.items():
    descriptive_stats = calculate_descriptive_statistics(series)
    descriptive_stats_dict[ticker] = descriptive_stats

for ticker, stats in descriptive_stats_dict.items():
    print(f"Descriptive Statistics for {ticker}:")
    for stat_name, value in stats.items():
        print(f"{stat_name}: {value}")
    print()

# =========================
# 5. TRAIN / VAL / TEST SPLITS
# =========================

data_splits: Dict[str, dict] = {}

for ticker, realized_volatility in realized_vol_dict.items():
    df = realized_volatility.reset_index()
    df.columns = ['date', 'realized_volatility']

    train_data, temp_data = train_test_split(df, test_size=0.5, shuffle=False)
    validation_data, test_data = train_test_split(temp_data, test_size=0.6, shuffle=False)

    data_splits[ticker] = {
        'train': train_data,
        'validation': validation_data,
        'test': test_data
    }

for ticker, splits in data_splits.items():
    print(f"Ticker: {ticker}")
    print(f"Training Set Size: {len(splits['train'])}")
    print(f"Validation Set Size: {len(splits['validation'])}")
    print(f"Test Set Size: {len(splits['test'])}\n")

# =========================
# 6. SPILLOVER (VAR-FEVD)
# =========================

def calculate_spillover_index(realized_vol_dict: Dict[str, pd.Series],
                              lag_order: int = 2,
                              forecast_horizon: int = 10) -> pd.DataFrame:
    """
    Calculate the volatility spillover index using the Diebold-Yilmaz methodology.
    """
    combined_data = pd.DataFrame(realized_vol_dict).dropna()

    model = VAR(combined_data)
    var_result = model.fit(lag_order)
    fevd = var_result.fevd(forecast_horizon)

    n = len(realized_vol_dict)
    spillover_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                spillover_matrix[i, j] = fevd.decomp[j][:, i].sum() / fevd.decomp[j].sum()

    spillover_index = pd.DataFrame(
        spillover_matrix,
        index=realized_vol_dict.keys(),
        columns=realized_vol_dict.keys()
    )
    return spillover_index * 100


train_realized_vol_dict: Dict[str, pd.Series] = {
    ticker: splits['train']['realized_volatility']
    for ticker, splits in data_splits.items()
}

spillover_index_train = calculate_spillover_index(train_realized_vol_dict)

print("Spillover Index Matrix (Training Data):")
print(spillover_index_train)

n = len(train_realized_vol_dict)
total_spillover_index_train = spillover_index_train.to_numpy().sum() / (n**2 - n)
print(f"\nTotal Spillover Index (Training Data): {total_spillover_index_train:.2f}%")

plt.figure(figsize=(10, 8))  # type: ignore
plt.title("Volatility Spillover Index Heatmap (Training Data)")
sns.heatmap(spillover_index_train, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.show()


def create_spillover_graph(spillover_matrix: pd.DataFrame) -> nx.DiGraph:
    """
    Create a directed graph from the spillover matrix.
    """
    G = nx.DiGraph()

    for node in spillover_matrix.columns:
        G.add_node(node)

    for i, source in enumerate(spillover_matrix.columns):
        for j, target in enumerate(spillover_matrix.columns):
            if i != j:
                weight = spillover_matrix.iloc[i, j]
                if weight > 0:
                    G.add_edge(source, target, weight=weight)

    return G


G = create_spillover_graph(spillover_index_train)

plt.figure(figsize=(12, 8))  # type: ignore
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000,
        font_size=12, font_weight='bold', edge_color='gray', width=2)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
plt.title('Volatility Spillover Directed Graph (Training Data)')
plt.show()

# Test & Validation spillovers
test_realized_vol_dict: Dict[str, pd.Series] = {
    ticker: splits['test']['realized_volatility']
    for ticker, splits in data_splits.items()
}
validation_realized_vol_dict: Dict[str, pd.Series] = {
    ticker: splits['validation']['realized_volatility']
    for ticker, splits in data_splits.items()
}

spillover_index_test = calculate_spillover_index(test_realized_vol_dict)
print("Spillover Index Matrix (Test Data):")
print(spillover_index_test)

n_test = len(test_realized_vol_dict)
total_spillover_index_test = spillover_index_test.to_numpy().sum() / (n_test**2 - n_test)
print(f"\nTotal Spillover Index (Test Data): {total_spillover_index_test:.2f}%")

plt.figure(figsize=(10, 8))  # type: ignore
plt.title("Volatility Spillover Index Heatmap (Test Data)")
sns.heatmap(spillover_index_test, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.show()

G_test = create_spillover_graph(spillover_index_test)
plt.figure(figsize=(12, 8))  # type: ignore
pos_test = nx.spring_layout(G_test)
nx.draw(G_test, pos_test, with_labels=True, node_color='lightgreen', node_size=3000,
        font_size=12, font_weight='bold', edge_color='gray', width=2)
edge_labels_test = nx.get_edge_attributes(G_test, 'weight')
nx.draw_networkx_edge_labels(G_test, pos_test, edge_labels=edge_labels_test, font_size=10)
plt.title('Volatility Spillover Directed Graph (Test Data)')
plt.show()

spillover_index_validation = calculate_spillover_index(validation_realized_vol_dict)
print("Spillover Index Matrix (Validation Data):")
print(spillover_index_validation)

n_val = len(validation_realized_vol_dict)
total_spillover_index_validation = spillover_index_validation.to_numpy().sum() / (n_val**2 - n_val)
print(f"\nTotal Spillover Index (Validation Data): {total_spillover_index_validation:.2f}%")

plt.figure(figsize=(10, 8))  # type: ignore
plt.title("Volatility Spillover Index Heatmap (Validation Data)")
sns.heatmap(spillover_index_validation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.show()

G_validation = create_spillover_graph(spillover_index_validation)
plt.figure(figsize=(12, 8))  # type: ignore
pos_validation = nx.spring_layout(G_validation)
nx.draw(G_validation, pos_validation, with_labels=True, node_color='lightcoral', node_size=3000,
        font_size=12, font_weight='bold', edge_color='gray', width=2)
edge_labels_validation = nx.get_edge_attributes(G_validation, 'weight')
nx.draw_networkx_edge_labels(G_validation, pos_validation, edge_labels=edge_labels_validation, font_size=10)
plt.title('Volatility Spillover Directed Graph (Validation Data)')
plt.show()

# =========================
# 7. CONVERT TO PYTORCH GEOMETRIC DATA
# =========================

def networkx_to_pyg_data(G: nx.Graph, realized_vol_dict: Dict[str, pd.Series]):
    """
    Convert a NetworkX graph and realized volatility dictionary
    into PyTorch Geometric Data format.
    """
    data = from_networkx(G)

    node_features = []
    for node in G.nodes():
        vol_series = realized_vol_dict[node].values
        node_features.append(torch.tensor(vol_series, dtype=torch.float).view(-1, 1))

    data.x = torch.cat(node_features, dim=1)
    return data


train_data = networkx_to_pyg_data(G, train_realized_vol_dict)
validation_data = networkx_to_pyg_data(G_validation, validation_realized_vol_dict)
test_data = networkx_to_pyg_data(G_test, test_realized_vol_dict)

# =========================
# 8. GCN + GAT MODEL + GRID SEARCH
# =========================

class GCN_GAT_Model(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_heads, num_layers=2, dropout_p=0.5):
        super(GCN_GAT_Model, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = torch.nn.ModuleList()
        self.gat_layers = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=dropout_p)

        self.gcn_layers.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        for _ in range(num_layers):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False))

        self.fc = torch.nn.Linear(hidden_dim, 8)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        out = self.fc(x)
        return out


node_feature_dim = train_data.x.shape[1]
hidden_dim_list = [32, 64]
num_heads_list = [2, 4]
num_layers_list = [2, 3]
learning_rates = [0.001, 0.0005]
dropout_rates = [0.1, 0.3]

param_combinations = list(itertools.product(
    hidden_dim_list, num_heads_list, num_layers_list, learning_rates, dropout_rates
))
best_val_loss = float('inf')
best_params = None

all_train_loss_values = []
all_validation_loss_values = []

criterion = torch.nn.MSELoss()

for params in param_combinations:
    hidden_dim, num_heads, num_layers, lr, dropout_rate = params

    model = GCN_GAT_Model(node_feature_dim, hidden_dim, num_heads, num_layers, dropout_p=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_values = []
    validation_loss_values = []
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data)
        loss = criterion(out[:-1], train_data.x[1:])
        loss.backward()
        optimizer.step()
        train_loss_values.append(loss.item())

        model.eval()
        with torch.no_grad():
            validation_out = model(validation_data)
            validation_loss = criterion(validation_out[:-1], validation_data.x[1:])
            validation_loss_values.append(validation_loss.item())

        if epoch % 10 == 0:
            print(f"[GCN+GAT] Params {params}, Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {validation_loss.item():.6f}")

    all_train_loss_values.append(train_loss_values)
    all_validation_loss_values.append(validation_loss_values)

    final_val_loss = validation_loss_values[-1]
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_params = params

best_index = param_combinations.index(best_params)

plt.figure(figsize=(12, 6))  # type: ignore
plt.plot(range(num_epochs), all_train_loss_values[best_index], label='Training Loss', color='blue')  # type: ignore
plt.plot(range(num_epochs), all_validation_loss_values[best_index], label='Validation Loss', color='red')  # type: ignore
plt.title('Training and Validation Loss Over Epochs (Best GCN+GAT Config)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

print(f"Best Hyperparameters (GCN+GAT): Hidden Dim: {best_params[0]}, Num Heads: {best_params[1]}, Num Layers: {best_params[2]}, Learning Rate: {best_params[3]}, Dropout Rate: {best_params[4]}")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

validation_targets = validation_data.x

def calculate_all_metrics(actuals, predictions, h):
    T = actuals.shape[0]
    if predictions.shape[0] > T - h:
        predictions = predictions[:T - h]
    actuals = actuals[h:]

    metrics_list = []

    for idx in range(actuals.shape[1]):
        act = actuals[:, idx].detach().cpu().numpy()
        pred = predictions[:, idx].detach().cpu().numpy()

        mse = mean_squared_error(act, pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(act, pred)
        mafe = np.mean(np.abs(pred - act))

        metrics_list.append({
            'Index': f"Index {idx+1}",
            'MAFE': mafe,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
        })

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df

# Re-train best GCN+GAT model on train_data
best_hidden, best_heads, best_layers, best_lr, best_dropout = best_params
best_model = GCN_GAT_Model(node_feature_dim, best_hidden, best_heads, best_layers, dropout_p=best_dropout)
best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr)

num_epochs = 50
for epoch in range(num_epochs):
    best_model.train()
    best_optimizer.zero_grad()
    out = best_model(train_data)
    loss = criterion(out[:-1], train_data.x[1:])
    loss.backward()
    best_optimizer.step()

best_model.eval()
with torch.no_grad():
    validation_out = best_model(validation_data)

horizons = [1, 5, 10, 22]
metrics_results_per_horizon = {}

for h in horizons:
    metrics_df = calculate_all_metrics(validation_targets, validation_out, h)
    metrics_results_per_horizon[f"Metrics for horizon {h}"] = metrics_df

for horizon, metrics_df in metrics_results_per_horizon.items():
    print(f"{horizon}:")
    print(metrics_df.to_string(index=False))

# =========================
# 9. BASELINE MLP MODEL + GRID SEARCH
# =========================

class BaselineMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(BaselineMLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_final = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x = data.x
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        x = F.relu(self.fc3(x)); x = self.dropout(x)
        out = self.fc_final(x)
        return out

hidden_dim_values = [32, 64, 128]
learning_rate_values = [0.0001, 0.001, 0.01]
dropout_rate_values = [0.3, 0.5, 0.7]

param_combinations = list(itertools.product(hidden_dim_values, learning_rate_values, dropout_rate_values))
best_val_loss = float('inf')
best_params = None
all_train_loss_values = []
all_validation_loss_values = []

for params in param_combinations:
    hidden_dim, lr, dropout_rate = params

    baseline_model = BaselineMLPModel(input_dim=train_data.x.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_loss_values = []
    validation_loss_values = []
    num_epochs = 50
    for epoch in range(num_epochs):
        baseline_model.train()
        optimizer.zero_grad()
        out = baseline_model(train_data)
        loss = criterion(out[:-1], train_data.x[1:])
        loss.backward()
        optimizer.step()
        train_loss_values.append(loss.item())

        baseline_model.eval()
        with torch.no_grad():
            validation_out = baseline_model(validation_data)
            validation_loss = criterion(validation_out[:-1], validation_data.x[1:])
            validation_loss_values.append(validation_loss.item())

        if epoch % 10 == 0:
            print(f"[MLP] Params {params}, Epoch {epoch}, Training Loss: {loss.item():.6f}, Validation Loss: {validation_loss.item():.6f}")

    all_train_loss_values.append(train_loss_values)
    all_validation_loss_values.append(validation_loss_values)

    final_val_loss = validation_loss_values[-1]
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_params = params

best_index = param_combinations.index(best_params)

plt.figure(figsize=(12, 6))  # type: ignore
plt.plot(range(num_epochs), all_train_loss_values[best_index], label='Training Loss', color='blue')  # type: ignore
plt.plot(range(num_epochs), all_validation_loss_values[best_index], label='Validation Loss', color='red')  # type: ignore
plt.title('Training and Validation Loss Over Epochs (Best MLP Config)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

print(f"Best Hyperparameters (MLP): Hidden Dim: {best_params[0]}, Learning Rate: {best_params[1]}, Dropout Rate: {best_params[2]}")

def calculate_metrics_for_all_indices(actuals, predictions, h):
    """
    Calculate evaluation metrics (MAFE, MSE, RMSE, MAPE) for each index at a given horizon h.
    """
    T = actuals.shape[0]
    if predictions.shape[0] > T - h:
        predictions = predictions[:T - h]
    actuals = actuals[h:]

    metrics_results = {}

    for idx in range(actuals.shape[1]):
        actual = actuals[:, idx]
        prediction = predictions[:, idx] if predictions.shape[1] > 1 else predictions.view(-1)

        eps = 1e-8
        mafe = torch.mean(torch.abs(prediction - actual)).item()
        mse = F.mse_loss(prediction, actual).item()
        rmse = torch.sqrt(F.mse_loss(prediction, actual)).item()
        mape = (torch.mean(torch.abs((actual - prediction) / (actual + eps))) * 100).item()

        metrics_results[f"Index {idx+1}"] = {
            "MAFE": mafe,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape
        }

    return metrics_results

validation_targets2 = validation_data.x

baseline_model.eval()
with torch.no_grad():
    validation_out2 = baseline_model(validation_data)

horizons = [1, 5, 10, 22]
metrics_results_per_horizon = {}

for h in horizons:
    metrics_results = calculate_metrics_for_all_indices(validation_targets2, validation_out2, h)
    metrics_results_per_horizon[f"Metrics for horizon {h}"] = metrics_results

for horizon, metrics_by_index in metrics_results_per_horizon.items():
    print(f"Metrics for horizon {horizon}:")
    print(f"  {'Index':<8} {'MAFE':<10} {'MSE':<10} {'RMSE':<10} {'MAPE':<12}")
    for index, metrics in metrics_by_index.items():
        print(f"{index:<8} {metrics['MAFE']:<10.6f} {metrics['MSE']:<10.6f} {metrics['RMSE']:<10.6f} {metrics['MAPE']:<12.6f} ")
