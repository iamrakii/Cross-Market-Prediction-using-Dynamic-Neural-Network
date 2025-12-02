# 🌐 Streamlit Web Interface - Quick Start

## What is Streamlit?

Streamlit is a Python framework that turns data scripts into **interactive web apps** with zero front-end knowledge required. Instead of terminal output, you get:

- 📊 Interactive charts and graphs
- 📋 Beautiful tables and metrics
- 🎯 Easy navigation with sidebar
- 🔄 Real-time interactivity
- 🌐 Works in any web browser

## Installation (3 Steps)

### Step 1: Create Fresh Environment
If you haven't already:

```powershell
setup.bat
```

Wait for the installation to complete (~5-10 minutes).

### Step 2: Install Streamlit & Plotly
After `ml_fresh` environment is ready:

**Option A - Using the provided script:**
```powershell
install_streamlit.bat
```

**Option B - Manual installation:**
```powershell
conda activate ml_fresh
pip install streamlit plotly
```

### Step 3: Launch the Web App
```powershell
conda activate ml_fresh
streamlit run project_streamlit.py
```

✅ **Your browser will automatically open to `http://localhost:8501`**

## What You'll See

### Home Page 🏠
Welcome screen with project overview and quick links to sections

### Data Analysis 📈
- **Interactive ticker selector** - Choose any of 8 global stock indices
- **Realized Volatility Chart** - 21-day rolling window volatility
- **Close Price Chart** - Historical stock prices
- Hover over charts for exact values

### Statistics 📊
- **Volatility Table** - Mean, Std Dev, Min, Max for all indices
- **Mean Volatility Bar Chart** - Compare across all markets
- **Standard Deviation Chart** - Which markets are most volatile

### Correlations 🔗
- **Spillover Heatmap** - Correlation matrix of all index pairs
- **Top 10 Effects Table** - Strongest market relationships
- Color coding shows strength (red = positive, blue = negative)

### Models 🤖
- **GCN + GAT Architecture** - Deep graph neural network explanation
- **Baseline MLP** - Traditional fully connected network
- **Performance Charts** - Compare models across forecast horizons
- **Metric Definitions** - RMSE, MAPE, MAE, MSE explained

### About 📋
- Full research methodology
- Data sources and preprocessing steps
- Technologies used
- Expected results

## Controls & Features

### Navigation
- Use **sidebar menu** (left side) to switch sections
- Click any section name to jump there instantly

### Interactive Charts
- **Hover** over graphs to see exact values
- **Zoom** by dragging to select area
- **Pan** by middle-mouse dragging
- **Download** chart as PNG (camera icon)
- **Reset** zoom (home icon)

### Tables
- Sort and read formatted data
- Easy copy-paste for external use

### Real-time Caching
- First load: ~2-3 seconds (generates synthetic data)
- Switching pages: Instant (from cache)
- No server lag

## Example Session

**Time: 2 minutes**

1. **Launch app** - `streamlit run project_streamlit.py`
2. **Home page** - Read overview (30 seconds)
3. **Data Analysis** - Select S&P 500 (^GSPC), view volatility chart (30 seconds)
4. **Statistics** - Compare all 8 indices side-by-side (30 seconds)
5. **Correlations** - See which markets affect each other (30 seconds)
6. **Models** - Understand GCN+GAT vs Baseline (optional)

## Stopping the App

**Press Ctrl+C** in the terminal where you ran `streamlit run project_streamlit.py`

Or simply close the browser window.

## File Details

| File | Purpose | Size |
|------|---------|------|
| `project_streamlit.py` | Main web app | ~400 lines |
| `install_streamlit.bat` | Streamlit installer | ~30 lines |
| `STREAMLIT_GUIDE.md` | Detailed guide | Complete reference |

## Troubleshooting

### Issue: Command not found
```powershell
# Make sure environment is activated
conda activate ml_fresh
```

### Issue: Port 8501 already in use
```powershell
# Use different port
streamlit run project_streamlit.py --server.port 8502
```

### Issue: Charts not showing
- Press Ctrl+R in browser
- Check browser console (F12)
- Restart app with Ctrl+C and re-run

## Next Steps

1. ✅ Run `install_streamlit.bat`
2. ✅ Execute `streamlit run project_streamlit.py`
3. ✅ Explore interactive dashboard
4. ✅ Read full STREAMLIT_GUIDE.md for advanced features

## Browser Compatibility

✅ Works on:
- Chrome (best experience)
- Firefox
- Safari
- Edge
- Mobile browsers (responsive)

---

**Need help?** See TROUBLESHOOTING.md for common issues  
**Want details?** Read STREAMLIT_GUIDE.md for comprehensive documentation  
**Questions?** Check the About section in the app
