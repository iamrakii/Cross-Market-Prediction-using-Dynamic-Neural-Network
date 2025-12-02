# Streamlit Web Interface Guide

## Overview
The project now includes a **Streamlit web interface** (`project_streamlit.py`) for interactive data exploration and visualization. This provides an intuitive dashboard in your browser instead of console output.

## Quick Start

### Step 1: Install Streamlit
If you're using the `ml_fresh` environment:

```powershell
conda activate ml_fresh
pip install streamlit plotly
```

Or if creating a fresh environment for Streamlit only:

```powershell
conda create -n streamlit-env python=3.9
conda activate streamlit-env
pip install streamlit plotly
```

### Step 2: Run the Application

```powershell
streamlit run project_streamlit.py
```

The web interface will automatically open in your default browser at `http://localhost:8501`

## Features

### 🏠 Home Page
- Project overview
- Quick links to other sections
- Key indices and features

### 📈 Data Analysis
- **Select any of 8 stock indices**
- **View realized volatility charts** (21-day rolling window)
- **Analyze historical close prices**
- Interactive hover tooltips for detailed data

### 📊 Statistics
- Comprehensive statistics table:
  - Mean volatility
  - Standard deviation
  - Min/Max values
  - Total data points
- Bar charts comparing indices:
  - Mean volatility comparison
  - Standard deviation comparison

### 🔗 Correlations
- **Spillover Correlation Matrix** (heatmap visualization)
- Shows volatility relationships between indices
- **Top 10 Strongest Spillover Effects** table
- Identifies which markets influence others most

### 🤖 Models
- **GCN + GAT Model** architecture details
  - Graph Convolutional Network
  - Graph Attention Network
  - Hyperparameter grid search options
  
- **Baseline MLP Model** specifications
  - 3-layer fully connected network
  - Quick benchmark comparison
  
- **Model Performance Comparison** chart
  - RMSE across forecast horizons (1, 5, 10, 22 days)
  - Visual comparison of architectures

### 📋 About
- Comprehensive project documentation
- Research methodology
- Data sources and preprocessing
- Technologies used

## Navigation

Use the **sidebar** to jump between sections:
1. Click the radio button for any section
2. Streamlit automatically reruns and displays the new content
3. All interactive elements (charts, tables) update instantly

## Interactive Features

### 📊 Charts
- **Hover over graphs** to see exact values
- **Zoom, pan, and download** using Plotly toolbar
- Responsive design adapts to screen size

### 📋 Tables
- Display formatted statistics
- Easy-to-read layout
- Copy-paste friendly

### 🎨 Visualizations
- Color-coded heatmaps for correlations
- Line graphs for time series
- Bar charts for comparisons

## Performance

The application uses **Streamlit's caching** (`@st.cache_data`) to:
- Generate data only once
- Reuse calculations across page refreshes
- Provide instant switching between pages

**First load**: ~2-3 seconds (generates synthetic data)
**Subsequent loads**: Instant (from cache)

## Troubleshooting

### Issue: "streamlit: command not found"
**Solution**: Make sure you activated the correct environment:
```powershell
conda activate streamlit-env
pip install streamlit
streamlit run project_streamlit.py
```

### Issue: "ModuleNotFoundError: No module named 'plotly'"
**Solution**: Install Plotly separately:
```powershell
pip install plotly
```

### Issue: Browser doesn't open automatically
**Solution**: Manually open the URL shown in terminal, typically:
```
http://localhost:8501
```

### Issue: Chart not displaying
**Solution**: Streamlit sometimes needs a full refresh:
- Press `Ctrl+R` in your browser
- Or check browser console for errors (F12)

## Running Alongside Console Version

You can run **both** the web interface and the original project simultaneously:

```powershell
# Terminal 1: Web interface
streamlit run project_streamlit.py

# Terminal 2: Original project (after environment setup)
python project.py
```

## Customization

### Change port (if 8501 is busy):
```powershell
streamlit run project_streamlit.py --server.port 8502
```

### Run in headless mode (no browser):
```powershell
streamlit run project_streamlit.py --headless
```

### Debug mode:
```powershell
streamlit run project_streamlit.py --logger.level=debug
```

## Browser Compatibility

- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ✅ Mobile browsers (responsive design)

## Next Steps

1. ✅ Install Streamlit and Plotly
2. ✅ Run `streamlit run project_streamlit.py`
3. ✅ Explore the interactive dashboard
4. ✅ Use the "Data Analysis" tab to explore different indices
5. ✅ Check "Correlations" to understand market relationships
6. ✅ Read "About" for full methodology

## File Structure

```
project_streamlit.py     # Main Streamlit application
├── Mock data generation
├── Volatility calculation
├── Page: Home (Overview)
├── Page: Data Analysis (Charts & exploration)
├── Page: Statistics (Metrics & comparisons)
├── Page: Correlations (Spillover matrix)
├── Page: Models (Architecture & performance)
└── Page: About (Full documentation)
```

## Notes

- **Data**: Uses same synthetic data as `project_mock.py` (deterministic with seed=42)
- **No external API calls**: Everything runs locally
- **Interactive**: Fully responsive to user selections
- **Fast**: Caching ensures smooth performance
- **Mobile-friendly**: Responsive design works on tablets/phones

## Getting Help

If you encounter issues:
1. Check TROUBLESHOOTING.md for common problems
2. Verify environment: `python -c "import streamlit; print(streamlit.__version__)"`
3. Check requirements: `pip list | findstr streamlit plotly`
4. Review Streamlit docs: https://docs.streamlit.io/

---

**Created**: December 2, 2025  
**Project**: Cross-Market Prediction using Dynamic Neural Network  
**Interface**: Streamlit Web Dashboard
