# QuantBot Dashboard

A local Streamlit app for exploring Qlib experiment runs stored in the `mlruns` directory.

## Setup

```bash
cd /Users/arjun/Documents/quantBot
pip install -r dashboard/requirements.txt
```

(If the `qlib` conda environment is active, use `pip` inside that environment.)

## Launch

```bash
cd /Users/arjun/Documents/quantBot
streamlit run dashboard/app.py
```

By default the app reads MLflow runs from `qlib/mlruns`. If your tracking directory differs, update the path from the sidebar.

## Features

- Browse MLflow experiments and runs.
- Visualize cumulative returns, daily excess returns, and trading costs.
- Inspect turnover (when recorded).
- View key risk metrics (annualized excess return, information ratio, max drawdown).

Extend the UI or backend to add backtest triggers, paper-trading controls, and live monitoring.
