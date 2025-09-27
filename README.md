# QuantBot

A research playground for building quantitative trading strategies on top of [Microsoft Qlib](https://github.com/microsoft/qlib). The repo bundles:

- Qlib source (as a submodule-style directory) for strategy research and customization.
- A Streamlit dashboard (`dashboard/app.py`) to explore MLflow backtests visually.
- Setup scripts and instructions to recreate the environment, download sample data, and validate the pipeline.

---

## 1. Prerequisites

- macOS or Linux with Python 3.8–3.11 (tested on macOS 14 + Python 3.11)
- [Conda](https://github.com/conda-forge/miniforge) **or** Python `venv`
- Git + [GitHub CLI](https://cli.github.com/) (only needed if you plan to push changes)
- Xcode Command Line Tools on macOS (`xcode-select --install`) — provides compilers for LightGBM and other native deps

Optional but recommended:

- [Homebrew](https://brew.sh/) to install `libomp` (needed by LightGBM)
- Node.js if you plan to extend the dashboard with custom tooling

---

## 2. Repository Structure

```
QuantBot/
├── README.md                   # You are here
├── dashboard/                  # Streamlit dashboard to inspect runs
│   ├── app.py
│   └── requirements.txt
├── qlib/                       # Vendored Qlib source (as a git link checkout)
│   ├── ...                     # upstream Microsoft Qlib project
└── .gitignore                  # Ignores large data / artifacts
```

Generated data lives **outside** the repository: `qlib_data/` for factor data, `qlib/mlruns/` for experiment logs, etc. These paths are git-ignored to keep the repo lightweight.

---

## 3. Environment Setup

Clone the repo and create an isolated environment:

```bash
# 1. Clone
git clone https://github.com/arjunkramadhati/QuantBot.git
cd QuantBot

# 2. Create Python environment (Conda example)
conda create -n qlib python=3.11
conda activate qlib

# Optional: avoid auto-activating base
conda config --set auto_activate_base false

# 3. Upgrade build tooling
pip install --upgrade pip setuptools wheel
```

Install project dependencies:

```bash
# Qlib core (from PyPI)
pip install qlib

# Dashboard & plotting utilities
pip install -r dashboard/requirements.txt

# Optional extras
pip install jupyterlab
```

### macOS-only: LightGBM / OpenMP

```bash
brew install libomp
# Persist library hints (if not already present)
echo 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' >> ~/.zshrc
echo 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' >> ~/.zshrc
echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
source ~/.zshrc
```

---

## 4. Configure Data Directories

Download Qlib’s pre-packaged Yahoo Finance bundles (choose the region(s) you need):

```bash
# Set a data home (adjust path if desired)
export QLIB_DATA=~/Documents/quantBot/qlib_data
mkdir -p "$QLIB_DATA"
# Persist the environment variable
if ! grep -q "QLIB_DATA" ~/.zshrc; then
  echo "export QLIB_DATA=$QLIB_DATA" >> ~/.zshrc
fi

# CN daily bundle
the python qlib/scripts/get_data.py qlib_data --name qlib_data --region cn --interval 1d \
  --target_dir "$QLIB_DATA/cn_data" --delete_old False

# US daily bundle (optional)
python qlib/scripts/get_data.py qlib_data --name qlib_data --region us --interval 1d \
  --target_dir "$QLIB_DATA/us_data" --delete_old False
```

> ⚠️ These datasets come from Yahoo Finance and are best-effort clean — expect occasional data quality warnings.

---

## 5. Quick Functional Checks

Once the data is in place, run the canonical Qlib workflow (CN market by default):

```bash
cd qlib
python examples/workflow_by_code.py
```

You should see logs similar to:

- Successful `qlib.init(...)`
- LightGBM training with early stopping
- IC / ICIR metrics
- Backtest output (annualized return, IR, drawdown, etc.)

Artifacts land under `qlib/mlruns/<experiment>/<run_id>/artifacts/` (git-ignored).

### Working with US data

Duplicate `examples/workflow_by_code.py` or create a custom script, changing the config to:

```python
qlib.init(provider_uri="/path/to/your/us_data", region="us")
```

and update the benchmark/task settings accordingly.

---

## 6. Streamlit Dashboard

Launch the research dashboard to browse MLflow runs visually:

```bash
cd /path/to/QuantBot
streamlit run dashboard/app.py
```

Features:

- Experiment selector (reads from `qlib/mlruns` by default)
- Cumulative return, daily excess return histograms, and trading-cost charts
- Turnover visualization (if recorded)
- Summary metrics (annualized excess return, information ratio, max drawdown)

Use a browser to visit [http://localhost:8501](http://localhost:8501). Update the sidebar path if your MLflow directory lives elsewhere.

---

## 7. Testing & Validation

Recommended smoke tests after setup:

1. **Environment check**
   ```bash
   python -c "import qlib, torch, lightgbm; print('deps ok')"
   ```

2. **Data availability**
   ```bash
   ls "$QLIB_DATA"    # ensure cn_data/us_data directories exist
   ```

3. **Backtest reproduction**
   ```bash
   cd qlib
   python examples/workflow_by_code.py
   ```

4. **Dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

If these complete without errors, the project is ready for strategy research or further customization.

---

## 8. Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch manually: `pip install torch torchvision torchaudio` |
| `OSError: lib_lightgbm.dylib ... libomp.dylib not loaded` | Install `libomp` via Homebrew and export the environment variables listed above |
| Streamlit cache errors | The dashboard already disables hashing for MLflow clients; restart the app after edits |
| Artifacts missing in dashboard | Ensure you ran a backtest and that `qlib/mlruns/.../portfolio_analysis/*.pkl` exists |
| Need to change experiment path | Use the sidebar text input to point to a different `mlruns` directory |

---

## 9. Contributing / Next Steps

- Add strategy configs under `qlib/examples/benchmarks/` or create new scripts under `QuantBot/`.
- Extend the dashboard with strategy controls (start/stop paper trading, parameter tweaks).
- Integrate broker APIs to move from research (backtests) to paper/live trading.

Feel free to open issues or pull requests in the GitHub repo for enhancements.

---

## 10. License

The surrounding glue code is MIT-licensed. Qlib itself is MIT-licensed by Microsoft; see `qlib/LICENSE` for details.
