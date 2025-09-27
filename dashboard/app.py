import os
os.environ["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYQLIB"] = os.environ.get(
    "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYQLIB", "0.0"
)
os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0")
import importlib
import pickle
from pathlib import Path
from typing import Dict, Optional

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MLRUNS_DIR = BASE_DIR / "qlib" / "mlruns"

_QLIB_READY: Optional[bool] = None


def ensure_qlib_ready() -> bool:
    global _QLIB_READY
    if _QLIB_READY:
        return True
    try:
        qlib_module = importlib.import_module("qlib")
        if not hasattr(qlib_module, "__version__"):
            qlib_module.__version__ = "0.0"
        _QLIB_READY = True
        return True
    except Exception as err:
        st.warning(f"Unable to import qlib for artifact deserialization: {err}")
        _QLIB_READY = False
        return False


@st.cache_resource(show_spinner=False)
def get_client(tracking_dir: Path) -> mlflow.tracking.MlflowClient:
    tracking_uri = tracking_dir.resolve().as_uri()
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


@st.cache_data(show_spinner=False)
def list_experiments(_client: mlflow.tracking.MlflowClient) -> Dict[str, str]:
    experiments = _client.search_experiments()
    return {exp.name: exp.experiment_id for exp in experiments}


@st.cache_data(show_spinner=False)
def list_runs(_client: mlflow.tracking.MlflowClient, experiment_id: str) -> pd.DataFrame:
    runs = _client.search_runs(experiment_ids=[experiment_id], order_by=["attributes.start_time DESC"])
    if not runs:
        return pd.DataFrame()
    records = []
    for run in runs:
        info = run.info
        data = {
            "run_id": info.run_id,
            "start_time": pd.to_datetime(info.start_time, unit="ms"),
            "end_time": pd.to_datetime(info.end_time, unit="ms") if info.end_time else None,
            "status": info.status,
        }
        data.update(run.data.params)
        for key, value in run.data.metrics.items():
            data[key] = value
        records.append(data)
    return pd.DataFrame(records)


def get_run_artifact_path(tracking_dir: Path, experiment_id: str, run_id: str, relative_path: str) -> Path:
    run_dir = tracking_dir / str(experiment_id) / run_id / "artifacts"
    return run_dir / relative_path


def load_pickle(path: Path, *, require_qlib: bool = False) -> Optional[object]:
    if not path.exists():
        return None
    if require_qlib and not ensure_qlib_ready():
        return None
    try:
        with path.open("rb") as fp:
            return pickle.load(fp)
    except Exception as err:
        st.warning(f"Failed to load {path.name}: {err}")
        return None


def render_cumulative_chart(report_df: pd.DataFrame):
    cumulative = report_df[["return", "bench"]].cumsum()
    cumulative.columns = ["Strategy", "Benchmark"]
    fig = px.line(
        cumulative.reset_index(),
        x="datetime",
        y=["Strategy", "Benchmark"],
        title="Cumulative Return",
        labels={"value": "Cumulative Return", "datetime": "Date", "variable": "Series"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_daily_return_chart(report_df: pd.DataFrame):
    fig = px.line(
        report_df.reset_index(),
        x="datetime",
        y="return",
        title="Daily Strategy Excess Return",
        labels={"return": "Excess Return", "datetime": "Date"},
    )
    st.plotly_chart(fig, use_container_width=True)

    hist = px.histogram(
        report_df,
        x="return",
        nbins=40,
        title="Distribution of Daily Excess Returns",
        labels={"return": "Excess Return"},
    )
    st.plotly_chart(hist, use_container_width=True)


def render_cost_chart(report_df: pd.DataFrame):
    cost_df = report_df[["cost", "total_cost"]].cumsum()
    cost_df.columns = ["Cost", "Total Cost"]
    fig = px.line(
        cost_df.reset_index(),
        x="datetime",
        y=["Cost", "Total Cost"],
        title="Cumulative Trading Costs",
        labels={"value": "Cost", "datetime": "Date", "variable": "Series"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_turnover_chart(indicator):
    turnover_series = None
    trade_indicator = getattr(indicator, "trade_indicator", None)
    if trade_indicator is not None:
        turnover_series = trade_indicator.get("turnover")
    if turnover_series is None:
        st.info("No turnover data available for this run.")
        return
    turnover_df = turnover_series.reset_index()
    turnover_df.columns = ["datetime", "turnover"]
    fig = px.line(
        turnover_df,
        x="datetime",
        y="turnover",
        title="Turnover Ratio",
        labels={"turnover": "Turnover", "datetime": "Date"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_run_details(tracking_dir: Path, experiment_id: str, run_id: str):
    report_path = get_run_artifact_path(tracking_dir, experiment_id, run_id, "portfolio_analysis/report_normal_1day.pkl")
    summary_path = get_run_artifact_path(tracking_dir, experiment_id, run_id, "portfolio_analysis/port_analysis_1day.pkl")
    indicator_path = get_run_artifact_path(tracking_dir, experiment_id, run_id, "portfolio_analysis/indicators_normal_1day_obj.pkl")

    report_df = load_pickle(report_path)
    summary_df = load_pickle(summary_path)
    indicator_obj = load_pickle(indicator_path, require_qlib=True)

    if report_df is None:
        st.warning("report_normal_1day.pkl not found for this run.")
        return

    st.markdown(f"### Run {run_id}")
    cols = st.columns(3)
    strategy_cum = report_df["return"].cumsum().iloc[-1]
    bench_cum = report_df["bench"].cumsum().iloc[-1]
    cols[0].metric("Final Strategy Cumulative Return", f"{strategy_cum:.2%}")
    cols[1].metric("Final Benchmark Cumulative Return", f"{bench_cum:.2%}")
    if summary_df is not None:
        try:
            ann = summary_df.loc[("excess_return_with_cost", "annualized_return"), "risk"]
            ir = summary_df.loc[("excess_return_with_cost", "information_ratio"), "risk"]
            mdd = summary_df.loc[("excess_return_with_cost", "max_drawdown"), "risk"]
            cols[2].metric("Information Ratio", f"{ir:.2f}")
            st.table(
                pd.DataFrame(
                    {
                        "Metric": ["Annualized Excess Return", "Information Ratio", "Max Drawdown"],
                        "Value": [f"{ann:.2%}", f"{ir:.2f}", f"{mdd:.2%}"],
                    }
                )
            )
        except Exception:
            pass

    render_cumulative_chart(report_df)
    render_daily_return_chart(report_df)
    render_cost_chart(report_df)

    if indicator_obj is not None:
        st.subheader("Turnover")
        render_turnover_chart(indicator_obj)


def main():
    st.set_page_config(page_title="QuantBot Dashboard", layout="wide")
    st.title("QuantBot Research Dashboard")

    with st.sidebar:
        st.header("Configuration")
        tracking_dir_input = st.text_input(
            "MLflow tracking directory",
            value=str(DEFAULT_MLRUNS_DIR),
            help="Path to the mlruns folder."
        )
        tracking_dir = Path(tracking_dir_input).expanduser()

    if not tracking_dir.exists():
        st.error("Tracking directory does not exist. Update the path in the sidebar.")
        return

    client = get_client(tracking_dir)
    experiments = list_experiments(client)
    if not experiments:
        st.info("No MLflow experiments found in this tracking directory.")
        return

    experiment_name = st.sidebar.selectbox("Experiment", sorted(experiments.keys()))
    experiment_id = experiments[experiment_name]

    runs_df = list_runs(client, experiment_id)
    if runs_df.empty:
        st.info("No runs found for this experiment.")
        return

    st.sidebar.subheader("Runs")
    run_options = runs_df["run_id"].tolist()
    run_id = st.sidebar.selectbox("Run ID", run_options)

    selected_run = runs_df[runs_df["run_id"] == run_id].iloc[0]
    st.sidebar.markdown("**Run details**")
    st.sidebar.dataframe(selected_run[["start_time", "end_time", "status"]].to_frame().T)

    render_run_details(tracking_dir, experiment_id, run_id)


if __name__ == "__main__":
    main()
