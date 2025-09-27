"""Collect and clean Finviz features for a list of tickers.

Example usage:
    python scripts/collect_finviz_features.py --tickers AAPL,MSFT,TSLA

This script snapshots cleaned numeric features into CSV files under
`data_cache/finviz/` so they can be joined to Qlib pipelines later.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from finvizfinance.quote import finvizfinance

# Helper parsers -------------------------------------------------------------


def parse_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "NaN", "nan"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_percent(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "NaN", "nan"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_high_low(value: str | None) -> Tuple[float | None, float | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if text in {"", "-", "NaN", "nan"}:
        return None, None
    parts = text.split()
    price = parse_float(parts[0]) if parts else None
    pct = parse_percent(parts[1]) if len(parts) > 1 else None
    return price, pct


# Field mapping --------------------------------------------------------------

COMMON_FIELD_MAP: Dict[str, Tuple[str, callable]] = {
    "Price": ("price", parse_float),
    "Change": ("change_pct", parse_percent),
    "P/E": ("pe", parse_float),
    "Forward P/E": ("forward_pe", parse_float),
    "PEG": ("peg", parse_float),
    "EPS this Y": ("eps_growth_this_year_pct", parse_percent),
    "EPS next Y": ("eps_next_year", parse_float),
    "EPS next 5Y": ("eps_next_5y_pct", parse_percent),
    "Sales Y/Y TTM": ("sales_yoy_ttm_pct", parse_percent),
    "EPS Q/Q": ("eps_qoq_pct", parse_percent),
    "Sales Q/Q": ("sales_qoq_pct", parse_percent),
    "ROE": ("roe_pct", parse_percent),
    "ROIC": ("roic_pct", parse_percent),
    "Gross Margin": ("gross_margin_pct", parse_percent),
    "Oper. Margin": ("oper_margin_pct", parse_percent),
    "Profit Margin": ("profit_margin_pct", parse_percent),
    "Debt/Eq": ("debt_to_equity", parse_float),
    "LT Debt/Eq": ("lt_debt_to_equity", parse_float),
    "Current Ratio": ("current_ratio", parse_float),
    "Quick Ratio": ("quick_ratio", parse_float),
    "Short Float": ("short_float_pct", parse_percent),
    "Short Ratio": ("short_ratio", parse_float),
    "Insider Own": ("insider_own_pct", parse_percent),
    "Insider Trans": ("insider_trans_pct", parse_percent),
    "Inst Own": ("institutional_own_pct", parse_percent),
    "Inst Trans": ("institutional_trans_pct", parse_percent),
    "Perf Week": ("perf_week_pct", parse_percent),
    "Perf Month": ("perf_month_pct", parse_percent),
    "SMA20": ("sma20_pct", parse_percent),
    "SMA50": ("sma50_pct", parse_percent),
    "SMA200": ("sma200_pct", parse_percent),
    "RSI (14)": ("rsi_14", parse_float),
}

HIGH_LOW_FIELDS = {
    "52W High": ("52w_high_price", "52w_high_distance_pct"),
    "52W Low": ("52w_low_price", "52w_low_distance_pct"),
}


def collect_quote_features(tickers: Iterable[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        stock = finvizfinance(ticker)
        fundament = stock.ticker_fundament()
        row: Dict[str, float | str | None] = {
            "ticker": ticker.upper(),
            "sector": fundament.get("Sector"),
            "industry": fundament.get("Industry"),
            "country": fundament.get("Country"),
        }
        for field, (col, parser) in COMMON_FIELD_MAP.items():
            value = fundament.get(field)
            row[col] = parser(value)
        for field, (price_col, distance_col) in HIGH_LOW_FIELDS.items():
            price, pct = parse_high_low(fundament.get(field))
            row[price_col] = price
            row[distance_col] = pct
        rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(1, "snapshot_utc", datetime.now(timezone.utc).isoformat())
    return df


# CLI -----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect cleaned Finviz features")
    parser.add_argument(
        "--tickers",
        default="AAPL,MSFT,TSLA",
        help="Comma-separated tickers to fetch (default: AAPL,MSFT,TSLA)",
    )
    parser.add_argument(
        "--output-dir",
        default="data_cache/finviz",
        help="Directory to store snapshots (default: data_cache/finviz)",
    )
    parser.add_argument(
        "--dump-raw",
        action="store_true",
        help="Persist the cleaned records to a JSON file alongside the CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided")

    df = collect_quote_features(tickers)

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    csv_path = output_base / f"quotes_features_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    if args.dump_raw:
        json_path = output_base / f"quotes_features_{timestamp}.json"
        json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2))

    print(f"Saved {len(df)} rows to {csv_path}")
    if args.dump_raw:
        print(f"Raw JSON written to {json_path}")


if __name__ == "__main__":
    main()
