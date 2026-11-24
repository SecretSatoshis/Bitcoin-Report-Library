"""
Bitcoin Report Library - Main Pipeline

Fetches every source, checks it is fresh and complete, calculates the metrics, builds the
report tables, and writes the release to csv/ with its manifest.
"""

# The module body fetches data and rewrites csv/, so an import (a test collector, an IDE
# indexer) must not run it.
if __name__ != "__main__":
    raise RuntimeError(
        "main.py is a script: importing it would fetch data and rewrite csv/. "
        "Import the pipeline modules instead, or run `python main.py`."
    )

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import agent21_examples
import annual_data
import cycles
import etf
import freshness
import metrics
import report_tables
import sources
from candle_data import CANDLE_FILES, build_candle_tables, write_candle_tables
from publication import format_float, round_for_publication
from release_manifest import write_release_manifest
from data_definitions import (
    CAGR_COLUMNS,
    CHANGE_COLUMNS,
    FIAT_MONEY_SUPPLY,
    FUNDAMENTALS_TEMPLATE,
    GOLD_SILVER_SUPPLY,
    GOLD_SUPPLY_BREAKDOWN,
    MARKET_DATA_START_DATE,
    MOVING_AVERAGE_METRICS,
    PRICE_OUTLOOK_LEVELS,
    PRICE_OUTLOOK_YEAR,
    STOCK_TICKERS,
    TICKERS,
    YOY_COLUMNS,
    default_report_date,
)
from data_validation import assert_ohlc_usable

# Daily candles start at genesis; BRK's pre-price candles are dropped on fetch.
DAILY_OHLC_START = "2009-01-03"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the daily Bitcoin Report Library release.")
    parser.add_argument(
        "--report-date",
        help="Completed UTC report date (YYYY-MM-DD); defaults to yesterday UTC.",
    )
    return parser.parse_args()


# Resolve the report date once so a run crossing UTC midnight uses one consistent cutoff.
_args = _parse_args()
report_date = (
    pd.to_datetime(_args.report_date).normalize()
    if _args.report_date
    else default_report_date()
)


# --- Fetch and check sources ---

with ThreadPoolExecutor(max_workers=4) as _executor:
    _data_future = _executor.submit(
        sources.get_data, TICKERS, MARKET_DATA_START_DATE, report_date=report_date
    )
    _ohlc_future = _executor.submit(sources.get_brk_ohlc, start=DAILY_OHLC_START)
    _annual_future = _executor.submit(annual_data.get_annual_reference_data, report_date)
    _etf_future = _executor.submit(etf.get_etf_release, report_date)

    data = _data_future.result()
    daily_ohlc = _ohlc_future.result()
    annual_reference = _annual_future.result()
    etf_files, etf_health = _etf_future.result()

freshness.warn_on_stale_market_data(data, report_date)
# Correlations need each asset's real trading days, so capture them before the fill.
correlation_input = metrics.observed_market_values(data, report_tables.CORRELATION_COLUMNS)
# Market data is filled within a bounded budget; on-chain data is never filled, so a
# missing or malformed BRK response fails the checks below.
data = freshness.forward_fill_market_data(data)
freshness.assert_onchain_freshness(data, report_date)
freshness.assert_no_internal_onchain_gaps(data, report_date)
freshness.assert_reference_data_fresh(report_date)
freshness.assert_price_outlook_current(report_date)
freshness.warn_on_stale_miner_efficiency(data, report_date)

# Daily BRK candles are the only OHLC source; weekly and monthly candles are built from them.
assert_ohlc_usable(daily_ohlc, label="Daily BRK OHLC")


# --- Calculate metrics ---

data = metrics.calculate_custom_on_chain_metrics(data)
data = metrics.calculate_moving_averages(data, MOVING_AVERAGE_METRICS)
data = metrics.calculate_btc_price_to_surpass_fiat(data, FIAT_MONEY_SUPPLY)
data = metrics.calculate_metal_market_caps(data, GOLD_SILVER_SUPPLY)
data = metrics.calculate_btc_price_to_surpass_metal_categories(data, GOLD_SUPPLY_BREAKDOWN)
data = metrics.calculate_btc_price_to_surpass_stocks(data, STOCK_TICKERS)
data, model_parameters = metrics.calculate_network_model_metrics(data, report_date)
data = metrics.electric_price_models(data)

# Period changes for CHANGE_COLUMNS, Bitcoin's YoY change and the 4-year CAGRs.
report_data = pd.concat(
    [data, metrics.calculate_all_changes(data[CHANGE_COLUMNS], YOY_COLUMNS),
     metrics.calculate_performance_changes(data[report_tables.CORRELATION_COLUMNS])], axis=1
)
report_data = report_data.merge(
    metrics.calculate_rolling_cagr_for_all_columns(data[CAGR_COLUMNS], 4),
    left_index=True,
    right_index=True,
    how="left",
)
# One set of 30/90/365-day matrices feeds both the matrix file and the performance
# table's 90-day Bitcoin correlation column.
correlation_matrices = report_tables.create_correlation_matrices(correlation_input, report_date)

# Sources include the partial current UTC day. Cut it here so no export includes it.
report_data = report_data.loc[:report_date]
# Round once, before any table is built, so every file derives from the published values.
report_data = round_for_publication(report_data)


# --- Build tables ---

prices = report_data["price_close"]
tables = {
    "summary_table.csv": report_tables.create_summary_table(report_data, report_date),
    "summary_history.csv": report_tables.create_summary_history(
        report_data, report_date, report_tables.HEADLINE_METRICS, comparison_days=30
    ),
    "fundamentals_table.csv": report_tables.create_fundamentals_table(
        report_data, FUNDAMENTALS_TEMPLATE, report_date
    ),
    "performance_table.csv": report_tables.create_full_performance_table(
        report_data, report_date, correlation_matrices[90].loc["price_close"]
    ),
    "correlation_matrix.csv": report_tables.create_correlation_matrix_table(
        correlation_matrices, report_date
    ),
    "relative_value_comparison.csv": report_tables.create_asset_valuation_table(
        report_data, report_date
    ),
    report_tables.RELATIVE_RANK_FILE: report_tables.create_relative_value_rank_changes(
        report_data, report_date
    ),
    "roi_table.csv": report_tables.calculate_roi_table(report_data, report_date),
    "mtd_return_comparison.csv": report_tables.create_period_returns_table(
        report_data, report_date, "mtd"
    ),
    "ytd_return_comparison.csv": report_tables.create_period_returns_table(
        report_data, report_date, "ytd"
    ),
    "report_ohlc_summary.csv": report_tables.create_report_ohlc_summary(daily_ohlc, report_date),
    # Published with the levels so consumers can check they show the current forecast.
    "price_outlook.csv": PRICE_OUTLOOK_LEVELS.assign(outlook_year=PRICE_OUTLOOK_YEAR),
    "drawdown_data.csv": cycles.compute_drawdowns(report_data),
    "cycle_low_data.csv": cycles.compute_cycle_lows(report_data),
    "halving_data.csv": cycles.compute_halving_days(report_data),
    annual_data.ANNUAL_REFERENCE_FILE: annual_reference,
    **etf_files,
}
# Published with their date index.
indexed_tables = {
    "monthly_heatmap_data.csv": report_tables.monthly_heatmap(report_data, report_date),
    "mtd_price_paths.csv": report_tables.create_price_paths(prices, report_date, "mtd"),
    "ytd_price_paths.csv": report_tables.create_price_paths(prices, report_date, "ytd"),
    "onchain_price_models.csv": report_tables.create_onchain_price_models(
        report_data, report_date
    ),
}
# Daily, weekly and monthly candles, checked against the master prices.
candle_tables = build_candle_tables(daily_ohlc, report_data, report_date)
# Figures for the example answers on the Agent 21 landing page.
examples = agent21_examples.build_agent21_examples(
    report_data, candle_tables[CANDLE_FILES[0]], PRICE_OUTLOOK_LEVELS, PRICE_OUTLOOK_YEAR, report_date
)


# --- Write the release ---
# Nothing is written until every table has built, so a failed run leaves csv/ untouched.

for filename, table in tables.items():
    table.to_csv(f"csv/{filename}", index=False, float_format=format_float)
for filename, table in indexed_tables.items():
    table.to_csv(f"csv/{filename}", float_format=format_float)
report_data.to_csv(
    "csv/master_metrics_data.csv.gz", compression="gzip", float_format=format_float
)
write_candle_tables(candle_tables)
# Beside the release, not in its manifest: a view of released values for one page and ETF health.
agent21_examples.write_agent21_examples(examples)
etf.write_etf_health(etf_health)

# Consumers verify files against the manifest before using them.
write_release_manifest(
    "csv",
    report_date,
    [*tables, *indexed_tables, "master_metrics_data.csv.gz", *candle_tables],
    model_parameters,
)
