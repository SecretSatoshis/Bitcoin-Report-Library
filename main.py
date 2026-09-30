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

import pandas as pd

import cycles
import freshness
import metrics
import report_tables
import sources
from candle_data import write_candle_tables
from release_manifest import write_release_manifest
from data_definitions import (
    CAGR_COLUMNS,
    CHANGE_COLUMNS,
    CORRELATION_COLUMNS,
    FIAT_MONEY_SUPPLY,
    FUNDAMENTALS_TEMPLATE,
    GOLD_SILVER_SUPPLY,
    GOLD_SUPPLY_BREAKDOWN,
    MARKET_DATA_START_DATE,
    MOVING_AVERAGE_METRICS,
    PRICE_OUTLOOK_LEVELS,
    PRICE_OUTLOOK_YEAR,
    REPORT_DATE,
    STOCK_TICKERS,
    TICKERS,
    YOY_COLUMNS,
)
from data_validation import assert_ohlc_usable

# Daily candles start at genesis; BRK's pre-price candles are dropped on fetch.
DAILY_OHLC_START = "2009-01-03"


# --- Fetch and check sources ---

data = sources.get_data(TICKERS, MARKET_DATA_START_DATE)

freshness.warn_on_stale_market_data(data, REPORT_DATE)
# Correlations need each asset's real trading days, so capture them before the fill.
correlation_input = metrics.observed_market_values(data, CORRELATION_COLUMNS)
# Market data is filled within a bounded budget; on-chain data is never filled, so a
# missing or malformed BRK response fails the checks below.
data = freshness.forward_fill_market_data(data)
freshness.assert_onchain_freshness(data, REPORT_DATE)
freshness.assert_no_internal_onchain_gaps(data, REPORT_DATE)
freshness.assert_reference_data_fresh(REPORT_DATE)
freshness.assert_price_outlook_current(REPORT_DATE)
freshness.warn_on_stale_miner_efficiency(data, REPORT_DATE)

# Daily BRK candles are the only OHLC source; weekly and monthly candles are built from them.
daily_ohlc = sources.get_brk_ohlc(start=DAILY_OHLC_START)
assert_ohlc_usable(daily_ohlc, label="Daily BRK OHLC")


# --- Calculate metrics ---

data = metrics.calculate_custom_on_chain_metrics(data)
data = metrics.calculate_moving_averages(data, MOVING_AVERAGE_METRICS)
data = metrics.calculate_btc_price_to_surpass_fiat(data, FIAT_MONEY_SUPPLY)
data = metrics.calculate_metal_market_caps(data, GOLD_SILVER_SUPPLY)
data = metrics.calculate_btc_price_to_surpass_metal_categories(data, GOLD_SUPPLY_BREAKDOWN)
data = metrics.calculate_btc_price_for_stock_mkt_caps(data, STOCK_TICKERS)
data, model_parameters = metrics.calculate_network_model_metrics(data, REPORT_DATE)
data = metrics.electric_price_models(data)

# Period changes for CHANGE_COLUMNS, Bitcoin's YoY change and the 4-year CAGRs.
report_data = pd.concat(
    [data, metrics.calculate_all_changes(data[CHANGE_COLUMNS], YOY_COLUMNS)], axis=1
)
report_data = report_data.merge(
    metrics.calculate_rolling_cagr_for_all_columns(data[CAGR_COLUMNS], 4),
    left_index=True,
    right_index=True,
    how="left",
)
correlation_results = metrics.create_btc_correlation_data(
    REPORT_DATE, TICKERS, correlation_input
)

# Sources include the partial current UTC day. Cut it here so no export includes it.
report_data = report_data.loc[:REPORT_DATE]
report_data.index.name = "date"


# --- Build tables ---

prices = report_data["price_close"]
tables = {
    "summary_table.csv": report_tables.create_summary_table(report_data, REPORT_DATE),
    "summary_history.csv": report_tables.create_summary_history(
        report_data, REPORT_DATE, report_tables.HEADLINE_METRICS, comparison_days=30
    ),
    "fundamentals_table.csv": report_tables.create_fundamentals_table(
        report_data, FUNDAMENTALS_TEMPLATE, REPORT_DATE
    ),
    "performance_table.csv": report_tables.create_full_performance_table(
        report_data, REPORT_DATE, correlation_results
    ),
    "relative_value_comparison.csv": report_tables.create_asset_valuation_table(
        report_data, REPORT_DATE
    ),
    "roi_table.csv": report_tables.calculate_roi_table(report_data, REPORT_DATE),
    "mtd_return_comparison.csv": report_tables.create_period_returns_table(
        report_data, REPORT_DATE, "mtd"
    ),
    "ytd_return_comparison.csv": report_tables.create_period_returns_table(
        report_data, REPORT_DATE, "ytd"
    ),
    "report_ohlc_summary.csv": report_tables.create_report_ohlc_summary(daily_ohlc, REPORT_DATE),
    # Published with the levels so consumers can check they show the current forecast.
    "price_outlook.csv": PRICE_OUTLOOK_LEVELS.assign(outlook_year=PRICE_OUTLOOK_YEAR),
    "drawdown_data.csv": cycles.compute_drawdowns(report_data),
    "cycle_low_data.csv": cycles.compute_cycle_lows(report_data),
    "halving_data.csv": cycles.compute_halving_days(report_data),
}
# Published with their date index.
indexed_tables = {
    "monthly_heatmap_data.csv": report_tables.monthly_heatmap(report_data, REPORT_DATE),
    "mtd_price_paths.csv": report_tables.create_price_paths(prices, REPORT_DATE, "mtd"),
    "ytd_price_paths.csv": report_tables.create_price_paths(prices, REPORT_DATE, "ytd"),
    "onchain_price_models.csv": report_tables.create_onchain_price_models(
        report_data, REPORT_DATE
    ),
}


# --- Write the release ---
# Nothing is written until every table has built, so a failed run leaves csv/ untouched.

for filename, table in tables.items():
    table.to_csv(f"csv/{filename}", index=False)
for filename, table in indexed_tables.items():
    table.to_csv(f"csv/{filename}")
report_data.to_csv("csv/master_metrics_data.csv.gz", index=True, compression="gzip")
candle_files = write_candle_tables(daily_ohlc, report_data, REPORT_DATE)

# Consumers verify files against the manifest before using them.
write_release_manifest(
    "csv",
    REPORT_DATE,
    [*tables, *indexed_tables, "master_metrics_data.csv.gz", *candle_files],
    model_parameters,
)
