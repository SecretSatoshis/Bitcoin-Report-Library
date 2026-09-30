"""
Bitcoin Report Library - Main Pipeline

Fetches every source, checks it is fresh and complete, calculates the metrics, builds the
report tables, and writes the release to csv/ with its manifest.
"""

# This module is the pipeline entry point: its body runs the whole fetch-and-export at
# import time. Refuse to be imported so a test collector, IDE indexer or stray
# `import main` cannot trigger network fetches and overwrite csv/ as a side effect.
if __name__ != "__main__":
    raise RuntimeError(
        "main.py is an executable pipeline, not an importable module — running it as a "
        "side effect of an import would fetch upstream data and rewrite csv/. Import "
        "the pipeline modules instead, or run `python main.py`."
    )

import pandas as pd

import cycles
import freshness
import metrics
import report_tables
import sources
from candle_data import weekly_ohlc, write_candle_tables
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

# Daily candles start at genesis; the published weekly OHLC history starts in 2017.
DAILY_OHLC_START = "2009-01-03"
WEEKLY_OHLC_START = "2017-01-01"


# --- Fetch and check sources ---

data = sources.get_data(TICKERS, MARKET_DATA_START_DATE)

## Market data is forward-filled within a bounded budget; on-chain series never are, so a
## missing or malformed BRK response fails below instead of repeating yesterday's values.
freshness.warn_on_stale_market_data(data, REPORT_DATE)
## Correlations need each asset's real trading days, so capture them before the fill
## turns weekends and holidays into carried-forward closes.
correlation_input = metrics.observed_market_values(data, CORRELATION_COLUMNS)
data = freshness.forward_fill_market_data(data)
freshness.assert_onchain_freshness(data, REPORT_DATE)
freshness.assert_no_internal_onchain_gaps(data, REPORT_DATE)
freshness.assert_reference_data_fresh(REPORT_DATE)
freshness.assert_price_outlook_current(REPORT_DATE)
freshness.warn_on_stale_miner_efficiency(data, REPORT_DATE)

## Daily BRK candles are the single OHLC source; weekly candles are aggregated from them
## so the open week is cut off at the report date like every other export.
daily_ohlc = sources.get_brk_ohlc(start=DAILY_OHLC_START)
assert_ohlc_usable(daily_ohlc, label="Daily BRK OHLC")
weekly_candles = weekly_ohlc(daily_ohlc, REPORT_DATE, start=WEEKLY_OHLC_START)


# --- Calculate metrics ---

data = metrics.calculate_custom_on_chain_metrics(data)
data = metrics.calculate_moving_averages(data, MOVING_AVERAGE_METRICS)
data = metrics.calculate_btc_price_to_surpass_fiat(data, FIAT_MONEY_SUPPLY)
data = metrics.calculate_metal_market_caps(data, GOLD_SILVER_SUPPLY)
data = metrics.calculate_btc_price_to_surpass_metal_categories(data, GOLD_SUPPLY_BREAKDOWN)
data = metrics.calculate_btc_price_for_stock_mkt_caps(data, STOCK_TICKERS)
data = metrics.calculate_network_model_metrics(data, REPORT_DATE)
data = metrics.electric_price_models(data)

## 7-day, 90-day, MTD and YTD changes for the price columns the reports read, Bitcoin's
## YoY change, and the 4-year CAGRs Chart Library's CAGR charts read.
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

## Upstream fetches include the partial, in-progress UTC day, whose 24h aggregates are a
## fraction of a real day. Truncate once, here, so no table or export can include it.
report_data = report_data.loc[:REPORT_DATE]


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
    # The outlook year travels with the levels so every consumer can verify it is looking
    # at the current forecast rather than trusting its own hardcoded copy.
    "price_outlook.csv": PRICE_OUTLOOK_LEVELS.assign(outlook_year=PRICE_OUTLOOK_YEAR),
    "drawdown_data.csv": cycles.compute_drawdowns(report_data),
    "cycle_low_data.csv": cycles.compute_cycle_lows(report_data),
    "halving_data.csv": cycles.compute_halving_days(report_data),
}
## These tables are published with their date index.
indexed_tables = {
    "monthly_heatmap_data.csv": report_tables.monthly_heatmap(report_data, REPORT_DATE),
    "mtd_returns_history.csv": report_tables.create_indexed_returns_history(
        prices, REPORT_DATE, "mtd"
    ),
    "ytd_returns_history.csv": report_tables.create_indexed_returns_history(
        prices, REPORT_DATE, "ytd"
    ),
    "onchain_price_models.csv": report_tables.create_onchain_price_models(
        report_data, REPORT_DATE
    ),
    "ohlc_data.csv": report_tables.weekly_ohlc_table(weekly_candles),
}


# --- Write the release ---
## Everything above has succeeded before any file is written, so a failed run never
## leaves a partly updated release in csv/.

for filename, table in tables.items():
    table.to_csv(f"csv/{filename}", index=False)
for filename, table in indexed_tables.items():
    table.to_csv(f"csv/{filename}")
report_data.to_csv("csv/master_metrics_data.csv.gz", index=True, compression="gzip")
candle_files = write_candle_tables(daily_ohlc, report_data, REPORT_DATE)

# Downstream consumers verify this complete release before rendering.
# The workflow validates the release before publishing its CSV directory.
write_release_manifest(
    "csv",
    REPORT_DATE,
    [*tables, *indexed_tables, "master_metrics_data.csv.gz", *candle_files],
)
