"""
Bitcoin Report Library - Main Pipeline

This script orchestrates the complete data pipeline for Bitcoin market and on-chain analytics.
It fetches data from multiple sources, calculates metrics, generates report tables, and exports
CSV files for downstream analysis.
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

# Import Packages
import pandas as pd
import warnings
import sys


# Ignore FutureWarning & Cache
warnings.simplefilter(action="ignore", category=FutureWarning)
sys.dont_write_bytecode = True

# Import Files
import cycles
import data_validation
import freshness
import metrics
import sources

from data_definitions import (
    TICKERS,
    STOCK_TICKERS,
    REPORT_DATE,
    MARKET_DATA_START_DATE,
    MOVING_AVERAGE_METRICS,
    CAGR_COLUMNS,
    FIAT_MONEY_SUPPLY,
    GOLD_SILVER_SUPPLY,
    GOLD_SUPPLY_BREAKDOWN,
    CHANGE_COLUMNS,
    YOY_COLUMNS,
    CORRELATION_COLUMNS,
    FUNDAMENTALS_TEMPLATE,
    PRICE_OUTLOOK_LEVELS,
    PRICE_OUTLOOK_YEAR,
)

# Fetch the data
data = sources.get_data(TICKERS, MARKET_DATA_START_DATE)

## Forward fill market data only.
## Equities/ETFs/FX print on trading days and miner efficiency prints monthly, so both
## need carrying forward onto Bitcoin's 365-day index. On-chain series print daily, and
## filling those would turn a missing or malformed BRK response into a silent repeat of
## yesterday's values, so they are validated instead.
freshness.warn_on_stale_market_data(data, REPORT_DATE)
## Correlations need each asset's real trading days, so capture them before the fill
## turns weekends and holidays into carried-forward closes.
correlation_df = metrics.observed_market_values(data, CORRELATION_COLUMNS)
data = freshness.forward_fill_market_data(data)
freshness.assert_onchain_freshness(data, REPORT_DATE)
freshness.assert_no_internal_onchain_gaps(data, REPORT_DATE)
freshness.assert_reference_data_fresh(REPORT_DATE)
freshness.assert_price_outlook_current(REPORT_DATE)
freshness.warn_on_stale_miner_efficiency(data, REPORT_DATE)

## BRK OHLC data — daily candles are the single source; weekly candles are aggregated
## from them so the open week is cut off at the report date like every other export.
daily_ohlc_start = "2009-01-03"
daily_ohlc_data = sources.get_brk_ohlc(start=daily_ohlc_start)
daily_ohlc_data.index = pd.to_datetime(daily_ohlc_data.index)
if daily_ohlc_data.index.tz is not None:
    daily_ohlc_data.index = daily_ohlc_data.index.tz_convert(None)
data_validation.assert_ohlc_usable(daily_ohlc_data, label="Daily BRK OHLC")

from candle_data import weekly_ohlc
WEEKLY_OHLC_START = "2017-01-01"
ohlc_data = weekly_ohlc(daily_ohlc_data, REPORT_DATE, start=WEEKLY_OHLC_START)
data_validation.assert_ohlc_usable(ohlc_data, label="Weekly OHLC")

# Calculate Custom Metrics
data = metrics.calculate_custom_on_chain_metrics(data)
data = metrics.calculate_moving_averages(data, MOVING_AVERAGE_METRICS)

## Fiat / Gold Calculations
data = metrics.calculate_btc_price_to_surpass_fiat(data, FIAT_MONEY_SUPPLY)
data = metrics.calculate_metal_market_caps(data, GOLD_SILVER_SUPPLY)
data = metrics.calculate_btc_price_to_surpass_metal_categories(data, GOLD_SUPPLY_BREAKDOWN)

## Calculate On-chain Models
data = metrics.calculate_btc_price_for_stock_mkt_caps(data, STOCK_TICKERS)
data = metrics.calculate_network_model_metrics(data, REPORT_DATE)
data = metrics.electric_price_models(data)

# Create Datasets

## Create Report Data - 7-day, 90-day, MTD and YTD changes for the price columns
## the reports read, plus Bitcoin's YoY change
changes = metrics.calculate_all_changes(data[CHANGE_COLUMNS], YOY_COLUMNS)
report_data = pd.concat([data, changes], axis=1)

## 4-year CAGR for the price columns Chart Library's CAGR charts read
cagr_results = metrics.calculate_rolling_cagr_for_all_columns(data[CAGR_COLUMNS], 4)
report_data = report_data.merge(cagr_results, left_index=True, right_index=True, how="left")

## Create Bitcoin Correlation Data (correlation_df was captured before the fill)
correlation_results = metrics.create_btc_correlation_data(
    REPORT_DATE, TICKERS, correlation_df
)

# Table Creation

# Import Report Functions
import report_tables

# Create ROI Table
roi_table = report_tables.calculate_roi_table(data, REPORT_DATE)

# Create Fundamentals Table
fundamentals_table = report_tables.create_fundamentals_table(
    report_data, FUNDAMENTALS_TEMPLATE, REPORT_DATE
)

# Create OHLC CSV
report_tables.calculate_ohlc(ohlc_data)
report_tables.create_report_ohlc_summary(daily_ohlc_data, REPORT_DATE)

# Create MTD Return Comparison Table
mtd_return_comp = report_tables.create_monthly_returns_table(report_data, REPORT_DATE)

# Create YTD Return Comparison Table
ytd_return_comp = report_tables.create_yearly_returns_table(report_data, REPORT_DATE)

# Create Relative Valuation Table
rv_table = report_tables.create_asset_valuation_table(report_data, REPORT_DATE)

# Create the summary table
summary_table = report_tables.create_summary_table(
    report_data, REPORT_DATE
)
# Create the performance table
performance_table = (
    report_tables.create_full_performance_table(
        report_data,
        REPORT_DATE,
        correlation_results,
    )
)


# Create Heat Map CSV
report_tables.monthly_heatmap(report_data, REPORT_DATE)


# CSV Exports

## Every exported frame is truncated to the report date. Upstream fetches return a
## partial, in-progress UTC day whose 24h aggregates (hash rate, miner revenue, tx
## count, supply issuance) are a fraction of a real day; publishing it puts a spurious
## final point on every downstream chart. Do this once, here, so no export can miss it.
report_data = report_data.loc[:REPORT_DATE]



## Fundamentals Table CSV
fundamentals_table.to_csv("csv/fundamentals_table.csv", index=False)

## Summary Table CSV
summary_table.to_csv("csv/summary_table.csv", index=False)

## Fixed Price Outlook CSV
## The outlook year travels with the levels so every consumer can verify it is looking
## at the current forecast rather than trusting its own hardcoded copy.
PRICE_OUTLOOK_LEVELS = PRICE_OUTLOOK_LEVELS.assign(outlook_year=PRICE_OUTLOOK_YEAR)
PRICE_OUTLOOK_LEVELS.to_csv("csv/price_outlook.csv", index=False)

## MTD / YTD Historical Returns — indexed to current-period start price.
## Each historical year's intra-period pattern is applied to the current year's
## starting price, so every line begins at the same dollar value and diverges
## based on each year's actual % change. Plus Median + Average across history.
# Skip years before 2014 — early Bitcoin data is too thin / volatile for clean comparison
INDEXED_RETURNS_MIN_YEAR = 2014

_price = report_data["price_close"]
mtd_history = report_tables.create_indexed_returns_history(
    _price, REPORT_DATE, "mtd", INDEXED_RETURNS_MIN_YEAR
)
mtd_history.to_csv("csv/mtd_returns_history.csv")

ytd_history = report_tables.create_indexed_returns_history(
    _price, REPORT_DATE, "ytd", INDEXED_RETURNS_MIN_YEAR
)
ytd_history.to_csv("csv/ytd_returns_history.csv")


## On-chain Price Models CSV - daily canonical BTC price + model values through report date,
## plus the 50-day, 3-month, 200-day, 1-year and 200-week moving averages
ONCHAIN_PRICE_MODEL_COLS = {
    "price_close": "BTC Price",
    "Electricity_Cost": "Electricity Cost",
    "metcalfe_value": "Metcalfe Value",
    "power_law_price": "Power Law Price",
    "sth_realized_price": "STH Realized Price",
    "lth_realized_price": "LTH Realized Price",
    "realized_price": "Realized Price",
}
onchain_subset = (
    report_data.loc[:REPORT_DATE, list(ONCHAIN_PRICE_MODEL_COLS.keys())]
    .dropna(subset=["price_close"])
)
onchain_subset["3x Realized Price"] = onchain_subset["realized_price"] * 3
onchain_subset = onchain_subset.rename(columns=ONCHAIN_PRICE_MODEL_COLS)
onchain_subset = report_tables.add_price_moving_averages(onchain_subset)
onchain_subset.index.name = "date"
onchain_subset.to_csv("csv/onchain_price_models.csv")


## Summary History CSV - inclusive 30-day comparison window (31 daily endpoints)
HEADLINE_METRICS = {
    "Bitcoin Price USD": "price_close",
    "Bitcoin Marketcap": "market_cap",
    "Sats Per Dollar": "sat_per_dollar",
    "Bitcoin Supply": "supply",
    "Bitcoin Miner Revenue": "coinbase_sum_24h_usd",
    "Bitcoin Transaction Volume": "transfer_volume_sum_24h_usd",
}
summary_history = report_tables.create_summary_history(
    report_data, REPORT_DATE, HEADLINE_METRICS, comparison_days=30
)
summary_history.to_csv("csv/summary_history.csv", index=False)

## Performance Table CSV
performance_table.to_csv("csv/performance_table.csv", index=False)

## Indexed Bitcoin Price Return Comparison CSVs
mtd_return_comp.to_csv("csv/mtd_return_comparison.csv", index=False)
ytd_return_comp.to_csv("csv/ytd_return_comparison.csv", index=False)

## Relative Value Comparison CSV
rv_table.to_csv("csv/relative_value_comparison.csv", index=False)

## ROI Table CSV
roi_table.to_csv("csv/roi_table.csv", index=False)

## Master CSV - All calculated metrics after analysis (includes change calculations)
## Gzipped to reduce file size (~99MB raw → ~5-10MB compressed)
report_data.to_csv("csv/master_metrics_data.csv.gz", index=True, compression="gzip")

from candle_data import write_candle_tables
write_candle_tables(daily_ohlc_data, report_data, REPORT_DATE)

# --- Chart-Ready CSV Exports --- #
# These datasets are consumed by Bitcoin-Chart-Library for visualization

## Drawdown data (ATH drawdown cycles)
drawdown_data = cycles.compute_drawdowns(report_data)
drawdown_data.to_csv("csv/drawdown_data.csv", index=False)

## Cycle low data (market cycle performance from lows)
cycle_low_data = cycles.compute_cycle_lows(report_data)
cycle_low_data.to_csv("csv/cycle_low_data.csv", index=False)

## Halving era data (performance indexed from each halving)
halving_data = cycles.compute_halving_days(report_data)
halving_data.to_csv("csv/halving_data.csv", index=False)


# Downstream consumers verify this complete release before rendering.
# The workflow validates the release before publishing its CSV directory.
from chart_manifest import write_release_manifest
write_release_manifest("csv", REPORT_DATE)
