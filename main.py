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
        "data_format or report_tables instead, or run `python main.py`."
    )

# Import Packages
import os

import pandas as pd
import warnings
import sys


# Ignore FutureWarning & Cache
warnings.simplefilter(action="ignore", category=FutureWarning)
sys.dont_write_bytecode = True

# Import Files
import data_format

from data_definitions import (
    tickers,
    stock_tickers,
    report_date,
    market_data_start_date,
    moving_avg_metrics,
    cagr_columns,
    fiat_money_data_top10,
    gold_silver_supply,
    gold_supply_breakdown,
    analysis_columns,
    stats_start_date,
    correlation_data,
    metrics_template,
    price_outlook_levels,
    PRICE_OUTLOOK_YEAR,
)

# Fetch the data
data = data_format.get_data(
    tickers,
    market_data_start_date,
    report_date=report_date,
    bitcoin_dominance_history_path="csv/bitcoin_dominance_history.csv",
)

## Forward fill market data only.
## Equities/ETFs/FX print on trading days and miner efficiency prints monthly, so both
## need carrying forward onto Bitcoin's 365-day index. On-chain series print daily, and
## filling those would turn a missing or malformed BRK response into a silent repeat of
## yesterday's values, so they are validated instead.
data_format.warn_on_stale_market_data(data, report_date)
data = data_format.forward_fill_market_data(data)
data_format.assert_onchain_freshness(data, report_date)
data_format.assert_no_internal_onchain_gaps(data, report_date)
data_format.assert_reference_data_fresh(report_date)
data_format.assert_price_outlook_current(report_date)
data_format.warn_on_stale_miner_efficiency(data, report_date)

## BRK OHLC data
ohlc_data = data_format.get_brk_ohlc(index="week1", start="2017-01-01")
ohlc_data.index = pd.to_datetime(ohlc_data.index)
if ohlc_data.index.tz is not None:
    ohlc_data.index = ohlc_data.index.tz_convert(None)
data_format.assert_ohlc_usable(ohlc_data, label="Weekly BRK OHLC")

daily_ohlc_start = (pd.to_datetime(report_date) - pd.Timedelta(days=14)).strftime(
    "%Y-%m-%d"
)
daily_ohlc_data = data_format.get_brk_ohlc(index="day1", start=daily_ohlc_start)
daily_ohlc_data.index = pd.to_datetime(daily_ohlc_data.index)
if daily_ohlc_data.index.tz is not None:
    daily_ohlc_data.index = daily_ohlc_data.index.tz_convert(None)
data_format.assert_ohlc_usable(daily_ohlc_data, label="Daily BRK OHLC")

# Calculate Custom Metrics
data = data_format.calculate_custom_on_chain_metrics(data)
data = data_format.calculate_moving_averages(data, moving_avg_metrics)

## Fiat / Gold Calculations
data = data_format.calculate_btc_price_to_surpass_fiat(data, fiat_money_data_top10)
data = data_format.calculate_metal_market_caps(data, gold_silver_supply)
data = data_format.calculate_gold_market_cap_breakdown(data, gold_supply_breakdown)
data = data_format.calculate_btc_price_to_surpass_metal_categories(data, gold_supply_breakdown)

## Calculate On-chain Models
data = data_format.calculate_btc_price_for_stock_mkt_caps(data, stock_tickers)
data = data_format.calculate_stock_to_flow_metrics(data)
data = data_format.calculate_network_model_metrics(data, report_date)
data = data_format.electric_price_models(data)

# Create Datasets

## Create Report Data - only calculate changes for columns that need them
analysis_data = data[analysis_columns]
report_data = data_format.run_data_analysis(analysis_data, stats_start_date)

## Merge the change columns back with the full data
report_data = pd.concat([data, report_data.drop(columns=analysis_columns)], axis=1)

## Create Growth Rate Data — only compute CAGR for the 13 columns actually used downstream.
## Filter to columns present in data (some valuation models may not exist on early dates).
cagr_input_cols = [c for c in cagr_columns if c in data.columns]
cagr_results = data_format.calculate_rolling_cagr_for_all_metrics(data[cagr_input_cols])

## Merge only the CAGR columns that Chart Library charts actually reference.
## Full CAGR data is exported separately as cagr_data.csv.
chart_cagr_columns = [
    "price_close_4_Year_CAGR",
    "SPY_close_4_Year_CAGR",
    "QQQ_close_4_Year_CAGR",
    "XLK_close_4_Year_CAGR",
    "XLF_close_4_Year_CAGR",
    "GLD_close_4_Year_CAGR",
    "AGG_close_4_Year_CAGR",
    "DX-Y.NYB_close_4_Year_CAGR",
    "WGMI_close_4_Year_CAGR",
]
available_cagr = [c for c in chart_cagr_columns if c in cagr_results.columns]
report_data = report_data.merge(
    cagr_results[available_cagr], left_index=True, right_index=True, how="left"
)

## Create Correlation Data (renamed to avoid variable collision)
correlation_df = data[correlation_data]

## Create Bitcoin Correlation Data
correlation_results = data_format.create_btc_correlation_data(
    report_date, tickers, correlation_df
)

# Table Creation

# Import Report Functions
import report_tables

# Creating trading range table $5000
bucket_counts_5k_df = report_tables.calculate_price_buckets(data, 5000, report_date)

# Creating trading range table $1000
bucket_counts_1k_df = report_tables.calculate_price_buckets(data, 1000, report_date)

# Create ROI Table
roi_table = report_tables.calculate_roi_table(data, report_date)

# Create Fundamentals Table
fundamentals_table = report_tables.create_fundamentals_table(
    report_data, metrics_template, report_date
)

# Create OHLC CSV
report_tables.calculate_ohlc(ohlc_data)
report_tables.create_report_ohlc_summary(daily_ohlc_data, report_date)

# Create MTD Return Comparison Table
mtd_return_comp = report_tables.create_monthly_returns_table(report_data, report_date)

# Create YTD Return Comparison Table
ytd_return_comp = report_tables.create_yearly_returns_table(report_data, report_date)

# Create Relative Valuation Table
rv_table = report_tables.create_asset_valuation_table(report_data, report_date)

# Create the summary table
summary_table = report_tables.create_summary_table(
    report_data, report_date
)
# Create the performance table
performance_table = (
    report_tables.create_full_performance_table(
        report_data,
        report_date,
        correlation_results,
    )
)


# Create Heat Map CSV
report_tables.monthly_heatmap(report_data, report_date)

# Create daily electricity tariff scenarios and retain the other energy-model
# definitions alongside them for explicit comparison.
electricity_cost_scenarios = report_tables.create_electricity_cost_scenarios(
    report_data, report_date
)
network_model_metrics = report_tables.create_network_model_metrics(
    report_data, report_date
)


# CSV Exports

## Every exported frame is truncated to the report date. Upstream fetches return a
## partial, in-progress UTC day whose 24h aggregates (hash rate, miner revenue, tx
## count, supply issuance) are a fraction of a real day; publishing it puts a spurious
## final point on every downstream chart. Do this once, here, so no export can miss it.
report_data = report_data.loc[:report_date]
cagr_results = cagr_results.loc[:report_date]


## Price Bucket CSVs
bucket_counts_5k_df.to_csv("csv/5k_bucket_table.csv", index=False)
bucket_counts_1k_df.to_csv("csv/1k_bucket_table.csv", index=False)

## Fundamentals Table CSV
fundamentals_table.to_csv("csv/fundamentals_table.csv", index=False)

## Summary Table CSV
summary_table.to_csv("csv/summary_table.csv", index=False)

## Fixed Price Outlook CSV
## The outlook year travels with the levels so every consumer can verify it is looking
## at the current forecast rather than trusting its own hardcoded copy.
price_outlook_levels = price_outlook_levels.assign(outlook_year=PRICE_OUTLOOK_YEAR)
price_outlook_levels.to_csv("csv/price_outlook.csv", index=False)

## MTD / YTD Historical Returns — indexed to current-period start price.
## Each historical year's intra-period pattern is applied to the current year's
## starting price, so every line begins at the same dollar value and diverges
## based on each year's actual % change. Plus Median + Average across history.
# Skip years before 2014 — early Bitcoin data is too thin / volatile for clean comparison
INDEXED_RETURNS_MIN_YEAR = 2014

_price = report_data["price_close"]
mtd_history = report_tables.create_indexed_returns_history(
    _price, report_date, "mtd", INDEXED_RETURNS_MIN_YEAR
)
mtd_history.to_csv("csv/mtd_returns_history.csv")

ytd_history = report_tables.create_indexed_returns_history(
    _price, report_date, "ytd", INDEXED_RETURNS_MIN_YEAR
)
ytd_history.to_csv("csv/ytd_returns_history.csv")


## On-chain Price Models CSV - daily canonical BTC price + model values through report date
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
    report_data.loc[:report_date, list(ONCHAIN_PRICE_MODEL_COLS.keys())]
    .dropna(subset=["price_close"])
)
onchain_subset["3x Realized Price"] = onchain_subset["realized_price"] * 3
onchain_subset = onchain_subset.rename(columns=ONCHAIN_PRICE_MODEL_COLS)
onchain_subset.index.name = "date"
onchain_subset.to_csv("csv/onchain_price_models.csv")

## Electricity tariff scenarios — power expense per observed BTC earned plus
## the retained legacy/production/Hayes/Energy Value models for comparison.
electricity_cost_scenarios.to_csv("csv/electricity_cost_scenarios.csv")

## Detailed Metcalfe, power-law, and hash-ribbon model inputs and outputs.
network_model_metrics.to_csv("csv/network_model_metrics.csv")


## Summary History CSV - inclusive 30-day comparison window (31 daily endpoints)
## Bitcoin Dominance is maintained separately in bitcoin_dominance_history.csv. Its
## required report-date value still flows into summary_table, but it stays out of this
## fixed 30-day headline window until the dedicated history has accumulated enough observations.
HEADLINE_METRICS = {
    "Bitcoin Price USD": "price_close",
    "Bitcoin Marketcap": "market_cap",
    "Sats Per Dollar": "sat_per_dollar",
    "Bitcoin Supply": "supply",
    "Bitcoin Miner Revenue": "coinbase_sum_24h_usd",
    "Bitcoin Transaction Volume": "transfer_volume_sum_24h_usd",
    "Bitcoin Fear & Greed Index": "fear_greed_value",
}
summary_history = report_tables.create_summary_history(
    report_data, report_date, HEADLINE_METRICS, comparison_days=30
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

## EOY Price Model Data CSV
eoy_model_data = report_tables.create_eoy_model_table(
    data, cagr_results, report_date
)
eoy_model_data.to_csv("csv/eoy_model_data.csv", index=True)

## Master CSV - All calculated metrics after analysis (includes change calculations)
## Gzipped to reduce file size (~99MB raw → ~5-10MB compressed)
report_data.to_csv("csv/master_metrics_data.csv.gz", index=True, compression="gzip")

## Remove old uncompressed master if it exists (prevent stale 99MB file in repo)
old_master = "csv/master_metrics_data.csv"
if os.path.exists(old_master):
    os.remove(old_master)

# --- Chart-Ready CSV Exports --- #
# These datasets are consumed by Bitcoin-Chart-Library for visualization

## Drawdown data (ATH drawdown cycles)
drawdown_data = data_format.compute_drawdowns(report_data)
drawdown_data.to_csv("csv/drawdown_data.csv", index=False)

## Cycle low data (market cycle performance from lows)
cycle_low_data = data_format.compute_cycle_lows(report_data)
cycle_low_data.to_csv("csv/cycle_low_data.csv", index=False)

## Halving era data (performance indexed from each halving)
halving_data = data_format.compute_halving_days(report_data)
halving_data.to_csv("csv/halving_data.csv", index=False)

## CAGR results
cagr_results.to_csv("csv/cagr_data.csv", index=True)

## Model coefficients — the power-law and Metcalfe fits are re-run every day over all
## observations through the report date, so `power_law_price` and `metcalfe_value` change
## retroactively for every historical date on every run. Publishing the coefficients that
## produced this release makes any cited value reproducible from the release it came from
## instead of drifting silently run over run.
MODEL_COEFFICIENT_COLUMNS = [
    "power_law_exponent",
    "power_law_scale",
    "metcalfe_scale_any_balance",
    "metcalfe_scale_0p001_btc",
    "metcalfe_scale_0p01_btc",
    "metcalfe_scale_0p1_btc",
]
_coefficient_row = report_data.loc[report_date]
model_coefficients = pd.DataFrame(
    {
        "coefficient": MODEL_COEFFICIENT_COLUMNS,
        "value": [
            float(_coefficient_row[column]) for column in MODEL_COEFFICIENT_COLUMNS
        ],
        "report_date": report_date.strftime("%Y-%m-%d"),
        "fit_end_date": report_date.strftime("%Y-%m-%d"),
    }
)
model_coefficients.to_csv("csv/model_coefficients.csv", index=False)
