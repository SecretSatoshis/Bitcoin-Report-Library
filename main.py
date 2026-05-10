"""
Bitcoin Report Library - Main Pipeline

This script orchestrates the complete data pipeline for Bitcoin market and on-chain analytics.
It fetches data from multiple sources, calculates metrics, generates report tables, and exports
CSV files for downstream analysis.
"""

# Import Packages
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
)

# Fetch the data
data = data_format.get_data(tickers, market_data_start_date)

## Forward fill the data for all columns
data = data.ffill()

## BRK OHLC data
ohlc_data = data_format.get_brk_ohlc(index="week1", start="2017-01-01")
ohlc_data.index = pd.to_datetime(ohlc_data.index)
if ohlc_data.index.tz is not None:
    ohlc_data.index = ohlc_data.index.tz_convert(None)

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
bucket_counts_5k_df = report_tables.calculate_price_buckets(data, 5000)

# Creating trading range table $1000
bucket_counts_1k_df = report_tables.calculate_price_buckets(data, 1000)

# Create ROI Table
roi_table = report_tables.calculate_roi_table(data, report_date)

# Create Fundamentals Table
fundamentals_table = report_tables.create_fundamentals_table(
    report_data, metrics_template
)

# Create OHLC CSV
report_tables.calculate_ohlc(ohlc_data)

# Create MTD Return Comparison Table
mtd_return_comp = report_tables.create_monthly_returns_table(report_data, report_date)

# Create YTD Return Comparison Table
ytd_return_comp = report_tables.create_yearly_returns_table(report_data, report_date)

# Create Relative Valuation Table
rv_table = report_tables.create_asset_valuation_table(data)

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
report_tables.monthly_heatmap(data)


# CSV Exports

## Price Bucket CSVs
bucket_counts_5k_df.to_csv("csv/5k_bucket_table.csv", index=False)
bucket_counts_1k_df.to_csv("csv/1k_bucket_table.csv", index=False)

## Fundamentals Table CSV
fundamentals_table.to_csv("csv/fundamentals_table.csv", index=False)

## Summary Table CSV
summary_table.to_csv("csv/summary_table.csv", index=False)

## Fixed Price Outlook CSV
price_outlook_levels.to_csv("csv/price_outlook.csv", index=False)

## MTD / YTD Historical Returns — indexed to current-period start price.
## Each historical year's intra-period pattern is applied to the current year's
## starting price, so every line begins at the same dollar value and diverges
## based on each year's actual % change. Plus Median + Average across history.
# Skip years before 2014 — early Bitcoin data is too thin / volatile for clean comparison
INDEXED_RETURNS_MIN_YEAR = 2014


def _build_indexed_returns(price_series, group_func, current_year):
    """Builds wide DataFrame: x-axis = day index, columns = years + Median + Average."""
    current_period = group_func(price_series, current_year)
    if current_period.empty or pd.isna(current_period.iloc[0]):
        return pd.DataFrame()
    base_price = current_period.iloc[0]

    out = {}
    for year in sorted(price_series.index.year.unique()):
        if year < INDEXED_RETURNS_MIN_YEAR:
            continue
        period = group_func(price_series, year)
        if period.empty:
            continue
        first = period.iloc[0]
        if pd.isna(first) or first <= 0:
            continue
        # Re-index to current year's starting dollar value, key by day-of-period
        scaled = (period / first) * base_price
        scaled.index = (
            scaled.index.day if group_func is _slice_current_month
            else scaled.index.dayofyear
        )
        out[str(year)] = scaled

    df_out = pd.DataFrame(out).sort_index()
    historical = [c for c in df_out.columns if int(c) < current_year]
    if historical:
        df_out["Median"] = df_out[historical].median(axis=1)
        df_out["Average"] = df_out[historical].mean(axis=1)
    return df_out


def _slice_current_month(series, year):
    latest_month = report_data.index.max().month
    return series[(series.index.year == year) & (series.index.month == latest_month)]


def _slice_full_year(series, year):
    return series[series.index.year == year]


_current_year = report_data.index.max().year
_price = report_data["price_close"].dropna()

mtd_history = _build_indexed_returns(_price, _slice_current_month, _current_year)
mtd_history.index.name = "day"
mtd_history.to_csv("csv/mtd_returns_history.csv")

ytd_history = _build_indexed_returns(_price, _slice_full_year, _current_year)
ytd_history.index.name = "day_of_year"
ytd_history.to_csv("csv/ytd_returns_history.csv")


## On-chain Price Models CSV - daily canonical BTC price + model values through report date
ONCHAIN_PRICE_MODEL_COLS = {
    "price_close": "BTC Price",
    "Hayes_Network_Price_Per_BTC": "Electricity Cost",
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


## Summary History CSV - last 30 days of headline metrics for dashboard sparklines & deltas
HEADLINE_METRICS = {
    "Bitcoin Price USD": "price_close",
    "Bitcoin Marketcap": "market_cap",
    "Sats Per Dollar": "sat_per_dollar",
    "Bitcoin Supply": "supply",
    "Bitcoin Miner Revenue": "coinbase_sum_24h_usd",
    "Bitcoin Transaction Volume": "transfer_volume_sum_24h_usd",
    "Bitcoin Dominance": "bitcoin_dominance",
    "Bitcoin Fear & Greed Index": "fear_greed_value",
}
last_30 = report_data.loc[:report_date].tail(30)
history_rows = []
for label, col in HEADLINE_METRICS.items():
    if col in last_30.columns:
        for date_idx, val in last_30[col].dropna().items():
            history_rows.append({
                "Metric": label,
                "date": (
                    date_idx.strftime("%Y-%m-%d")
                    if hasattr(date_idx, "strftime")
                    else str(date_idx)
                ),
                "Value": val,
            })
pd.DataFrame(history_rows).to_csv("csv/summary_history.csv", index=False)

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
eoy_model_data = report_tables.create_eoy_model_table(data, cagr_results)
eoy_model_data.to_csv("csv/eoy_model_data.csv", index=True)

## Master CSV - All calculated metrics after analysis (includes change calculations)
## Gzipped to reduce file size (~99MB raw → ~5-10MB compressed)
report_data.to_csv("csv/master_metrics_data.csv.gz", index=True, compression="gzip")

## Remove old uncompressed master if it exists (prevent stale 99MB file in repo)
import os
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
