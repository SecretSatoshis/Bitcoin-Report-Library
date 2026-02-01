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
    fiat_money_data_top10,
    gold_silver_supply,
    gold_supply_breakdown,
    analysis_columns,
    stats_start_date,
    correlation_data,
    metrics_template,
)

# Fetch the data
data = data_format.get_data(tickers, market_data_start_date)

## Forward fill the data for all columns
data = data.ffill()

## Kraken OHLC data
start_timestamp = int(pd.Timestamp("2017-01-01").timestamp())
ohlc_data = data_format.get_kraken_ohlc("XBTUSD", start_timestamp)

## Get Bitcoin Difficulty Data from BRK API
difficulty_report = data_format.check_difficulty_change(data)
difficulty_report_df = pd.DataFrame([difficulty_report])

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

## Create Difficulty Period Change Data
difficulty_period_changes = data_format.calculate_difficulty_period_change(
    difficulty_report, report_data
)

## Create 52 Week High Low Based On Report Timeframe
weekly_high_low = data_format.calculate_52_week_high_low(report_data, report_date)

## Create Growth Rate Data
cagr_results = data_format.calculate_rolling_cagr_for_all_metrics(data)

## Filter Stat Data
stat_data = data[correlation_data]

## Create Sharpe Ratio Data
sharpe_results = data_format.calculate_daily_sharpe_ratios(stat_data)

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

# Create OHLC Table
ohlc_table = report_tables.calculate_ohlc(ohlc_data)

# Create MTD Return Comparison Table
mtd_return_comp = report_tables.create_monthly_returns_table(report_data)

# Create YTD Return Comparison Table
ytd_return_comp = report_tables.create_yearly_returns_table(report_data)

# Create Relative Valuation Table
rv_table = report_tables.create_asset_valuation_table(data)

# Create the summary table
summary_table = report_tables.create_summary_table(
    report_data, report_date
)
# Create the equity performance table
equity_table = report_tables.create_equity_performance_table(
    report_data, report_date, correlation_results
)
# Create sector performance table
sector_table = report_tables.create_sector_performance_table(
    report_data, report_date, correlation_results
)

# Create macro performance table
macro_table = report_tables.create_macro_performance_table(
    report_data, report_date, correlation_results
)

# Create bitcoin performance table
bitcoin_table = report_tables.create_bitcoin_performance_table(
    report_data, report_date, correlation_results
)

# Create the performance table
performance_table = (
    report_tables.create_full_performance_table(
        report_data,
        report_date,
        correlation_results,
    )
)


# Create Heat Maps
monthly_heatmap = report_tables.monthly_heatmap(data)


# CSV Exports

## Price Bucket CSVs
bucket_counts_5k_df.to_csv("csv/5k_bucket_table.csv", index=False)
bucket_counts_1k_df.to_csv("csv/1k_bucket_table.csv", index=False)

## Fundamentals Table CSV
fundamentals_table.to_csv("csv/fundamentals_table.csv", index=False)

## Summary Table CSV
summary_table.to_csv("csv/summary_table.csv", index=False)

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

## Master CSV - All calculated metrics after analysis
data.to_csv("csv/master_metrics_data.csv", index=True)
