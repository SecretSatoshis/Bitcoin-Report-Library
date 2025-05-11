## File main.py

# Import Packages
import pandas as pd
import warnings
import datapane as dp
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
    filter_data_columns,
    stats_start_date,
    correlation_data,
    metrics_template,
    chart_template,
)

# Fetch the data
data = data_format.get_data(tickers, market_data_start_date)

## Forward fill the data for all columns
data.ffill(inplace=True)

## Kraken OHLC data
start_timestamp = int(pd.Timestamp("2017-01-01").timestamp())
ohlc_data = data_format.get_kraken_ohlc("XBTUSD", start_timestamp)

## Get Bitcoin Difficulty Blockchain Data Output To Pandas
difficulty_report = data_format.check_difficulty_change()
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

## Flatten the list of columns from the dictionary
columns_to_keep = [item for sublist in filter_data_columns.values() for item in sublist]
filter_data = data[columns_to_keep]

# Create Datasets

## Create Report Data
report_data = data_format.run_data_analysis(filter_data, stats_start_date)

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

## Create Correlation Data
correlation_data = data[correlation_data]

## Create Bitcoin Correlation Data
correlation_results = data_format.create_btc_correlation_data(
    report_date, tickers, correlation_data
)

# Table Creation

# Import Report Functions
import report_tables
from datapane_weekly_bitcoin_recap import generate_report_layout_weekly_bitcoin_recap

# Create the fundamentals table
fundamentals_table = report_tables.create_bitcoin_fundamentals_table(
    report_data, difficulty_period_changes, weekly_high_low, report_date, cagr_results
)
## Create the styled fundamentals table
styled_fundamentals_table = report_tables.style_bitcoin_fundamentals_table(
    fundamentals_table
)
## Create a DataPane table with the styled table
fundamentals_table_dp = dp.Table(styled_fundamentals_table, name="Fundamentals_Table")

# Creating trading range table $5000
bucket_counts_df = report_tables.calculate_price_buckets(data, 5000)
bucket_counts_df[bucket_counts_df["Price Range ($)"] != "$0K-$5K"]
bucket_counts_df = report_tables.style_bucket_counts_table(bucket_counts_df)
trading_range_table_5k = dp.Table(bucket_counts_df)

# Creating trading range table $1000
bucket_counts_df = report_tables.calculate_price_buckets(data, 1000)
trading_range_table_1k = report_tables.create_price_buckets_chart(bucket_counts_df)

# Create ROI Table
roi_table = report_tables.calculate_roi_table(data, report_date)
roi_table = report_tables.style_roi_table(roi_table)
roi_table = dp.Table(roi_table)


# Create Weekly Fundamentals Table
weekly_fundamentals_table = report_tables.create_weekly_metrics_table(
    report_data, metrics_template
)
weekly_fundamentals_table_fig = dp.Table(weekly_fundamentals_table)
weekly_fundamentals_table_df = weekly_fundamentals_table.data

# Create OHLC Table
latest_weekly_ohlc = report_tables.calculate_weekly_ohlc(ohlc_data)

# Create MTD Return Comparison Table
mtd_return_comp = report_tables.create_monthly_returns_table(report_data)

# Create YTD Return Comparison Table
ytd_return_comp = report_tables.create_yearly_returns_table(report_data)

# Create Relative Valuation Table
rv_table = report_tables.create_asset_valuation_table(data)

# Create the summary update table
summary_table = report_tables.create_summary_table_weekly_bitcoin_recap(
    report_data, report_date
)
summary_big_numbers = report_tables.create_summary_big_numbers_weekly_bitcoin_recap(
    summary_table, report_date
)
summary_dp = dp.Table(summary_table, name="Weekly_Summary")

# Create the equity performance table
equity_table = report_tables.create_equity_performance_table(
    report_data, report_date, correlation_results
)
styled_equity_table = report_tables.style_performance_table_weekly_bitcoin_recap(
    equity_table
)
styled_equity_table_dp = dp.Table(styled_equity_table, name="equity_performance_table")

# Create sector performance table
sector_table = report_tables.create_sector_performance_table(
    report_data, report_date, correlation_results
)
styled_sector_table = report_tables.style_performance_table_weekly_bitcoin_recap(
    sector_table
)
styled_sector_table_dp = dp.Table(styled_sector_table, name="sector_performance_table")

# Create macro performance table
macro_table = report_tables.create_macro_performance_table_weekly_bitcoin_recap(
    report_data, report_date, correlation_results
)
styled_macro_table = report_tables.style_performance_table_weekly_bitcoin_recap(
    macro_table
)
styled_macro_table_dp = dp.Table(styled_macro_table, name="macro_performance_table")

# Create bitcoin performance table
bitcoin_table = report_tables.create_bitcoin_performance_table(
    report_data, report_date, correlation_results
)
styled_bitcoin_table = report_tables.style_performance_table_weekly_bitcoin_recap(
    bitcoin_table
)
styled_bitcoin_table_dp = dp.Table(
    styled_bitcoin_table, name="bitcoin_performance_table"
)

# Create the performance table
weekly_bitcoin_recap_performance_table = (
    report_tables.create_full_weekly_bitcoin_recap_performance(
        report_data,
        report_date,
        correlation_results,
    )
)

# Create the styled performance table
styled_weekly_bitcoin_recap_performance_table = (
    report_tables.style_performance_table_weekly_bitcoin_recap(
        weekly_bitcoin_recap_performance_table
    )
)

# Chart Creation

# Create Heat Maps
plotly_monthly_heatmap_chart = report_tables.monthly_heatmap(data)

# Create YOY Plot
yoy_plot = report_tables.create_yoy_change_chart(report_data, "PriceUSD_365_change")

# Create OHLC Plot
ohlc_plot = report_tables.create_ohlc_chart(ohlc_data, report_data, chart_template)

# Datapane Report Creation

## Configure Datapane Report
report_layout_weekly_bitcoin_recap = generate_report_layout_weekly_bitcoin_recap(
    summary_big_numbers,
    ohlc_plot,
    styled_equity_table_dp,
    styled_sector_table_dp,
    styled_macro_table_dp,
    styled_bitcoin_table_dp,
    plotly_monthly_heatmap_chart,
    trading_range_table_1k,
    roi_table,
    fundamentals_table_dp,
    weekly_fundamentals_table_fig,
    report_data,
    report_date,
)


## DataPane Styling
custom_formatting = dp.Formatting(
    light_prose=False,
    accent_color="#000",
    bg_color="#EEE",  # White background
    text_alignment=dp.TextAlignment.LEFT,
    font=dp.FontChoice.SANS,
    width=dp.Width.FULL,
)

# Create Weekly Bitcoin Recap
dp.save_report(
    report_layout_weekly_bitcoin_recap,
    path="html/Weekly_Bitcoin_Recap.html",
    formatting=custom_formatting,
)

# CSV Exports

## Price Bucket CSVs
bucket_counts_df.to_csv("csv/5k_bucket_table.csv")
bucket_counts_df.to_csv("csv/1k_bucket_table.csv")

## Weekly Fundamentals CSV
weekly_fundamentals_table_df.to_csv("csv/weekly_fundamentals_table.csv", index=False)

## Weekly Bitcoin Recap Summary CSV
summary_table.to_csv("csv/weekly_bitcoin_recap_summary.csv", index=False)

## Weekly Bitcoin Recap Performance Table CSV
styled_weekly_bitcoin_recap_performance_table.data.to_csv(
    "csv/weekly_bitcoin_recap_performance_table.csv", index=False
)

## Indexed Bitcoin Price Return Comparison CSVs
mtd_return_comp.to_csv(
    "csv/weekly_bitcoin_recap_mtd_return_comparison.csv", index=False
)
ytd_return_comp.to_csv(
    "csv/weekly_bitcoin_recap_ytd_return_comparison.csv", index=False
)

## Relative Value Comparison CSV
rv_table.to_csv("csv/weekly_bitcoin_recap_relative_value_comparison.csv", index=False)

## EOY Price Model Data CSV
eoy_model_data = report_tables.create_eoy_model_table(data, cagr_results)
eoy_model_data.to_csv("csv/eoy_model_data.csv", index=True)
