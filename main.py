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
    market_data_start_date,
    moving_avg_metrics,
    fiat_money_data_top10,
    gold_silver_supply,
    gold_supply_breakdown,
    stock_tickers,
    report_date,
    filter_data_columns,
    stats_start_date,
    valuation_metrics,
    correlation_data,
    metrics_template,
    chart_template,
)

# Fetch the data
data = data_format.get_data(tickers, market_data_start_date)
data = data_format.calculate_custom_on_chain_metrics(data)
data = data_format.calculate_moving_averages(data, moving_avg_metrics)
data = data_format.calculate_btc_price_to_surpass_fiat(data, fiat_money_data_top10)
data = data_format.calculate_metal_market_caps(data, gold_silver_supply)
data = data_format.calculate_gold_market_cap_breakdown(data, gold_supply_breakdown)
data = data_format.calculate_btc_price_to_surpass_metal_categories(
    data, gold_supply_breakdown
)
data = data_format.calculate_btc_price_for_stock_mkt_caps(data, stock_tickers)
data = data_format.calculate_stock_to_flow_metrics(data)
data = data_format.electric_price_models(data)

# Forward fill the data for all columns
data.ffill(inplace=True)

# Flatten the list of columns from the dictionary
columns_to_keep = [item for sublist in filter_data_columns.values() for item in sublist]

# Filter the dataframe
filter_data = data[columns_to_keep]

# Run Data Analysis On Report Data
report_data = data_format.run_data_analysis(filter_data, stats_start_date)

# Kraken OHLC data
start_timestamp = int(pd.Timestamp("2017-01-01").timestamp())
ohlc_data = data_format.get_kraken_ohlc("XBTUSD", start_timestamp)

# Get Bitcoin Difficulty Blockchain Data
difficulty_report = data_format.check_difficulty_change()
# Calcualte Difficulty Period Changes
difficulty_period_changes = data_format.calculate_difficulty_period_change(
    difficulty_report, report_data
)
# Format Bitcoin Difficulty Blockchain Data Output To Pandas
difficulty_report = pd.DataFrame([difficulty_report])

# Calcualte Valuation Target Data
valuation_data = data_format.create_valuation_data(
    report_data, valuation_metrics, report_date
)

# Calcualte 52 Week High Low Based On Report Timeframe
weekly_high_low = data_format.calculate_52_week_high_low(report_data, report_date)

# Calcualte Growth Rate Data
cagr_results = data_format.calculate_rolling_cagr_for_all_metrics(data)

# Calcuate Sharpe Ratio Data
sharpe_data = data[correlation_data]
sharpe_results = data_format.calculate_daily_sharpe_ratios(sharpe_data)

# Calcuate Correlations
correlation_data = data[correlation_data]
# Drop NA Values
correlation_data = correlation_data.dropna()
# Calculate Bitcoin Correlations
correlation_results = data_format.create_btc_correlation_tables(
    report_date, tickers, correlation_data
)
### Tabel Creation

# Import Report Table Functions
import report_tables

# Datapane Report Imports
from datapane_difficulty_adjustment_report import generate_report_layout_difficulty
from datapane_weekly_market_update import generate_report_layout_weekly
from datapane_weekly_bitcoin_recap import generate_report_layout_weekl_bitcoin_recap

### Difficulty Adjustment Report

# Create the difficulty update table
difficulty_update_table = report_tables.create_difficulty_update_table(
    report_data, difficulty_report, report_date, difficulty_period_changes
)
# Create the difficulty big numbers
difficulty_big_numbers = report_tables.create_difficulty_big_numbers(
    difficulty_update_table
)
# Create the difficulty big bumbers block
difficulty_update_summary_dp = dp.Table(
    difficulty_update_table, name="Difficulty_Summary"
)

# Create the performance table
performance_table = report_tables.create_performance_table(
    report_data,
    difficulty_period_changes,
    report_date,
    weekly_high_low,
    cagr_results,
    sharpe_results,
    correlation_results,
)

# Create the styled performance table
styled_performance_table = report_tables.style_performance_table_weekly(
    performance_table
)
# Create a DataPane table with the styled table
performance_table_dp = dp.Table(styled_performance_table, name="Performance_Table")

# Create the fundamentals table
fundamentals_table = report_tables.create_bitcoin_fundamentals_table(
    report_data, difficulty_period_changes, weekly_high_low, report_date, cagr_results
)
# Create the styled fundamentals table
styled_fundamentals_table = report_tables.style_bitcoin_fundamentals_table(
    fundamentals_table
)
# Create a DataPane table with the styled table
fundamentals_table_dp = dp.Table(styled_fundamentals_table, name="Fundamentals_Table")

# Create the valuation table
valuation_table = report_tables.create_bitcoin_valuation_table(
    report_data, difficulty_period_changes, weekly_high_low, valuation_data, report_date
)
# Create the styled valuation table
styled_valuation_table = report_tables.style_bitcoin_valuation_table(valuation_table)
# Create a DataPane table with the styled table
valuation_table_dp = dp.Table(styled_valuation_table, name="Valuation_Table")

# Create the model table
model_table = report_tables.create_bitcoin_model_table(
    report_data, report_date, cagr_results
)
# Create the styled model table
styled_model_table = report_tables.style_bitcoin_model_table(model_table)

# Configure Datapane Report
report_layout_difficulty = generate_report_layout_difficulty(
    difficulty_big_numbers,
    performance_table_dp,
    fundamentals_table_dp,
    valuation_table_dp,
)

# DataPane Styling
custom_formatting = dp.Formatting(
    light_prose=False,
    accent_color="#000",
    bg_color="#EEE",  # White background
    text_alignment=dp.TextAlignment.LEFT,
    font=dp.FontChoice.SANS,
    width=dp.Width.FULL,
)


### Weekly Market Update

# Create the difficulty update table
weekly_summary_table = report_tables.create_weekly_summary_table(
    report_data, report_date
)
weekly_summary_big_numbers = report_tables.create_weekly_summary_big_numbers(
    weekly_summary_table
)
weekly_summary_dp = dp.Table(weekly_summary_table, name="Weekly_Summary")

# Create the crypto performance table
crypto_performance_table = report_tables.create_crypto_performance_table(
    report_data, data, report_date, correlation_results
)
styled_crypto_performance_table = report_tables.style_performance_table(
    crypto_performance_table
)
styled_crypto_performance_table_dp = dp.Table(
    styled_crypto_performance_table, name="crypto_performance_table"
)

# Create index performance table
index_performance_table = report_tables.create_index_performance_table(
    report_data, data, report_date, correlation_results
)
styled_index_performance_table = report_tables.style_performance_table(
    index_performance_table
)
styled_index_performance_table_dp = dp.Table(
    styled_index_performance_table, name="index_performance_table"
)

# Create macro performance table
macro_performance_table = report_tables.create_macro_performance_table(
    report_data, data, report_date, correlation_results
)
styled_macro_performance_table = report_tables.style_performance_table(
    macro_performance_table
)
styled_macro_performance_table_dp = dp.Table(
    styled_macro_performance_table, name="macro_performance_table"
)

# Create equities performance table
equities_performance_table = report_tables.create_equities_performance_table(
    report_data, data, report_date, correlation_results
)
styled_equities_performance_table = report_tables.style_performance_table(
    equities_performance_table
)
styled_equities_performance_table_dp = dp.Table(
    styled_equities_performance_table, name="equities_performance_table"
)

# Creating trading range table $5000
bucket_counts_df = data_format.calculate_price_buckets(data, 5000)
bucket_counts_df[bucket_counts_df["Price Range ($)"] != "$0K-$5K"]
bucket_counts_df = data_format.style_bucket_counts_table(bucket_counts_df)
trading_range_table_5k = dp.Table(bucket_counts_df)
# Creating trading range table $1000
bucket_counts_df = data_format.calculate_price_buckets(data, 1000)
trading_range_table_1k = data_format.create_price_buckets_chart(bucket_counts_df)

# Create ROI Table
roi_table = data_format.calculate_roi_table(data, report_date)
roi_table = data_format.style_roi_table(roi_table)
roi_table = dp.Table(roi_table)

# Create MA Table
ma_table = data_format.calculate_ma_table(data)
ma_table = data_format.style_ma_table(ma_table)
ma_table = dp.Table(ma_table)

# Create Fundamentals & Valuation Table
table_df = data_format.create_metrics_table(report_data, metrics_template)
table_fig = dp.Table(table_df)
table_df = table_df.data

# Create OHLC Table
latest_weekly_ohlc = data_format.calculate_weekly_ohlc(ohlc_data)

# Create MTD Return Comparison Table
mtd_return_comp = data_format.create_monthly_returns_table(report_data)

# Create YTD Return Comparison Table
ytd_return_comp = data_format.create_yearly_returns_table(report_data)

# Create Relative Valuation Table
rv_table = data_format.create_asset_valuation_table(data)

# Creat Heat Maps
plotly_monthly_heatmap_chart = data_format.monthly_heatmap(data)
plotly_weekly_heatmap_chart = data_format.weekly_heatmap(data)

# Create YOY Plot
yoy_plot = data_format.plot_yoy_change(report_data, "PriceUSD_365_change")

# Creat OHLC Plot
ohlc_plot = data_format.create_ohlc_chart(ohlc_data, report_data, chart_template)

# Configure Datapane Report
report_layout_weekly = generate_report_layout_weekly(
    weekly_summary_big_numbers,
    styled_crypto_performance_table_dp,
    styled_index_performance_table_dp,
    styled_macro_performance_table_dp,
    styled_equities_performance_table_dp,
    trading_range_table_5k,
    trading_range_table_1k,
    roi_table,
    table_fig,
    plotly_monthly_heatmap_chart,
    plotly_weekly_heatmap_chart,
    yoy_plot,
    ohlc_plot,
)


## Weekly Bitcoin Recap

# Create the summary update table
summary_table = report_tables.create_summary_table_weekly_bitcoin_recap(
    report_data, report_date
)
summary_big_numbers = report_tables.create_summary_big_numbers_weekly_bitcoin_recap(
    summary_table
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

# Configure Datapane Report
report_layout_weekly_bitcoin_recap = generate_report_layout_weekl_bitcoin_recap(
    summary_big_numbers,
    ohlc_plot,
    styled_equity_table_dp,
    styled_sector_table_dp,
    styled_macro_table_dp,
    styled_bitcoin_table_dp,
    plotly_monthly_heatmap_chart,
    report_data,
    report_date,
)

# Create the performance table
weekly_bitcoin_recap_performance_table = (
    report_tables.create_full_weekly_bitcoin_recap_performance(
        report_data,
        difficulty_period_changes,
        report_date,
        weekly_high_low,
        cagr_results,
        sharpe_results,
        correlation_results,
    )
)

# Create the styled performance table
styled_weekly_bitcoin_recap_performance_table = (
    report_tables.style_performance_table_weekly(weekly_bitcoin_recap_performance_table)
)

# DataPane Styling
custom_formatting = dp.Formatting(
    light_prose=False,
    accent_color="#000",
    bg_color="#EEE",  # White background
    text_alignment=dp.TextAlignment.LEFT,
    font=dp.FontChoice.SANS,
    width=dp.Width.FULL,
)

# Create CSV Files
difficulty_update_table.to_csv("csv/difficulty_table.csv", index=False)
styled_performance_table.data.to_csv("csv/performance_table.csv", index=False)
styled_fundamentals_table.data.to_csv("csv/fundamentals_table.csv", index=False)
styled_valuation_table.data.to_csv("csv/valuation_table.csv", index=False)
styled_model_table.data.to_csv("csv/model_table.csv", index=False)

weekly_summary_table.to_csv("csv/weekly_summary.csv", index=False)
styled_crypto_performance_table.data.to_csv(
    "csv/crypto_performance_table.csv", index=False
)
styled_index_performance_table.data.to_csv(
    "csv/index_performance_table.csv", index=False
)
styled_macro_performance_table.data.to_csv(
    "csv/macro_performance_table.csv", index=False
)
styled_equities_performance_table.data.to_csv(
    "csv/equities_performance_table.csv", index=False
)
bucket_counts_df.to_csv("csv/5k_bucket_table.csv")
bucket_counts_df.to_csv("csv/1k_bucket_table.csv")
table_df.to_csv("csv/fundamentals_valuation_table.csv", index=False)

# Weekly Bitcoin Recap CSV
summary_table.to_csv("csv/weekly_bitcoin_recap_summary.csv", index=False)
styled_weekly_bitcoin_recap_performance_table.data.to_csv(
    "csv/weekly_bitcoin_recap_performance_table.csv", index=False
)

# Indexed Return Comparison Tables
mtd_return_comp.to_csv("csv/weekly_bitcoin_recap_mtd_return_comparison.csv", index=False)
ytd_return_comp.to_csv("csv/weekly_bitcoin_recap_ytd_return_comparison.csv", index=False)

# Relative Value Comparison Table
rv_table.to_csv("csv/weekly_bitcoin_recap_relative_value_comparison.csv", index=False)

# Create Weekly Market Summary Report
dp.save_report(
    report_layout_weekly,
    path="html/Weekly_Market_Summary.html",
    formatting=custom_formatting,
)

# Create Difficulty Report
dp.save_report(
    report_layout_difficulty,
    path="html/Difficulty_Adjustment_Report.html",
    formatting=custom_formatting,
)

# Creat Weekly Bitcoin Recap
dp.save_report(
    report_layout_weekly_bitcoin_recap,
    path="html/Weekly_Bitcoin_Recap.html",
    formatting=custom_formatting,
)


