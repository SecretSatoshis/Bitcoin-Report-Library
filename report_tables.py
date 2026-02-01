"""
Report Tables Module - Bitcoin Analytics Table Generation

This module generates formatted tabular outputs for Bitcoin market and on-chain analytics.
All functions produce CSV-ready DataFrames without styling, optimized for direct ingestion
into spreadsheets, BI tools, or visualization platforms.

Key Responsibilities:
    - Fundamentals Tables: Network performance, security, economics, and valuation metrics
    - Performance Tables: Multi-asset return comparisons across equities, sectors, macro, and Bitcoin
    - ROI Analysis: Historical return calculations for multiple time periods
    - Price Distribution: Trading day counts by price bucket ranges
    - Temporal Analysis: Monthly/yearly return comparisons, OHLC resampling, heatmaps
    - Valuation Models: Relative value comparisons and end-of-year projections

Output Format:
    All functions return unstyled pandas DataFrames ready for CSV export. Formatting is
    minimal and data-focused to ensure compatibility with downstream analysis tools.
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pandas.tseries.offsets import MonthEnd
import calendar
from data_definitions import SATS_PER_BTC


def calculate_price_buckets(data, bucket_size):
    """
    Calculates the number of unique trading days the price spent in each bucket range.

    Parameters:
    data (pd.DataFrame): DataFrame with DateTime index and a 'price_close' column.
    bucket_size (int): The size of each price bucket.

    Returns:
    pd.DataFrame: DataFrame containing counts of unique trading days in each price bucket.
    """
    # Ensure the DataFrame is sorted by time
    data = data.sort_index(ascending=True)

    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")

    # Filter out NaN values before processing
    data = data.dropna(subset=["price_close"])

    # Remove duplicate intra-day price fluctuations by keeping only one entry per day
    data = data.groupby(data.index.floor("D")).first()

    # Define the bucket ranges for price intervals
    max_price = data["price_close"].max()
    bucket_ranges = pd.interval_range(
        start=0,
        end=(max_price // bucket_size + 1) * bucket_size,
        freq=bucket_size,
        closed="left",
    )

    # Assign each price to a bucket
    data["PriceBucket"] = pd.cut(data["price_close"], bins=bucket_ranges)

    # Ensure we only count valid price buckets
    bucket_days_count = data["PriceBucket"].value_counts().sort_index()

    # Get the current price and its bucket
    current_price = data["price_close"].iloc[-1]
    current_bucket = pd.cut([current_price], bins=bucket_ranges)[0]

    # Extract the count of unique days for the current price bucket
    current_bucket_days = bucket_days_count.get(current_bucket, 0)

    # Create a DataFrame for bucket counts with formatted price ranges
    bucket_counts_df = bucket_days_count.reset_index()
    bucket_counts_df.columns = ["Price Range ($)", "Count"]
    bucket_counts_df["Price Range ($)"] = bucket_counts_df["Price Range ($)"].apply(
        lambda x: f"${int(x.left / 1000)}K-${int(x.right / 1000)}K"
    )

    return bucket_counts_df


def calculate_roi_table(data, report_date, price_column="price_close"):
    """
    Calculates the return on investment (ROI) for Bitcoin over various time frames from the report date.

    Parameters:
    data (pd.DataFrame): DataFrame containing price data with a DateTime index.
    report_date (str or datetime): The date for which to calculate ROI.
    price_column (str): The column name for Bitcoin price data.

    Returns:
    pd.DataFrame: DataFrame containing ROI, Start Date, and BTC Price for each time frame.
    """
    if price_column not in data.columns:
        raise ValueError(
            f"The price column '{price_column}' does not exist in the data."
        )

    if data.empty:
        raise ValueError("The input data is empty.")

    periods = {
        "1 day": 1,
        "3 day": 3,
        "7 day": 7,
        "30 day": 30,
        "90 day": 90,
        "1 Year": 365,
        "2 Year": 730,
        "4 Year": 1460,
        "5 Year": 1825,
        "10 Year": 3650,
    }

    today = pd.to_datetime(report_date).normalize()

    # Pre-compute the 'Start Date' and 'BTC Price' for each period
    start_dates = {
        period: today - pd.Timedelta(days=days) for period, days in periods.items()
    }
    btc_prices = {
        period: data.loc[start_dates[period], price_column]
        if start_dates[period] in data.index
        else None
        for period in periods
    }

    roi_data = {
        period: data[price_column].pct_change(days).iloc[-1] * 100
        for period, days in periods.items()
    }

    # Combine the ROI, Start Dates, and BTC Prices into a DataFrame
    roi_table = pd.DataFrame(
        {
            "Time Frame": periods.keys(),
            "ROI": roi_data.values(),
            "Start Date": start_dates.values(),
            "BTC Price": btc_prices.values(),
        }
    )
    return roi_table.set_index("Time Frame")


def create_fundamentals_table(df, metrics_template):
    """
    Generates a fundamentals metrics table with Monday - Sunday columns.

    This function creates a weekly breakdown of Bitcoin network fundamentals organized by
    section headers (Network Performance, Network Security, Network Economics, Network Valuation).
    Each metric shows the 7-day average, week-over-week change, and daily values for the current week.

    Parameters:
    df (pd.DataFrame): DataFrame with DatetimeIndex containing all columns specified in metrics_template.
                       Must have at least 14 days of data for week-over-week calculations.
    metrics_template (dict): Nested dictionary structure from data_definitions.py with format:
                             {section_name: {metric_label: (column_name, format_type)}}
                             Example: {"Network Performance": {"Hash Rate": ("hash_rate", "number")}}

    Returns:
    pd.DataFrame: Table with columns:
        - Metric: Section headers and metric names
        - 7 Day Avg: Mean value for the past 7 days
        - 7 Day Avg % Change: Week-over-week percentage change
        - Monday through Sunday: Individual daily values for current week
    """
    table_data = []

    latest_date = df.index.max()
    start_of_week = latest_date - timedelta(days=latest_date.weekday())
    weekly_index = pd.date_range(start=start_of_week.normalize(), periods=7, freq="D")

    for section, metrics in metrics_template.items():
        table_data.append(
            {
                "Metric": section,
                "7 Day Avg": None,
                "7 Day Avg % Change": None,
                "Monday": None,
                "Tuesday": None,
                "Wednesday": None,
                "Thursday": None,
                "Friday": None,
                "Saturday": None,
                "Sunday": None,
            }
        )

        for metric_display_name, (column_name, _) in metrics.items():
            week_avg = df[column_name].tail(7).mean()
            prev_week_avg = df[column_name].tail(14).head(7).mean()
            pct_change = (
                ((week_avg - prev_week_avg) / prev_week_avg) * 100
                if prev_week_avg != 0
                else np.nan
            )

            weekly_values = df[column_name].reindex(weekly_index).tolist()

            table_data.append(
                {
                    "Metric": metric_display_name,
                    "7 Day Avg": week_avg,
                    "7 Day Avg % Change": pct_change,
                    "Monday": weekly_values[0],
                    "Tuesday": weekly_values[1],
                    "Wednesday": weekly_values[2],
                    "Thursday": weekly_values[3],
                    "Friday": weekly_values[4],
                    "Saturday": weekly_values[5],
                    "Sunday": weekly_values[6],
                }
            )

    table_df = pd.DataFrame(table_data)

    return table_df


## Summary and Performance Tables


def create_summary_table(report_data, report_date):
    """
    Generates a summary table for Bitcoin's key metrics with categorized column headers.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical Bitcoin data, indexed by date.
    - report_date (str or pd.Timestamp): Specific date for which the summary is generated.

    Returns:
    - pd.DataFrame: DataFrame containing a summary of Bitcoin metrics for the specified report date.
    """

    # Extract key metrics from report_data
    price_usd = report_data.loc[report_date, "price_close"]
    market_cap = report_data.loc[report_date, "market_cap"]
    sats_per_dollar = SATS_PER_BTC / price_usd

    bitcoin_supply = report_data.loc[report_date, "supply_btc"]
    miner_revenue_30d = report_data.loc[report_date, "30_day_ma_coinbase_usd_sum"]
    tx_volume_30d = report_data.loc[report_date, "30_day_ma_sent_usd"]
    btc_dominance = report_data.loc[report_date, "bitcoin_dominance"]

    # Placeholder for additional derived metrics
    fear_greed = "Neutral"
    bitcoin_valuation = "Fair Value"

    # Define categories for organization
    categorized_data = {
        "Market Data": {
            "Bitcoin Price USD": price_usd,
            "Bitcoin Marketcap": market_cap,
            "Sats Per Dollar": sats_per_dollar,
        },
        "On-chain Data": {
            "Bitcoin Supply": bitcoin_supply,
            "Bitcoin Miner Revenue": miner_revenue_30d,
            "Bitcoin Transaction Volume": tx_volume_30d,
        },
        "Investor Sentiment": {
            "Bitcoin Dominance": btc_dominance,
            "Bitcoin Market Sentiment": fear_greed,
            "Bitcoin Valuation": bitcoin_valuation,
        },
    }

    # Convert the dictionary into a structured DataFrame
    weekly_summary_df = pd.DataFrame.from_dict(
        {k: v for d in categorized_data.values() for k, v in d.items()},
        orient="index",
        columns=["Value"],
    )

    # Add a category column
    weekly_summary_df["Category"] = [
        category for category, metrics in categorized_data.items() for _ in metrics
    ]

    return weekly_summary_df


def format_value(value, format_type):
    """
    Helper function to format a value based on a specified type.

    Parameters:
    - value: The value to format. Can be an integer, float, or date object, depending on format_type.
    - format_type (str): The format type to apply to the value. Accepted values are:
        - "percentage": Formats the value as a percentage with two decimal places.
        - "integer": Formats the value as an integer with thousand separators.
        - "float": Formats the value as a float with thousand separators and no decimal places.
        - "currency": Formats the value as currency (e.g., $100,000) with thousand separators.
        - "date": Formats the value as a date in the format "YYYY-MM-DD".
        - Defaults to string formatting if the format_type is unrecognized.

    Returns:
    - str: The formatted value as a string.
    """

    # Format the value based on the specified format_type
    if format_type == "percentage":
        return f"{value:.2f}%"  # Format as a percentage with 2 decimal places

    elif format_type == "integer":
        return f"{int(value):,}"  # Format as an integer with thousand separators

    elif format_type == "float":
        return (
            f"{value:,.0f}"  # Format as a float with thousand separators, no decimals
        )

    elif format_type == "currency":
        return f"${value:,.0f}"  # Format as currency with thousand separators

    elif format_type == "date":
        return value.strftime("%Y-%m-%d")  # Format date as YYYY-MM-DD

    else:
        # Default: return as a string if format_type is unrecognized
        return str(value)


def _build_performance_table(
    report_data: pd.DataFrame,
    report_date,
    correlation_results: dict,
    asset_configs: list
) -> pd.DataFrame:
    """
    Generic performance table builder for any asset category.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for all assets.
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (dict): Dictionary with correlation DataFrames for different periods.
    - asset_configs (list): List of dicts with 'name', 'label', 'ticker' keys.
                            Example: [{"name": "BTC", "label": "Bitcoin - [BTC]", "ticker": "price_close"}]

    Returns:
    - pd.DataFrame: Performance metrics for the specified assets.
    """
    performance_metrics = []

    for config in asset_configs:
        ticker = config["ticker"]
        # Handle special case for Bitcoin price_close column
        price_col = ticker if ticker == "price_close" else f"{ticker}_close"
        corr_col = ticker if ticker == "price_close" else f"{ticker}_close"

        metrics = {
            "Asset": config["label"],
            "Price": report_data.loc[report_date, price_col],
            "7 Day Return": report_data.loc[report_date, f"{price_col}_7_change"],
            "MTD Return": report_data.loc[report_date, f"{price_col}_MTD_change"],
            "YTD Return": report_data.loc[report_date, f"{price_col}_YTD_change"],
            "90 Day Return": report_data.loc[report_date, f"{price_col}_90_change"],
            "90 Day BTC Correlation": correlation_results["price_close_90_days"].loc[
                "price_close", corr_col
            ] if ticker != "price_close" else 1,  # BTC correlation with itself is 1
        }
        performance_metrics.append(metrics)

    return pd.DataFrame(performance_metrics)


def create_equity_performance_table(report_data, report_date, correlation_results):
    """
    Creates a performance table summarizing key metrics for selected equity ETFs.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for the assets.
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (dict): Dictionary with correlation DataFrames for different periods.

    Returns:
    - pd.DataFrame: A DataFrame containing the performance metrics for the selected assets.
    """
    asset_configs = [
        {"name": "BTC", "label": "Bitcoin - [BTC]", "ticker": "price_close"},
        {"name": "SPY", "label": "S&P 500 Index ETF - [SPY]", "ticker": "SPY"},
        {"name": "QQQ", "label": "Nasdaq-100 ETF - [QQQ]", "ticker": "QQQ"},
        {"name": "VTI", "label": "US Total Stock Market ETF - [VTI]", "ticker": "VTI"},
        {"name": "VXUS", "label": "International Stock ETF - [VXUS]", "ticker": "VXUS"},
    ]
    return _build_performance_table(report_data, report_date, correlation_results, asset_configs)


def create_sector_performance_table(report_data, report_date, correlation_results):
    """
    Creates a sector performance table for selected sector ETFs.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for the assets.
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (dict): Dictionary with correlation DataFrames for different periods.

    Returns:
    - pd.DataFrame: A DataFrame containing the performance metrics for the selected sector ETFs.
    """
    asset_configs = [
        {"name": "BTC", "label": "Bitcoin - [BTC]", "ticker": "price_close"},
        {"name": "XLK", "label": "Technology Sector ETF - [XLK]", "ticker": "XLK"},
        {"name": "XLF", "label": "Financials Sector ETF - [XLF]", "ticker": "XLF"},
        {"name": "XLE", "label": "Energy Sector ETF - [XLE]", "ticker": "XLE"},
        {"name": "XLRE", "label": "Real Estate Sector ETF - [XLRE]", "ticker": "XLRE"},
    ]
    return _build_performance_table(report_data, report_date, correlation_results, asset_configs)


def create_macro_performance_table(
    report_data, report_date, correlation_results
):
    """
    Creates a macro performance table for macroeconomic indicators.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for the macro indicators.
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (pd.DataFrame): DataFrame with correlation values between macro indicators and Bitcoin.

    Returns:
    - pd.DataFrame: A DataFrame containing the performance metrics for the selected macro indicators.
    """
    asset_configs = [
        {"name": "BTC", "label": "Bitcoin - [BTC]", "ticker": "price_close"},
        {"name": "DXY", "label": "US Dollar Index - [DXY]", "ticker": "DX=F"},
        {"name": "GLD", "label": "Gold ETF - [GLD]", "ticker": "GLD"},
        {"name": "AGG", "label": "Aggregate Bond ETF - [AGG]", "ticker": "AGG"},
        {"name": "BCOM", "label": "Bloomberg Commodity Index - [BCOM]", "ticker": "^BCOM"},
    ]
    return _build_performance_table(report_data, report_date, correlation_results, asset_configs)


def create_bitcoin_performance_table(report_data, report_date, correlation_results):
    """
    Creates a Bitcoin performance table for Bitcoin-related equities.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for Bitcoin and equities.
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (pd.DataFrame): DataFrame with correlation values between Bitcoin and related equities.

    Returns:
    - pd.DataFrame: A DataFrame containing the performance metrics for Bitcoin and related equities.
    """
    asset_configs = [
        {"name": "BTC", "label": "Bitcoin - [BTC]", "ticker": "price_close"},
        {"name": "MSTR", "label": "MicroStrategy - [MSTR]", "ticker": "MSTR"},
        {"name": "XYZ", "label": "Block - [XYZ]", "ticker": "XYZ"},
        {"name": "COIN", "label": "Coinbase - [COIN]", "ticker": "COIN"},
        {"name": "WGMI", "label": "Bitcoin Miners ETF - [WGMI]", "ticker": "WGMI"},
    ]
    return _build_performance_table(report_data, report_date, correlation_results, asset_configs)


def create_full_performance_table(
    report_data,
    report_date,
    correlation_results,
):
    """
    Combines data from all performance tables into a single comprehensive table.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing report data for all assets.
    - report_date (str or pd.Timestamp): Date for which the report is generated.
    - correlation_results (dict): Dictionary of correlation DataFrames for each period (e.g., 90 days) with BTC as baseline.

    Returns:
    - pd.DataFrame: A comprehensive DataFrame summarizing performance metrics for all assets.
    """
    # Combine performance data from all existing tables
    all_performance_metrics = {}

    # Merge Equity Performance Table Data
    equity_data = create_equity_performance_table(
        report_data, report_date, correlation_results
    )
    for index, row in equity_data.iterrows():
        all_performance_metrics[row["Asset"]] = row.to_dict()

    # Merge Sector Performance Table Data
    sector_data = create_sector_performance_table(
        report_data, report_date, correlation_results
    )
    for index, row in sector_data.iterrows():
        all_performance_metrics[row["Asset"]] = row.to_dict()

    # Merge Macro Performance Table Data
    macro_data = create_macro_performance_table(
        report_data, report_date, correlation_results
    )
    for index, row in macro_data.iterrows():
        all_performance_metrics[row["Asset"]] = row.to_dict()

    # Merge Bitcoin Performance Table Data
    bitcoin_data = create_bitcoin_performance_table(
        report_data, report_date, correlation_results
    )
    for index, row in bitcoin_data.iterrows():
        all_performance_metrics[row["Asset"]] = row.to_dict()

    # Create the final combined DataFrame
    full_performance_df = pd.DataFrame(all_performance_metrics.values())

    return full_performance_df


def monthly_heatmap(data, export_csv=True):
    """
    Creates monthly and yearly Bitcoin returns heatmap data with statistical aggregations.

    This function generates a matrix of monthly returns organized by year (rows) and month (columns),
    with an additional yearly return column. It includes statistical rows (4-year average, median,
    average) and handles incomplete current month data by calculating month-to-date returns.

    Parameters:
    data (pd.DataFrame): DataFrame with DatetimeIndex and 'price_close' column. Data is filtered
                         to start from 2012-01-01 within the function.
    export_csv (bool): If True, exports heatmap data to csv/monthly_heatmap_data.csv. Default: True.

    Returns:
    pd.DataFrame: Heatmap matrix with:
        - Rows: Years (2012+), plus "4-Year Average", "Median", "Average"
        - Columns: Month names (Jan-Dec) plus "Yearly"
        - Values: Decimal return values (0.05 = 5% gain, -0.03 = 3% loss)
        - Current incomplete month shows MTD return
        - Statistical rows exclude incomplete month data
    """
    # Filter data to start from January 2011
    data = data[data.index >= pd.to_datetime("2012-01-01")]

    # Calculate monthly returns
    monthly_returns = data["price_close"].resample("M").last().pct_change()

    # Calculate YTD returns based on the first price of the year
    start_of_year = data["price_close"].groupby(data.index.year).transform("first")
    ytd_returns = (data["price_close"] / start_of_year) - 1

    # Aggregate YTD returns by year
    yearly_returns = ytd_returns.groupby(data.index.year).last()

    # Prepare the data for the heatmap: years as rows, months as columns
    heatmap_data = (
        monthly_returns.groupby(
            [monthly_returns.index.year, monthly_returns.index.month]
        )
        .mean()
        .unstack()
    )

    # Add the YTD returns as a separate column
    heatmap_data[13] = yearly_returns

    # Get the last date in the data to check if the current month is complete
    last_date = data.index[-1]
    current_year, current_month = last_date.year, last_date.month

    # Check if the current month is incomplete
    is_incomplete_month = last_date.day != (last_date + MonthEnd(0)).day

    if is_incomplete_month:
        # Get the first price of the current month
        start_of_month = data["price_close"].loc[last_date.strftime("%Y-%m")].iloc[0]
        # Calculate the MTD return for the incomplete month
        current_month_return = (data["price_close"].iloc[-1] / start_of_month) - 1
        # Add the MTD return to the heatmap for display
        if current_year in heatmap_data.index:
            heatmap_data.loc[current_year, current_month] = current_month_return

    # Create a copy of the data excluding the incomplete month for additional calculations
    heatmap_data_excluded = heatmap_data.copy()
    if is_incomplete_month and current_year in heatmap_data.index:
        heatmap_data_excluded.loc[current_year, current_month] = pd.NA

    # Add the "4-Year Average" row (last 4 years for each month)
    heatmap_data.loc["4-Year Average"] = heatmap_data_excluded.iloc[-4:].apply(
        lambda col: col[~col.isna()].mean(), axis=0
    )

    # Add the "Median" row, excluding the incomplete month
    heatmap_data.loc["Median"] = heatmap_data_excluded.apply(
        lambda col: col[~col.isna()].median(), axis=0
    )

    # Add the "Average" row, excluding the incomplete month
    heatmap_data.loc["Average"] = heatmap_data_excluded.apply(
        lambda col: col[~col.isna()].mean(), axis=0
    )

    # Rename columns to month names
    month_names = [calendar.month_abbr[i] for i in range(1, 13)] + ["Yearly"]
    heatmap_data.columns = month_names

    # Optionally export the heatmap data to CSV
    if export_csv:
        heatmap_data.to_csv("csv/monthly_heatmap_data.csv")

    return heatmap_data


## CSV Exports


def calculate_ohlc(ohlc_data, output_file="csv/ohlc_data.csv"):
    """
    Resamples daily OHLC data to weekly intervals and saves full history to CSV.

    This function converts daily Bitcoin price data into weekly candlestick data by taking
    the first open, highest high, lowest low, and last close for each week. Weeks are
    defined as Sunday-Saturday periods. NaN rows are dropped before export.

    Parameters:
    ohlc_data (pd.DataFrame): DataFrame with DatetimeIndex and columns: 'Open', 'High', 'Low', 'Close'.
                              Index must be datetime-compatible for resampling.
    output_file (str): Path for CSV export. Default: "csv/ohlc_data.csv"

    Returns:
    pd.DataFrame: Weekly OHLC DataFrame with columns:
        - Open: First open price of the week
        - High: Highest price during the week
        - Low: Lowest price during the week
        - Close: Last close price of the week
        Index is weekly period end dates (Sundays).
    """
    # Ensure the index is a datetime index for resampling
    ohlc_data.index = pd.to_datetime(ohlc_data.index)

    # Resample daily data to get weekly OHLC values
    resampled_ohlc = ohlc_data.resample("W").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )

    # Drop any rows with NaN values (incomplete weeks)
    resampled_ohlc = resampled_ohlc.dropna()

    # Export full history to CSV
    resampled_ohlc.to_csv(output_file)

    return resampled_ohlc


def create_eoy_model_table(report_data, cagr_results):
    """
    Generates end-of-year price model projection table combining valuation metrics and CAGR data.

    This function creates a dataset used for projecting Bitcoin's end-of-year price based on
    multiple valuation models (Realized Price, Thermocap, 200-day MA, Energy Value) and their
    historical 4-year compound annual growth rates. The output is used for price modeling and
    forecasting analysis.

    Parameters:
    report_data (pd.DataFrame): DataFrame with DatetimeIndex containing Bitcoin valuation metrics:
                                price_close, realised_price, thermocap_price, 200_day_ma_price_close,
                                Lagged_Energy_Value, mvrv_ratio, thermocap_multiple, 200_day_multiple,
                                Energy_Value_Multiple.
    cagr_results (pd.DataFrame): DataFrame with DatetimeIndex containing 4-year CAGR calculations for
                                 the valuation models (output from calculate_rolling_cagr_for_all_metrics).

    Returns:
    pd.DataFrame: Combined DataFrame with columns:
        - Current values: price_close, realised_price, thermocap_price, 200_day_ma_price_close,
                         Lagged_Energy_Value
        - Multiples: mvrv_ratio, thermocap_multiple, 200_day_multiple, Energy_Value_Multiple
        - Growth rates: *_4_Year_CAGR for each valuation model
        Merged on DatetimeIndex with left join (preserves all report_data dates).
    """
    # Define the columns to extract from report_data
    columns_of_interest = [
        "price_close",
        "realised_price",
        "thermocap_price",
        "200_day_ma_price_close",
        "Lagged_Energy_Value",
        "mvrv_ratio",
        "thermocap_multiple",
        "200_day_multiple",
        "Energy_Value_Multiple",
    ]

    # Define the CAGR columns to extract from cagr_results
    cagr_columns = [
        "price_close_4_Year_CAGR",
        "realised_price_4_Year_CAGR",
        "thermocap_price_4_Year_CAGR",
        "200_day_ma_price_close_4_Year_CAGR",
        "Lagged_Energy_Value_4_Year_CAGR",
    ]

    # Ensure the specified columns exist in report_data before extracting
    available_columns = [
        col for col in columns_of_interest if col in report_data.columns
    ]
    available_cagr_columns = [
        col for col in cagr_columns if col in cagr_results.columns
    ]

    # Extract the relevant data from both datasets
    report_data_filtered = report_data[available_columns]
    cagr_results_filtered = cagr_results[available_cagr_columns]

    # Merge both datasets on the index (assuming they share the same date index)
    full_data = report_data_filtered.merge(
        cagr_results_filtered, left_index=True, right_index=True, how="left"
    )

    return full_data


def create_monthly_returns_table(selected_metrics):
    """
    Generates a month-to-date (MTD) return comparison table indexed to the current month.

    This function compares Bitcoin's performance for the current month across all historical
    years where data is available. It calculates returns from the start of the current month
    to both the current date and the end of month, providing historical context for current
    performance.

    Parameters:
    selected_metrics (pd.DataFrame): DataFrame with DatetimeIndex containing at minimum
                                      a 'price_close' column. Data filtered to 2014-01-01+.

    Returns:
    pd.DataFrame: Table with columns:
        - Year: The calendar year
        - Start Price ($): Price at start of current month for that year
        - End Price ($): Price at end of current month for that year
        - Return (%): Full month return percentage
        - Report Date Return (%): Return from month start to current date
        Final rows include current year data and median historical projection.
    """
    today = datetime.today().date()
    current_year = today.year
    current_month = today.month
    current_day = today.day

    # Ensure data is filtered to entries from January 1, 2014, onwards
    selected_metrics = selected_metrics[selected_metrics.index >= "2014-01-01"].copy()

    monthly_returns = {}
    report_date_returns = {}

    # Get the starting price for the current month of the current year
    current_month_data = selected_metrics[
        (selected_metrics.index.year == current_year)
        & (selected_metrics.index.month == current_month)
    ]
    if current_month_data.empty:
        return None  # No data for current month

    current_start_price = current_month_data["price_close"].iloc[0]

    # Calculate monthly returns for each year
    for year in selected_metrics.index.year.unique():
        monthly_data = selected_metrics[
            (selected_metrics.index.year == year)
            & (selected_metrics.index.month == current_month)
        ]

        if not monthly_data.empty:
            start_price = monthly_data["price_close"].iloc[0]
            end_price = monthly_data["price_close"].iloc[-1]
            return_pct = (end_price / start_price - 1) * 100
            monthly_returns[year] = (start_price, end_price, return_pct)

            # Report Date Return Calculation
            report_date_data = monthly_data[(monthly_data.index.day == current_day)]
            if not report_date_data.empty:
                report_date_price = report_date_data["price_close"].iloc[-1]
                report_date_return = (report_date_price / start_price - 1) * 100
                report_date_returns[year] = report_date_return
            else:
                report_date_returns[year] = None

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(
        monthly_returns,
        orient="index",
        columns=["Start Price ($)", "End Price ($)", "Return (%)"],
    )
    df.index.name = "Year"

    # Add report date return column
    df["Report Date Return (%)"] = pd.Series(report_date_returns)

    # Extract the current year's data
    current_year_row = df.loc[[current_year]].reset_index()

    # Calculate the historical median return
    median_return = df["Return (%)"].median()
    median_end_price = current_start_price * (1 + median_return / 100)

    # Calculate the median return at the current date (not full period)
    median_report_date_return = df["Report Date Return (%)"].median()

    # Create the projected median row
    median_row = pd.DataFrame(
        {
            "Year": ["Median Projection"],
            "Start Price ($)": [current_start_price],
            "End Price ($)": [median_end_price],
            "Return (%)": [median_return],
            "Report Date Return (%)": [median_report_date_return],
        }
    )

    # Concatenate current year and median projection rows
    df_filtered = pd.concat([current_year_row, median_row], ignore_index=True)

    return df_filtered


def create_yearly_returns_table(selected_metrics):
    """
    Generates a year-to-date (YTD) return comparison table indexed to the current day of year.

    This function compares Bitcoin's performance for the current year across all historical
    years where data is available. It calculates returns from January 1st to both the current
    day of year and year-end, providing historical context for current YTD performance.

    Parameters:
    selected_metrics (pd.DataFrame): DataFrame with DatetimeIndex containing at minimum
                                      a 'price_close' column. Data filtered to 2014-01-01+.

    Returns:
    pd.DataFrame: Table with columns:
        - Year: The calendar year
        - Start Price ($): Price on January 1st for that year
        - End Price ($): Price on December 31st for that year
        - Return (%): Full year return percentage
        - Report Date Return (%): Return from January 1st to current day of year
        Final rows include current year data and median historical projection.
    """
    today = datetime.today().date()
    current_year = today.year
    current_day_of_year = today.timetuple().tm_yday

    # Ensure data is filtered correctly
    selected_metrics = selected_metrics[selected_metrics.index >= "2014-01-01"].copy()

    yearly_returns = {}
    report_date_returns = {}

    # Get the starting price for the current year
    current_year_data = selected_metrics[selected_metrics.index.year == current_year]
    if current_year_data.empty:
        return None  # No data for current year

    current_start_price = current_year_data["price_close"].iloc[0]

    # Calculate yearly returns for each year
    for year in selected_metrics.index.year.unique():
        yearly_data = selected_metrics[selected_metrics.index.year == year]

        if not yearly_data.empty:
            start_price = yearly_data["price_close"].iloc[0]
            end_price = yearly_data["price_close"].iloc[-1]
            return_pct = (end_price / start_price - 1) * 100
            yearly_returns[year] = (start_price, end_price, return_pct)

            # Report Date Return Calculation
            report_date_data = yearly_data[
                yearly_data.index.dayofyear == current_day_of_year
            ]
            if not report_date_data.empty:
                report_date_price = report_date_data["price_close"].iloc[-1]
                report_date_return = (report_date_price / start_price - 1) * 100
                report_date_returns[year] = report_date_return
            else:
                report_date_returns[year] = None

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(
        yearly_returns,
        orient="index",
        columns=["Start Price ($)", "End Price ($)", "Return (%)"],
    )
    df.index.name = "Year"

    # Add report date return column
    df["Report Date Return (%)"] = pd.Series(report_date_returns)

    # Extract the current year's data
    current_year_row = df.loc[[current_year]].reset_index()

    # Calculate the historical median return
    median_return = df["Return (%)"].median()
    median_end_price = current_start_price * (1 + median_return / 100)

    # **Fix the Median Report Date Return (%)**
    median_report_date_return_pct = df["Report Date Return (%)"].dropna().median()

    # Create the projected median row
    median_row = pd.DataFrame(
        {
            "Year": ["Median Projection"],
            "Start Price ($)": [current_start_price],
            "End Price ($)": [median_end_price],
            "Return (%)": [median_return],
            "Report Date Return (%)": [median_report_date_return_pct],
        }
    )

    # Concatenate current year and median projection rows
    df_filtered = pd.concat([current_year_row, median_row], ignore_index=True)

    return df_filtered


def create_asset_valuation_table(report_data):
    """
    Generates relative valuation comparison table showing Bitcoin price if it matched other asset market caps.

    This function calculates what Bitcoin's price would be if its market cap equaled various
    benchmark assets including stocks (AAPL, NVDA, META, AMZN), precious metals (gold, silver),
    and fiat money supplies (US M0, UK M0, gold reserves). It shows the percentage move required
    for Bitcoin to reach each valuation milestone.

    Parameters:
    report_data (pd.DataFrame): DataFrame with latest row containing:
        - price_close: Current Bitcoin price
        - market_cap: Current Bitcoin market cap
        - *_mc_btc_price: Calculated BTC price if matching each asset's market cap
        - *_MarketCap or *_cap: Market cap values for comparison assets (in USD or billions)
        - gold/silver market cap columns with billion_usd suffixes

    Returns:
    pd.DataFrame: Table with columns:
        - Asset: Name of comparison asset (Bitcoin, stocks, gold, fiat currencies)
        - Market Cap (USD): Current market cap of the asset in USD
        - Market Cap BTC Price: What Bitcoin price would be at that market cap
        - BTC % Move to Marketcap BTC Price: Percentage gain/loss needed to reach target
        Values formatted as strings with $ and % symbols, "N/A" for missing data.
    """
    assets = [
        {"name": "Bitcoin", "data": "price_close", "marketcap": "market_cap"},
        {
            "name": "Total Silver Market",
            "data": "silver_marketcap_btc_price",
            "marketcap": "silver_marketcap_billion_usd",
        },
        {
            "name": "UK M0",
            "data": "United_Kingdom_btc_price",
            "marketcap": "United_Kingdom_cap",
        },
        {"name": "Meta", "data": "META_mc_btc_price", "marketcap": "META_MarketCap"},
        {"name": "Amazon", "data": "AMZN_mc_btc_price", "marketcap": "AMZN_MarketCap"},
        {
            "name": "Gold Country Holdings",
            "data": "gold_official_country_holdings_marketcap_btc_price",
            "marketcap": "gold_marketcap_official_country_holdings_billion_usd",
        },
        {"name": "NVIDIA", "data": "NVDA_mc_btc_price", "marketcap": "NVDA_MarketCap"},
        {
            "name": "Gold Private Investment",
            "data": "gold_private_investment_marketcap_btc_price",
            "marketcap": "gold_marketcap_private_investment_billion_usd",
        },
        {"name": "Apple", "data": "AAPL_mc_btc_price", "marketcap": "AAPL_MarketCap"},
        {
            "name": "US M0",
            "data": "United_States_btc_price",
            "marketcap": "United_States_cap",
        },
        {
            "name": "Total Gold Market",
            "data": "gold_marketcap_btc_price",
            "marketcap": "gold_marketcap_billion_usd",
        },
    ]

    # Get the latest values (last row)
    latest_data = report_data.iloc[-1]
    bitcoin_price = latest_data.get("price_close", float("nan"))

    valuation_data = []
    for asset in assets:
        marketcap_btc_price = latest_data.get(asset["data"], float("nan"))
        marketcap_value = latest_data.get(asset["marketcap"], float("nan"))

        # Avoid division by zero or invalid values
        if (
            pd.notna(bitcoin_price)
            and pd.notna(marketcap_btc_price)
            and bitcoin_price > 0
        ):
            percent_move = ((marketcap_btc_price - bitcoin_price) / bitcoin_price) * 100
        else:
            percent_move = "N/A"

        valuation_data.append(
            {
                "Asset": asset["name"],
                "Market Cap (USD)": f"${marketcap_value:,.0f}"
                if pd.notna(marketcap_value)
                else "N/A",
                "Market Cap BTC Price": f"${marketcap_btc_price:,.0f}"
                if pd.notna(marketcap_btc_price)
                else "N/A",
                "BTC % Move to Marketcap BTC Price": f"{percent_move:.0f}%"
                if percent_move != "N/A"
                else "N/A",
            }
        )

    valuation_df = pd.DataFrame(valuation_data)

    return valuation_df
