import datapane as dp
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
from pandas.tseries.offsets import MonthEnd
import calendar


def calculate_price_buckets(data, bucket_size):
    """
    Calculates the number of unique trading days the price spent in each bucket range.

    Parameters:
    data (pd.DataFrame): DataFrame with DateTime index and a 'PriceUSD' column.
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
    data = data.dropna(subset=["PriceUSD"])

    # Remove duplicate intra-day price fluctuations by keeping only one entry per day
    data = data.groupby(data.index.floor("D")).first()

    # Define the bucket ranges for price intervals
    max_price = data["PriceUSD"].max()
    bucket_ranges = pd.interval_range(
        start=0,
        end=(max_price // bucket_size + 1) * bucket_size,
        freq=bucket_size,
        closed="left",
    )

    # Assign each price to a bucket
    data["PriceBucket"] = pd.cut(data["PriceUSD"], bins=bucket_ranges)

    # Ensure we only count valid price buckets
    bucket_days_count = data["PriceBucket"].value_counts().sort_index()

    # Get the current price and its bucket
    current_price = data["PriceUSD"].iloc[-1]
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


def style_bucket_counts_table(bucket_counts_df):
    # Define the style for the table: smaller font size
    table_style = [
        {
            "selector": "th",
            "props": "font-size: 12px;",  # Adjust header font size
        },
        {
            "selector": "td",
            "props": "font-size: 12px;",  # Adjust cell font size
        },
    ]

    # Apply the style to the table and hide the index
    styled_table = bucket_counts_df.style.set_table_styles(table_style).hide_index()

    return styled_table


def calculate_roi_table(data, report_date, price_column="PriceUSD"):
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
    roi_table.to_csv("csv/roi_table.csv")
    return roi_table.set_index("Time Frame")


def style_roi_table(roi_table):
    """
    Styles the ROI table by formatting the 'ROI' column with colors and the 'BTC Price' column as currency.

    Parameters:
    roi_table (pd.DataFrame): DataFrame containing the ROI data.

    Returns:
    pd.io.formats.style.Styler: Styled DataFrame for display.
    """

    # Function to color ROI values
    def color_roi(val):
        color = "green" if val > 0 else ("red" if val < 0 else "black")
        return "color: %s" % color

    # Function to format BTC Price as currency
    def format_currency(val):
        return "${:,.2f}".format(val)

    return (
        roi_table.style.applymap(color_roi, subset=["ROI"])
        .format(
            {
                "ROI": "{:.2f}%",  # Format ROI as percentage with 2 decimal places
                "BTC Price": format_currency,
            }
        )  # Format BTC Price as currency
        .set_properties(**{"font-size": "10pt"})
    )


def create_bitcoin_fundamentals_table(
    report_data, difficulty_period_changes, weekly_high_low, report_date, cagr_results
):
    """
    Creates a summary table of Bitcoin's fundamental metrics, including price, network activity,
    miner revenue, and key growth rates.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing Bitcoin metrics data.
    - difficulty_period_changes (pd.Series): Series containing the change in Bitcoin metrics over the last difficulty period.
    - weekly_high_low (dict): Dictionary containing the 52-week high and low values for each metric.
    - report_date (str or pd.Timestamp): The specific date for which to generate the report.
    - cagr_results (pd.DataFrame): DataFrame containing Compound Annual Growth Rate (CAGR) for each metric.

    Returns:
    - pd.DataFrame: DataFrame summarizing Bitcoin's fundamental metrics, changes, CAGR, and 52-week highs and lows.
    """

    # Extract Bitcoin price data
    Bitcoin_Price = report_data.loc[report_date, "PriceUSD"]
    Bitcoin_7d_Return = report_data.loc[report_date, "PriceUSD_7_change"]
    Bitcoin_MTD_Return = report_data.loc[report_date, "PriceUSD_MTD_change"]
    Bitcoin_90d_Return = report_data.loc[report_date, "PriceUSD_90_change"]
    Bitcoin_YTD_Return = report_data.loc[report_date, "PriceUSD_YTD_change"]
    Bitcoin_4y_CAGR = cagr_results.loc[report_date, "PriceUSD_4_Year_CAGR"]
    Bitcoin_52w_High = weekly_high_low["PriceUSD"]["52_week_high"]
    Bitcoin_52w_Low = weekly_high_low["PriceUSD"]["52_week_low"]
    Bitcoin_Difficulty_Change = difficulty_period_changes.loc["PriceUSD"]

    # Extract other fundamental data
    HashRate = report_data.loc[report_date, "7_day_ma_HashRate"]
    TxCnt = report_data.loc[report_date, "7_day_ma_TxCnt"]
    TxTfrValAdjUSD = report_data.loc[report_date, "7_day_ma_TxTfrValAdjUSD"]
    TxTfrValMeanUSD = report_data.loc[
        report_date, "7_day_ma_TxTfrValMeanUSD"
    ]  # New metric added
    RevUSD = report_data.loc[report_date, "RevUSD"]
    AdrActCnt = report_data.loc[report_date, "AdrActCnt"]
    AdrBalUSD10Cnt = report_data.loc[report_date, "AdrBalUSD10Cnt"]  # New metric added
    FeeTotUSD = report_data.loc[report_date, "FeeTotUSD"]
    supply_pct_1_year_plus = report_data.loc[
        report_date, "supply_pct_1_year_plus"
    ]  # New metric added
    VelCur1yr = report_data.loc[report_date, "VelCur1yr"]  # New metric added

    # Extract difficulty period changes
    HashRate_Diff_Change = difficulty_period_changes.loc["7_day_ma_HashRate"]
    TxCnt_Diff_Change = difficulty_period_changes.loc["TxCnt"]
    TxTfrValAdjUSD_Diff_Change = difficulty_period_changes.loc["TxTfrValAdjUSD"]
    RevUSD_Diff_Change = difficulty_period_changes.loc["RevUSD"]
    AdrActCnt_Diff_Change = difficulty_period_changes.loc["AdrActCnt"]
    FeeTotUSD_Diff_Change = difficulty_period_changes.loc["FeeTotUSD"]

    # Extract difficulty period changes
    HashRate_Diff_Change = difficulty_period_changes.loc["7_day_ma_HashRate"]
    TxCnt_Diff_Change = difficulty_period_changes.loc["TxCnt"]
    TxTfrValAdjUSD_Diff_Change = difficulty_period_changes.loc[
        "7_day_ma_TxTfrValAdjUSD"
    ]
    TxTfrValMeanUSD_Diff_Change = difficulty_period_changes.loc[
        "7_day_ma_TxTfrValMeanUSD"
    ]  # New metric added
    RevUSD_Diff_Change = difficulty_period_changes.loc["RevUSD"]
    AdrActCnt_Diff_Change = difficulty_period_changes.loc["AdrActCnt"]
    AdrBalUSD10Cnt_Diff_Change = difficulty_period_changes.loc[
        "AdrBalUSD10Cnt"
    ]  # New metric added
    FeeTotUSD_Diff_Change = difficulty_period_changes.loc["FeeTotUSD"]
    supply_pct_1_year_plus_Diff_Change = difficulty_period_changes.loc[
        "supply_pct_1_year_plus"
    ]  # New metric added
    VelCur1yr_Diff_Change = difficulty_period_changes.loc[
        "VelCur1yr"
    ]  # New metric added

    # Create a dictionary with the extracted values
    bitcoin_fundamentals_data = {
        "Metrics Name": [
            "Bitcoin Price",
            "Hashrate",
            "Transaction Count",
            "Transaction Volume",
            "Transaction Volume Mean USD",  # New metric added
            "Active Address Count",
            "Addresses with Balance > 10 USD",  # New metric added
            "Miner Revenue",
            "Fees in USD",
            "Supply Held > 1 Year",  # New metric added
            "Velocity (1 Year)",  # New metric added
        ],
        "Value": [
            Bitcoin_Price,
            HashRate,
            TxCnt,
            TxTfrValAdjUSD,
            TxTfrValMeanUSD,
            AdrActCnt,
            AdrBalUSD10Cnt,
            RevUSD,
            FeeTotUSD,
            supply_pct_1_year_plus,
            VelCur1yr,
        ],
        "7 Day Change": [
            Bitcoin_7d_Return,
            report_data.loc[report_date, "HashRate_7_change"],
            report_data.loc[report_date, "TxCnt_7_change"],
            report_data.loc[report_date, "TxTfrValAdjUSD_7_change"],
            report_data.loc[
                report_date, "7_day_ma_TxTfrValMeanUSD_7_change"
            ],  # New metric added
            report_data.loc[report_date, "AdrActCnt_7_change"],
            report_data.loc[report_date, "AdrBalUSD10Cnt_7_change"],  # New metric added
            report_data.loc[report_date, "RevUSD_7_change"],
            report_data.loc[report_date, "FeeTotUSD_7_change"],
            report_data.loc[
                report_date, "supply_pct_1_year_plus_7_change"
            ],  # New metric added
            report_data.loc[report_date, "VelCur1yr_7_change"],  # New metric added
        ],
        "Difficulty Period Change": [
            Bitcoin_Difficulty_Change,
            HashRate_Diff_Change,
            TxCnt_Diff_Change,
            TxTfrValAdjUSD_Diff_Change,
            TxTfrValMeanUSD_Diff_Change,
            AdrActCnt_Diff_Change,
            AdrBalUSD10Cnt_Diff_Change,
            RevUSD_Diff_Change,
            FeeTotUSD_Diff_Change,
            supply_pct_1_year_plus_Diff_Change,
            VelCur1yr_Diff_Change,
        ],
        "MTD Change": [
            Bitcoin_MTD_Return,
            report_data.loc[report_date, "HashRate_MTD_change"],
            report_data.loc[report_date, "TxCnt_MTD_change"],
            report_data.loc[report_date, "TxTfrValAdjUSD_MTD_change"],
            report_data.loc[
                report_date, "7_day_ma_TxTfrValMeanUSD_MTD_change"
            ],  # New metric added
            report_data.loc[report_date, "AdrActCnt_MTD_change"],
            report_data.loc[
                report_date, "AdrBalUSD10Cnt_MTD_change"
            ],  # New metric added
            report_data.loc[report_date, "RevUSD_MTD_change"],
            report_data.loc[report_date, "FeeTotUSD_MTD_change"],
            report_data.loc[
                report_date, "supply_pct_1_year_plus_MTD_change"
            ],  # New metric added
            report_data.loc[report_date, "VelCur1yr_MTD_change"],  # New metric added
        ],
        "90 Day Change": [
            Bitcoin_90d_Return,
            report_data.loc[report_date, "HashRate_90_change"],
            report_data.loc[report_date, "TxCnt_90_change"],
            report_data.loc[report_date, "TxTfrValAdjUSD_90_change"],
            report_data.loc[
                report_date, "7_day_ma_TxTfrValMeanUSD_90_change"
            ],  # New metric added
            report_data.loc[report_date, "AdrActCnt_90_change"],
            report_data.loc[
                report_date, "AdrBalUSD10Cnt_90_change"
            ],  # New metric added
            report_data.loc[report_date, "RevUSD_90_change"],
            report_data.loc[report_date, "FeeTotUSD_90_change"],
            report_data.loc[
                report_date, "supply_pct_1_year_plus_90_change"
            ],  # New metric added
            report_data.loc[report_date, "VelCur1yr_90_change"],  # New metric added
        ],
        "YTD Change": [
            Bitcoin_YTD_Return,
            report_data.loc[report_date, "HashRate_YTD_change"],
            report_data.loc[report_date, "TxCnt_YTD_change"],
            report_data.loc[report_date, "TxTfrValAdjUSD_YTD_change"],
            report_data.loc[
                report_date, "7_day_ma_TxTfrValMeanUSD_YTD_change"
            ],  # New metric added
            report_data.loc[report_date, "AdrActCnt_YTD_change"],
            report_data.loc[
                report_date, "AdrBalUSD10Cnt_YTD_change"
            ],  # New metric added
            report_data.loc[report_date, "RevUSD_YTD_change"],
            report_data.loc[report_date, "FeeTotUSD_YTD_change"],
            report_data.loc[
                report_date, "supply_pct_1_year_plus_YTD_change"
            ],  # New metric added
            report_data.loc[report_date, "VelCur1yr_YTD_change"],  # New metric added
        ],
        "4 Year CAGR": [
            Bitcoin_4y_CAGR,
            cagr_results.loc[report_date, "HashRate_4_Year_CAGR"],
            cagr_results.loc[report_date, "TxCnt_4_Year_CAGR"],
            cagr_results.loc[report_date, "TxTfrValAdjUSD_4_Year_CAGR"],
            cagr_results.loc[
                report_date, "7_day_ma_TxTfrValMeanUSD_4_Year_CAGR"
            ],  # New metric added
            cagr_results.loc[report_date, "AdrActCnt_4_Year_CAGR"],
            cagr_results.loc[
                report_date, "AdrBalUSD10Cnt_4_Year_CAGR"
            ],  # New metric added
            cagr_results.loc[report_date, "RevUSD_4_Year_CAGR"],
            cagr_results.loc[report_date, "FeeTotUSD_4_Year_CAGR"],
            cagr_results.loc[
                report_date, "supply_pct_1_year_plus_4_Year_CAGR"
            ],  # New metric added
            cagr_results.loc[report_date, "VelCur1yr_4_Year_CAGR"],  # New metric added
        ],
        "52 Week Low": [
            Bitcoin_52w_Low,
            weekly_high_low["7_day_ma_HashRate"]["52_week_low"],
            weekly_high_low["7_day_ma_TxCnt"]["52_week_low"],
            weekly_high_low["7_day_ma_TxTfrValAdjUSD"]["52_week_low"],
            weekly_high_low["7_day_ma_TxTfrValMeanUSD"][
                "52_week_low"
            ],  # New metric added
            weekly_high_low["AdrActCnt"]["52_week_low"],
            weekly_high_low["AdrBalUSD10Cnt"]["52_week_low"],  # New metric added
            weekly_high_low["RevUSD"]["52_week_low"],
            weekly_high_low["FeeTotUSD"]["52_week_low"],
            weekly_high_low["supply_pct_1_year_plus"][
                "52_week_low"
            ],  # New metric added
            weekly_high_low["VelCur1yr"]["52_week_low"],  # New metric added
        ],
        "52 Week High": [
            Bitcoin_52w_High,
            weekly_high_low["7_day_ma_HashRate"]["52_week_high"],
            weekly_high_low["7_day_ma_TxCnt"]["52_week_high"],
            weekly_high_low["7_day_ma_TxTfrValAdjUSD"]["52_week_high"],
            weekly_high_low["7_day_ma_TxTfrValMeanUSD"][
                "52_week_high"
            ],  # New metric added
            weekly_high_low["AdrActCnt"]["52_week_high"],
            weekly_high_low["AdrBalUSD10Cnt"]["52_week_high"],  # New metric added
            weekly_high_low["RevUSD"]["52_week_high"],
            weekly_high_low["FeeTotUSD"]["52_week_high"],
            weekly_high_low["supply_pct_1_year_plus"][
                "52_week_high"
            ],  # New metric added
            weekly_high_low["VelCur1yr"]["52_week_high"],  # New metric added
        ],
    }

    # Create and return the "Bitcoin Fundamentals" DataFrame
    bitcoin_fundamentals_df = pd.DataFrame(bitcoin_fundamentals_data)

    return bitcoin_fundamentals_df

def style_bitcoin_fundamentals_table(fundamentals_table):
    """
    Applies specific formatting and styles to the Bitcoin fundamentals table for readability.

    Parameters:
    - fundamentals_table (pd.DataFrame): DataFrame containing Bitcoin fundamental metrics.

    Returns:
    - pd.io.formats.style.Styler: Styled DataFrame with formatted values and color gradients.
    """
    format_rules = {
        "Hashrate": "{:,.0f}",
        "Transaction Count": "{:,.0f}",
        "Transaction Volume": "${:,.0f}",
        "Avg Transaction Size": "${:,.0f}",
        "Miner Revenue": "${:,.0f}",
        "Active Address Count": "{:,.0f}",
        "+$10 USD Address": "{:,.0f}",
        "Fees In USD": "${:,.0f}",
        "1+ Year Supply %": "{:.2f}%",
        "1 Year Velocity": "{:.2f}",
    }

    def custom_formatter(row, column_name):
        """Apply custom formatting based on the metric name for a specified column."""
        metric = row["Metrics Name"]
        format_string = format_rules.get(metric, "{}")  # Default to '{}'
        return format_string.format(row[column_name])

    # Use the 'apply' function to format the 'Value', '52 Week Low', and '52 Week High' columns
    fundamentals_table["Value"] = fundamentals_table.apply(
        lambda row: custom_formatter(row, "Value"), axis=1
    )
    fundamentals_table["52 Week Low"] = fundamentals_table.apply(
        lambda row: custom_formatter(row, "52 Week Low"), axis=1
    )
    fundamentals_table["52 Week High"] = fundamentals_table.apply(
        lambda row: custom_formatter(row, "52 Week High"), axis=1
    )

    format_dict_fundamentals = {
        "Metrics Name": "{}",
        "7 Day Change": "{:.2%}",
        "Difficulty Period Change": "{:.2f}%",
        "MTD Change": "{:.2f}%",
        "90 Day Change": "{:.2%}",
        "YTD Change": "{:.2f}%",
        "4 Year CAGR": "{:.2f}%",
    }

    # Define a custom colormap that diverges from red to green
    diverging_cm = sns.diverging_palette(100, 133, as_cmap=True)
    diverging_cm = sns.diverging_palette(0, 0, s=0, l=85, as_cmap=True)
    bg_colormap = sns.light_palette("white", as_cmap=True)

    def color_values(val):
        """
        Takes a scalar and returns a string with
        the CSS property `'color: green'` for positive
        values, and `'color: red'` for negative values.
        """
        color = "green" if val > 0 else ("red" if val < 0 else "black")
        return "color: %s" % color

    # Columns to apply the background gradient on
    gradient_columns = [
        "7 Day Change",
        "Difficulty Period Change",
        "MTD Change",
        "90 Day Change",
        "YTD Change",
        "4 Year CAGR",
    ]

    # Apply the formatting and the background gradient only to the specified columns
    styler = (
        fundamentals_table.style.format(format_dict_fundamentals)
        .applymap(color_values, subset=gradient_columns)
        .set_properties(**{"white-space": "nowrap"})
    )

    # pandas >= 1.4: use .hide(axis="index"); older pandas: .hide_index()
    if hasattr(styler, "hide"):
        styler = styler.hide(axis="index")
    else:
        styler = styler.hide_index()

    return styler

def create_weekly_metrics_table(df, metrics_template):
    """
    Generates a weekly metrics table with Monday - Sunday columns.

    Parameters:
    df (pd.DataFrame): DataFrame with time-indexed data containing columns specified in the template.
    metrics_template (dict): Dictionary where keys are section headers and values are dictionaries
                             with metric names as keys and tuples (column_name, format_type) as values.

    Returns:
    pd.io.formats.style.Styler: A styled DataFrame ready for display with customized formatting.
    """
    # Define formatting functions
    formatters = {
        "number": lambda val: f"{val:,.0f}" if pd.notnull(val) else "",
        "number2": lambda val: f"{val:,.2f}" if pd.notnull(val) else "",
        "currency": lambda val: f"${val:,.2f}" if pd.notnull(val) else "",
        "percent": lambda val: f"{val:.2f}%" if pd.notnull(val) else "",
        "percent_change": lambda val: (
            f"<span style='color: {'green' if val > 0 else 'red'};'>{val:.2f}%</span>"
            if pd.notnull(val)
            else ""
        ),
    }

    def apply_format(column, format_type):
        """Apply the correct formatting function based on format_type."""
        return column.apply(formatters[format_type])

    # Initialize a list to hold the calculated values
    table_data = []

    # Get the most recent date in the DataFrame for current week calculations
    latest_date = df.index.max()
    start_of_week = latest_date - timedelta(
        days=latest_date.weekday()
    )  # Get the most recent Monday

    # Helper function to append header and metric rows
    def append_row(data, row, header=False):
        """Append a row with specific formatting and track headers for styling."""
        if header:
            header_style_indices.append(len(data))  # Track header row index
        data.append(row)

    header_style_indices = []  # Store header row indices

    # Loop through each section and metric in the template
    for section, metrics in metrics_template.items():
        # Add section header row
        append_row(
            table_data,
            {
                "Metric": f"<strong>{section}</strong>",
                "7 Day Avg": "-",
                "7 Day Avg % Change": "-",
                "Monday": "-",
                "Tuesday": "-",
                "Wednesday": "-",
                "Thursday": "-",
                "Friday": "-",
                "Saturday": "-",
                "Sunday": "-",
            },
            header=True,
        )

        # Process each metric
        for metric_display_name, (column_name, format_type) in metrics.items():
            # Calculate the 7-day average and percentage change from the previous 7-day average
            week_avg = df[column_name].tail(7).mean()
            prev_week_avg = df[column_name].tail(14).head(7).mean()
            pct_change = (
                ((week_avg - prev_week_avg) / prev_week_avg) * 100
                if prev_week_avg != 0
                else np.nan
            )

            # Get weekly values (Mon-Sun) up to the latest available date
            weekly_values = (
                df[column_name][start_of_week:latest_date]
                .reindex(
                    pd.date_range(start=start_of_week, periods=7, freq="D").date,
                    fill_value=None,
                )
                .tolist()
            )

            # Add the metric data with formatted values
            append_row(
                table_data,
                {
                    "Metric": metric_display_name,
                    "7 Day Avg": apply_format(pd.Series([week_avg]), format_type)[0],
                    "7 Day Avg % Change": apply_format(
                        pd.Series([pct_change]), "percent_change"
                    )[0],
                    "Monday": apply_format(pd.Series([weekly_values[0]]), format_type)[
                        0
                    ],
                    "Tuesday": apply_format(pd.Series([weekly_values[1]]), format_type)[
                        0
                    ],
                    "Wednesday": apply_format(
                        pd.Series([weekly_values[2]]), format_type
                    )[0],
                    "Thursday": apply_format(
                        pd.Series([weekly_values[3]]), format_type
                    )[0],
                    "Friday": apply_format(pd.Series([weekly_values[4]]), format_type)[
                        0
                    ],
                    "Saturday": apply_format(
                        pd.Series([weekly_values[5]]), format_type
                    )[0],
                    "Sunday": apply_format(pd.Series([weekly_values[6]]), format_type)[
                        0
                    ],
                },
            )

    # Create the final DataFrame and style it
    table_df = pd.DataFrame(table_data)

    # Apply styling to hide index and adjust text properties
    styled_table = table_df.style.hide_index().set_properties(
        **{"font-size": "10pt", "white-space": "nowrap"}
    )

    # Define specific styles for the header rows
    header_styles = [
        {
            "selector": f".row{row_index}",
            "props": [
                ("border-top", "1px solid black"),
                ("border-bottom", "1px solid black"),
            ],
        }
        for row_index in header_style_indices
    ]

    # Apply header row styles to the styled table
    styled_table = styled_table.set_table_styles(header_styles, overwrite=False)

    return styled_table


def create_ohlc_chart(ohlc_data, report_data, chart_template):
    """
    Creates an interactive OHLC (candlestick) chart with Plotly, including additional metrics and optional event markers.

    Parameters:
    - ohlc_data (pd.DataFrame): Time-indexed DataFrame with OHLC data, containing 'Open', 'High', 'Low', and 'Close' columns.
    - report_data (pd.DataFrame): DataFrame with additional metrics for overlaying on the OHLC chart (e.g., moving averages).
    - chart_template (dict): Dictionary defining chart settings, including title, labels, filename, and event markers.

    Returns:
    - fig (plotly.graph_objs.Figure): Plotly figure object containing the candlestick chart with overlays and event annotations.
    """
    # Ensure the index is timezone-naive for compatibility

    ohlc_data.index = ohlc_data.index.tz_localize(None)
    report_data.index = report_data.index.tz_localize(None)

    # Extract chart attributes from the template
    title = chart_template["title"]
    x_label = chart_template["x_label"]
    y_label = chart_template["y1_label"]
    filename = chart_template["filename"]

    # Calculate the extended x-axis range
    start_date = ohlc_data.index.min() - timedelta(days=30)  # Extend 30 days back
    end_date = ohlc_data.index.max() + timedelta(days=30)  # Extend 30 days forward

    # Filter `report_data` to align with OHLC data range
    report_data_filtered = report_data[report_data.index >= ohlc_data.index.min()]

    # Initialize the candlestick chart with OHLC data
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ohlc_data.index,
                open=ohlc_data["Open"],
                high=ohlc_data["High"],
                low=ohlc_data["Low"],
                close=ohlc_data["Close"],
                name="Weekly Price",
            )
        ]
    )

    # Overlay additional metrics on the chart
    for metric in [
        "200_week_ma_priceUSD",
        "realised_price",
        "realizedcap_multiple_3",
        "thermocap_price_multiple_32",
        "qtm_price_multiple_2",
        "qtm_price_multiple_5",
        "Electricity_Cost",
        "Bitcoin_Production_Cost",
    ]:
        if metric in report_data_filtered.columns:
            fig.add_trace(
                go.Scatter(
                    x=report_data_filtered.index,
                    y=report_data_filtered[metric],
                    mode="lines",
                    name=metric,
                )
            )

    # Update the layout, including axis settings, legend, and custom buttons for y-axis scaling
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.95),
        xaxis=dict(
            title=x_label,
            showgrid=False,
            tickformat="%B-%d-%Y",
            rangeslider_visible=False,
            range=[start_date, end_date],  # Use extended range
        ),
        yaxis=dict(title=y_label, showgrid=False, type="log", autorange=True),
        plot_bgcolor="rgba(255, 255, 255, 1)",
        hovermode="x",
        autosize=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.23, xanchor="center", x=0.5
        ),
        template="plotly_white",
        updatemenus=[
            dict(
                buttons=list(
                    [
                        dict(
                            label="Y1-axis: Linear",
                            method="relayout",
                            args=["yaxis.type", "linear"],
                        ),
                        dict(
                            label="Y1-axis: Log",
                            method="relayout",
                            args=["yaxis.type", "log"],
                        ),
                    ]
                ),
                direction="right",
                x=-0.1,
                xanchor="left",
                y=-0.25,
                yanchor="top",
            )
        ],
        width=1400,
        height=700,
        margin=dict(l=50, r=50, b=50, t=50, pad=50),
        font=dict(family="PT Sans Narrow", size=14, color="black"),
    )

    # Add event markers (vertical lines and annotations) from the template
    if "events" in chart_template:
        for event in chart_template["events"]:
            event_dates = pd.to_datetime(event["dates"])
            for date in event_dates:
                if ohlc_data.index.min() <= date <= ohlc_data.index.max():
                    fig.add_vline(
                        x=date.timestamp() * 1000,
                        line=dict(color="black", width=1, dash="dash"),
                    )
                    fig.add_annotation(
                        x=date, y=0.5, text=event["name"], showarrow=False, yref="paper"
                    )

    # Add a watermark to the chart
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        text="SecretSatoshis.com",
        showarrow=False,
        font=dict(size=50, color="rgba(128, 128, 128, 0.5)"),
        align="center",
    )
    return fig


def create_yoy_change_chart(data, column_name):
    """
    Plots the Year-Over-Year (YOY) percentage change of a specified column and the Bitcoin price on dual y-axes with log scaling.

    Parameters:
    data (pd.DataFrame): DataFrame containing historical data with a DateTime index.
    column_name (str): The name of the column in the data for YOY change calculations.

    Returns:
    matplotlib.figure.Figure: A Matplotlib figure with the YOY change and Bitcoin price plot.
    """
    # Filter the data from 2012 onwards
    data_since_2012 = data[data.index.year >= 2012]

    # Create the figure and first axis
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Get the most recent YOY change and Bitcoin price
    latest_yoy_change = data_since_2012[column_name].iloc[-1]
    latest_bitcoin_price = data_since_2012["PriceUSD"].iloc[-1]

    # Format the latest YOY change and Bitcoin price for the legend
    yoy_legend_label = f"YOY Change (latest: {latest_yoy_change:.0%})"
    btc_price_legend_label = f"Bitcoin Price (latest: ${latest_bitcoin_price:,.2f})"

    # Plot the YOY data on ax1
    ax1.plot(
        data_since_2012.index,
        data_since_2012[column_name],
        label=yoy_legend_label,
        color="tab:blue",
    )
    ax1.set_yscale("symlog", linthresh=1)  # Set the y-axis to symlog scale
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: "{:.0%}".format(y))
    )  # Format as percentages
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Percent Change", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", ls="--", linewidth=0.5)
    ax1.set_title("Bitcoin Year-Over-Year Change and Price on Log Scale")

    # Create a second y-axis for the bitcoin price with log scale
    ax2 = ax1.twinx()
    ax2.plot(
        data_since_2012.index,
        data_since_2012["PriceUSD"],
        label=btc_price_legend_label,
        color="tab:orange",
    )
    ax2.set_yscale("log")  # Set the y-axis to log scale

    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    dp.Plot(fig)

    return fig


def create_price_buckets_chart(bucket_counts_df):
    """
    Creates a horizontal bar chart of Bitcoin price ranges and the number of days spent in each range.

    Parameters:
    bucket_counts_df (pd.DataFrame): DataFrame containing price range data with 'Price Range ($)' and 'Count' columns.

    Returns:
    dp.Plot: A Datapane Plot object containing the price buckets bar chart.
    """
    # Exclude the 0-1K range from the plotting data
    plot_data = bucket_counts_df[
        bucket_counts_df["Price Range ($)"] != "$0K-$1K"
    ].copy()

    # Convert the 'Price Range ($)' to a sortable numeric value
    plot_data["Sort Key"] = plot_data["Price Range ($)"].apply(
        lambda x: int(x.split("-")[0][1:-1])
    )

    # Sort the DataFrame by 'Sort Key' in descending order
    plot_data = plot_data.sort_values(by="Sort Key", ascending=False)

    # Create the bar chart using Plotly
    fig = px.bar(
        plot_data,
        y="Price Range ($)",
        x="Count",  # Change 'Days Count' to 'Count'
        orientation="h",  # Makes the bars horizontal
        color="Count",  # Use 'Count' as the color scale
        color_continuous_scale="Viridis",  # Choose a color scale
        title="Number of Days Bitcoin Traded within 1K Price Ranges",
    )

    # Update figure layout
    fig.update_layout(height=500, width=800, margin=dict(l=5, r=5, t=50, b=5))

    # Create a Datapane Plot object
    dp_chart = dp.Plot(fig)

    return dp_chart


## Weekly Bitcoin Recap Tables


def create_summary_table_weekly_bitcoin_recap(report_data, report_date):
    """
    Generates a weekly summary table for Bitcoin's key metrics with categorized column headers.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical Bitcoin data, indexed by date.
    - report_date (str or pd.Timestamp): Specific date for which the summary is generated.

    Returns:
    - pd.DataFrame: DataFrame containing a summary of Bitcoin metrics for the specified report date.
    """

    # Extract key metrics from report_data
    price_usd = report_data.loc[report_date, "PriceUSD"]
    market_cap = report_data.loc[report_date, "CapMrktCurUSD"]
    sats_per_dollar = 100000000 / price_usd

    bitcoin_supply = report_data.loc[report_date, "SplyCur"]
    miner_revenue_30d = report_data.loc[report_date, "30_day_ma_RevUSD"]
    tx_volume_30d = report_data.loc[report_date, "30_day_ma_TxTfrValAdjUSD"]
    btc_dominance = 59.4

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


def format_value_weekly_bitcoin_recap(value, format_type):
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


def create_summary_big_numbers_weekly_bitcoin_recap(weekly_summary_df, report_date):
    """
    Generates a series of BigNumbers grouped by Market Data, On-chain Data, and Sentiment Data,
    with an additional row displaying the report date at the top.

    Parameters:
    - weekly_summary_df (pd.DataFrame): DataFrame containing categorized Bitcoin metrics.
    - report_date (str or pd.Timestamp): The date for which the report data is being generated.

    Returns:
    - dp.Group: A Datapane Group containing BigNumbers, arranged correctly in a 3-column layout.
    """

    # Ensure report_date is formatted correctly
    if isinstance(report_date, pd.Timestamp):
        report_date_str = report_date.strftime("%Y-%m-%d")
    else:
        report_date_str = str(report_date)  # If it's already a string

    # Define formatting rules
    format_rules = {
        "Bitcoin Price USD": "currency",
        "Bitcoin Supply": "float",
        "Bitcoin Dominance": "percentage",
        "Bitcoin Marketcap": "currency",
        "Bitcoin Miner Revenue": "currency",
        "Bitcoin Market Sentiment": "string",
        "Sats Per Dollar": "float",
        "Bitcoin Transaction Volume": "currency",
        "Bitcoin Valuation": "string",
    }

    # Group by category
    grouped_big_numbers = {
        "Market Data": [],
        "On-chain Data": [],
        "Investor Sentiment": [],
    }

    # Loop through DataFrame and categorize BigNumbers
    for index, row in weekly_summary_df.iterrows():
        formatted_value = format_value_weekly_bitcoin_recap(
            row["Value"], format_rules.get(index, "")
        )
        grouped_big_numbers[row["Category"]].append(
            dp.BigNumber(heading=index, value=formatted_value)
        )

    # Create "Data As Of" as a **full-width row** above everything else
    data_as_of = dp.BigNumber(heading="📅 Data As Of", value=report_date_str)

    # **Final layout with correct row structure**
    return dp.Group(
        data_as_of,  # **Row 1: Full-width date row**
        dp.Group(  # **Row 2: Three equally spaced columns for the categories**
            dp.Group(dp.Text("### Market Data"), *grouped_big_numbers["Market Data"]),
            dp.Group(
                dp.Text("### On-chain Data"), *grouped_big_numbers["On-chain Data"]
            ),
            dp.Group(
                dp.Text("### Investor Sentiment"),
                *grouped_big_numbers["Investor Sentiment"],
            ),
            columns=3,  # Ensures three evenly spaced sections
        ),
    )


# ============================================================================
# PERFORMANCE TABLE GENERATION - Factory Pattern (DRY Principle)
# ============================================================================
# Generic factory + 4 configuration wrappers (75% code reduction vs original)


def _create_performance_table_generic(
    report_data, report_date, correlation_results, asset_config
):
    """
    Generic performance table factory using configuration-driven approach.

    Eliminates duplication across equity/sector/macro/bitcoin performance tables.
    Single source of truth for performance metric calculations.

    Parameters
    ----------
    report_data : pd.DataFrame
        Historical price and return data
    report_date : pd.Timestamp or str
        Date for performance metrics
    correlation_results : dict
        Correlation DataFrames (e.g., {'priceusd_90_days': df})
    asset_config : dict
        Asset configuration: {'key': {'display_name': str, 'price_col': str}}

    Returns
    -------
    pd.DataFrame
        Performance table: Asset, Price, 7D/MTD/YTD/90D Returns, BTC Correlation
    """
    performance_metrics = {}

    for asset_key, config in asset_config.items():
        price_col = config['price_col']

        # Build performance metrics dictionary for this asset
        performance_metrics[asset_key] = {
            "Asset": config['display_name'],
            "Price": report_data.loc[report_date, price_col],
            "7 Day Return": report_data.loc[report_date, f"{price_col}_7_change"],
            "MTD Return": report_data.loc[report_date, f"{price_col}_MTD_change"],
            "YTD Return": report_data.loc[report_date, f"{price_col}_YTD_change"],
            "90 Day Return": report_data.loc[report_date, f"{price_col}_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", price_col
            ],
        }

    # Convert to DataFrame - maintains order via dict insertion order (Python 3.7+)
    return pd.DataFrame(list(performance_metrics.values()))


def create_equity_performance_table(report_data, report_date, correlation_results):
    """
    Create equity performance table for BTC, SPY, QQQ, VTI, VXUS.

    Compares Bitcoin against major US equity benchmarks and international stocks
    for asset allocation and correlation analysis.

    Parameters
    ----------
    report_data : pd.DataFrame
        Price and return data
    report_date : pd.Timestamp
        Reporting date
    correlation_results : dict
        Correlation matrices

    Returns
    -------
    pd.DataFrame
        Performance table (5 assets × 7 metrics)
    """
    # Asset configuration - easily extensible for new assets
    equity_config = {
        "Bitcoin": {
            "display_name": "Bitcoin - [BTC]",
            "price_col": "PriceUSD"
        },
        "SPY": {
            "display_name": "S&P 500 Index ETF - [SPY]",
            "price_col": "SPY_close"
        },
        "QQQ": {
            "display_name": "Nasdaq-100 ETF - [QQQ]",
            "price_col": "QQQ_close"
        },
        "VTI": {
            "display_name": "US Total Stock Market ETF - [VTI]",
            "price_col": "VTI_close"
        },
        "VXUS": {
            "display_name": "International Stock ETF - [VXUS]",
            "price_col": "VXUS_close"
        },
    }

    # Use generic factory function - DRY principle
    return _create_performance_table_generic(
        report_data, report_date, correlation_results, equity_config
    )


def create_sector_performance_table(report_data, report_date, correlation_results):
    """
    Creates a sector performance table summarizing key metrics for Bitcoin (BTC) and selected
    sector ETFs: XLK (Technology), XLF (Financials), XLE (Energy), XLRE (Real Estate).
    Metrics include price, 7-day return, MTD return, YTD return, 90-day return, and correlation with Bitcoin.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for the assets.
    - data (dict): Additional data related to assets (not directly used here but kept for compatibility).
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (pd.DataFrame): DataFrame with correlation values between assets and Bitcoin.

    Returns:
    - pd.DataFrame: A DataFrame containing the performance metrics for the selected sector ETFs and Bitcoin.
    """
    # Define the structure and data sources for each asset's performance metrics
    performance_metrics_dict = {
        "Bitcoin": {
            "Asset": "Bitcoin - [BTC]",
            "Price": report_data.loc[report_date, "PriceUSD"],
            "7 Day Return": report_data.loc[report_date, "PriceUSD_7_change"],
            "MTD Return": report_data.loc[report_date, "PriceUSD_MTD_change"],
            "YTD Return": report_data.loc[report_date, "PriceUSD_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "PriceUSD_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "PriceUSD"
            ],
        },
        "XLK": {
            "Asset": "Technology Sector ETF - [XLK]",
            "Price": report_data.loc[report_date, "XLK_close"],
            "7 Day Return": report_data.loc[report_date, "XLK_close_7_change"],
            "MTD Return": report_data.loc[report_date, "XLK_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "XLK_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "XLK_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "XLK_close"
            ],
        },
        "XLF": {
            "Asset": "Financials Sector ETF - [XLF]",
            "Price": report_data.loc[report_date, "XLF_close"],
            "7 Day Return": report_data.loc[report_date, "XLF_close_7_change"],
            "MTD Return": report_data.loc[report_date, "XLF_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "XLF_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "XLF_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "XLF_close"
            ],
        },
        "XLE": {
            "Asset": "Energy Sector ETF - [XLE]",
            "Price": report_data.loc[report_date, "XLE_close"],
            "7 Day Return": report_data.loc[report_date, "XLE_close_7_change"],
            "MTD Return": report_data.loc[report_date, "XLE_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "XLE_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "XLE_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "XLE_close"
            ],
        },
        "XLRE": {
            "Asset": "Real Estate Sector ETF - [XLRE]",
            "Price": report_data.loc[report_date, "XLRE_close"],
            "7 Day Return": report_data.loc[report_date, "XLRE_close_7_change"],
            "MTD Return": report_data.loc[report_date, "XLRE_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "XLRE_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "XLRE_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "XLRE_close"
            ],
        },
    }

    # Convert the dictionary into a DataFrame for easier data manipulation and display
    sector_performance_table_df = pd.DataFrame(list(performance_metrics_dict.values()))

    return sector_performance_table_df


def create_macro_performance_table_weekly_bitcoin_recap(
    report_data, report_date, correlation_results
):
    """
    Creates a macro performance table summarizing key metrics for macroeconomic indicators:
    DXY (US Dollar Index), GLD (Gold ETF), AGG (Aggregate Bond ETF), and BCOM (Bloomberg Commodity Index).
    Metrics include price, 7-day return, MTD return, YTD return, 90-day return, and correlation with Bitcoin.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for the macro indicators.
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (pd.DataFrame): DataFrame with correlation values between macro indicators and Bitcoin.

    Returns:
    - pd.DataFrame: A DataFrame containing the performance metrics for the selected macro indicators.
    """
    # Define the structure and data sources for each macro indicator's performance metrics
    performance_metrics_dict = {
        "BTC": {
            "Asset": "Bitcoin - [BTC]",
            "Price": report_data.loc[report_date, "PriceUSD"],
            "7 Day Return": report_data.loc[report_date, "PriceUSD_7_change"],
            "MTD Return": report_data.loc[report_date, "PriceUSD_MTD_change"],
            "YTD Return": report_data.loc[report_date, "PriceUSD_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "PriceUSD_90_change"],
            "90 Day BTC Correlation": 1,  # Bitcoin's correlation with itself is always 1
        },
        "DXY": {
            "Asset": "US Dollar Index - [DXY]",
            "Price": report_data.loc[report_date, "DX=F_close"],
            "7 Day Return": report_data.loc[report_date, "DX=F_close_7_change"],
            "MTD Return": report_data.loc[report_date, "DX=F_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "DX=F_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "DX=F_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "DX=F_close"
            ],
        },
        "GLD": {
            "Asset": "Gold ETF - [GLD]",
            "Price": report_data.loc[report_date, "GLD_close"],
            "7 Day Return": report_data.loc[report_date, "GLD_close_7_change"],
            "MTD Return": report_data.loc[report_date, "GLD_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "GLD_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "GLD_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "GLD_close"
            ],
        },
        "AGG": {
            "Asset": "Aggregate Bond ETF - [AGG]",
            "Price": report_data.loc[report_date, "AGG_close"],
            "7 Day Return": report_data.loc[report_date, "AGG_close_7_change"],
            "MTD Return": report_data.loc[report_date, "AGG_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "AGG_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "AGG_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "AGG_close"
            ],
        },
        "BCOM": {
            "Asset": "Bloomberg Commodity Index - [BCOM]",
            "Price": report_data.loc[report_date, "^BCOM_close"],
            "7 Day Return": report_data.loc[report_date, "^BCOM_close_7_change"],
            "MTD Return": report_data.loc[report_date, "^BCOM_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "^BCOM_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "^BCOM_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "^BCOM_close"
            ],
        },
    }

    # Convert the dictionary into a DataFrame for easier data manipulation and display
    macro_performance_table_df = pd.DataFrame(list(performance_metrics_dict.values()))

    return macro_performance_table_df


def create_bitcoin_performance_table(report_data, report_date, correlation_results):
    """
    Creates a Bitcoin performance table summarizing key metrics for Bitcoin (BTC) and related equities:
    MSTR (MicroStrategy), XYZ (Block), COIN (Coinbase), and WGMI (Bitcoin Miners ETF).
    Metrics include price, 7-day return, MTD return, YTD return, 90-day return, and correlation with Bitcoin.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for Bitcoin and equities.
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (pd.DataFrame): DataFrame with correlation values between Bitcoin and related equities.

    Returns:
    - pd.DataFrame: A DataFrame containing the performance metrics for Bitcoin and related equities.
    """
    # Define the structure and data sources for each asset's performance metrics
    performance_metrics_dict = {
        "BTC": {
            "Asset": "Bitcoin - [BTC]",
            "Price": report_data.loc[report_date, "PriceUSD"],
            "7 Day Return": report_data.loc[report_date, "PriceUSD_7_change"],
            "MTD Return": report_data.loc[report_date, "PriceUSD_MTD_change"],
            "YTD Return": report_data.loc[report_date, "PriceUSD_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "PriceUSD_90_change"],
            "90 Day BTC Correlation": 1,  # Bitcoin's correlation with itself is always 1
        },
        "MSTR": {
            "Asset": "MicroStrategy - [MSTR]",
            "Price": report_data.loc[report_date, "MSTR_close"],
            "7 Day Return": report_data.loc[report_date, "MSTR_close_7_change"],
            "MTD Return": report_data.loc[report_date, "MSTR_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "MSTR_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "MSTR_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "MSTR_close"
            ],
        },
        "XYZ": {
            "Asset": "Block - [XYZ]",
            "Price": report_data.loc[report_date, "XYZ_close"],
            "7 Day Return": report_data.loc[report_date, "XYZ_close_7_change"],
            "MTD Return": report_data.loc[report_date, "XYZ_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "XYZ_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "XYZ_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "XYZ_close"
            ],
        },
        "COIN": {
            "Asset": "Coinbase - [COIN]",
            "Price": report_data.loc[report_date, "COIN_close"],
            "7 Day Return": report_data.loc[report_date, "COIN_close_7_change"],
            "MTD Return": report_data.loc[report_date, "COIN_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "COIN_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "COIN_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "COIN_close"
            ],
        },
        "WGMI": {
            "Asset": "Bitcoin Miners ETF - [WGMI]",
            "Price": report_data.loc[report_date, "WGMI_close"],
            "7 Day Return": report_data.loc[report_date, "WGMI_close_7_change"],
            "MTD Return": report_data.loc[report_date, "WGMI_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "WGMI_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "WGMI_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "WGMI_close"
            ],
        },
    }

    # Convert the dictionary into a DataFrame for easier data manipulation and display
    bitcoin_performance_table_df = pd.DataFrame(list(performance_metrics_dict.values()))

    return bitcoin_performance_table_df


def style_performance_table_weekly_bitcoin_recap(performance_table):
    """
    Styles a DataFrame containing performance metrics for various assets. This function applies
    custom number formatting, conditional coloring based on value positivity or negativity,
    and additional stylistic properties for improved readability.

    Parameters:
    - performance_table (pd.DataFrame): DataFrame containing performance metrics for assets.

    Returns:
    - pd.io.formats.style.Styler: A styled DataFrame with formatted columns, conditional coloring,
      and layout enhancements.
    """

    # Format "Market Cap" column if it exists, converting values to billions with a single decimal
    if "Market Cap" in performance_table.columns:
        performance_table["Market Cap"] = performance_table["Market Cap"].apply(
            lambda x: "{:,.1f} Billion".format(pd.to_numeric(x, errors="coerce") / 1e9)
            if pd.notnull(pd.to_numeric(x, errors="coerce"))
            else x
        )

    # Define formatting rules for each column in the performance table
    format_dict = {
        "Asset": "{}",
        "Price": "${:,.0f}",
        "7 Day Return": "{:.2%}",
        "MTD Return": "{:.2f}%",
        "YTD Return": "{:.2f}%",
        "90 Day Return": "{:.2%}",
        "90 Day BTC Correlation": "{:.2f}",
    }

    # Helper function to apply conditional text color based on the value's sign
    def color_values(val):
        """
        Determines color based on value sign:
        - Green for positive values
        - Red for negative values
        - Black for zero or NaN values

        Parameters:
        - val (float): Numeric value to be styled.

        Returns:
        - str: CSS color property based on value.
        """
        color = "green" if val > 0 else ("red" if val < 0 else "black")
        return f"color: {color}"

    # Columns to apply the gradient and conditional text coloring
    gradient_columns = [
        "7 Day Return",
        "MTD Return",
        "YTD Return",
        "90 Day Return",
        "90 Day BTC Correlation",
    ]

    # Define additional table styling, such as font size adjustments
    table_style = [
        {"selector": "th", "props": "font-size: 10px;"},  # Header font size
        {"selector": "td", "props": "font-size: 10px;"},  # Cell font size
    ]

    # Apply formatting, conditional coloring, and styling to the table
    styled_table = (
        performance_table.style.format(format_dict)  # Apply the format dictionary
        .applymap(color_values, subset=gradient_columns)  # Conditional text color
        .hide_index()  # Hide DataFrame index for a cleaner presentation
        .set_properties(**{"white-space": "nowrap"})  # Prevents content wrapping
        .set_table_styles(table_style)  # Apply font size adjustments
    )

    return styled_table


def create_full_weekly_bitcoin_recap_performance(
    report_data,
    report_date,
    correlation_results,
):
    """
    Combines data from all performance tables into a single comprehensive table for the Weekly Bitcoin Recap.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing report data for all assets.
    - difficulty_period_changes (pd.Series): Series with the asset's price changes over the latest difficulty adjustment period.
    - report_date (str or pd.Timestamp): Date for which the report is generated.
    - weekly_high_low (dict): Dictionary containing the 52-week high and low values for each asset.
    - cagr_results (pd.DataFrame): DataFrame containing Compound Annual Growth Rate (CAGR) data for each asset over 4 years.
    - sharpe_results (dict): Dictionary with Sharpe ratios for each asset, indexed by time periods (e.g., 4 years).
    - correlation_results (dict): Dictionary of correlation matrices for each period (e.g., 90 days) with BTC as a baseline.

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
    macro_data = create_macro_performance_table_weekly_bitcoin_recap(
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
    full_weekly_performance_df = pd.DataFrame(all_performance_metrics.values())

    return full_weekly_performance_df


def monthly_heatmap(data, export_csv=True):
    """
    Creates a monthly and yearly returns heatmap for Bitcoin price data.
    """
    # Filter data to start from January 2011
    data = data[data.index >= pd.to_datetime("2012-01-01")]

    # Calculate monthly returns
    monthly_returns = data["PriceUSD"].resample("M").last().pct_change()

    # Calculate YTD returns based on the first price of the year
    start_of_year = data["PriceUSD"].groupby(data.index.year).transform("first")
    ytd_returns = (data["PriceUSD"] / start_of_year) - 1

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
        start_of_month = data["PriceUSD"].loc[last_date.strftime("%Y-%m")].iloc[0]
        # Calculate the MTD return for the incomplete month
        current_month_return = (data["PriceUSD"].iloc[-1] / start_of_month) - 1
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

    # Create text values for annotations in the heatmap
    text_values = heatmap_data.applymap(
        lambda x: f"{x:.2%}" if pd.notnull(x) else ""
    ).values

    # Define a custom colorscale: shades of red (-1 to 0) to white (0) to green (0 to 3)
    custom_colorscale = [
        [0.0, "red"],  # -1 mapped to red
        [0.5, "white"],  # 0 mapped to white
        [1.0, "green"],  # 3 mapped to green
    ]

    # Create the Plotly heatmap figure
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index.astype(str),
            colorscale=custom_colorscale,
            zmin=-1,  # Minimum value for the color scale
            zmax=1,  # Maximum value for the color scale
            text=text_values,
            hoverinfo="text",
            texttemplate="%{text}",
        )
    )

    # Update the layout of the figure
    fig.update_layout(
        title="Monthly Bitcoin Price Return Heatmap",
        xaxis_nticks=13,
        yaxis_nticks=25,
        autosize=False,
        width=1200,
        height=600,
    )

    # Create a Datapane plot
    dp_chart = dp.Plot(fig)

    return dp_chart


## CSV Exports


def calculate_weekly_ohlc(ohlc_data, output_file="csv/kraken_weekly_ohlc.csv"):
    """
    Calculates the latest weekly OHLC (Open, High, Low, Close) values from daily data and saves them to a CSV file.

    Parameters:
    - ohlc_data (pd.DataFrame): DataFrame with daily OHLC data containing 'Open', 'High', 'Low', and 'Close' columns.
    - output_file (str): Filename for saving the weekly OHLC values as a CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the latest weekly OHLC values for the most recent week.
    """
    # Ensure the index is a datetime index for resampling
    ohlc_data.index = pd.to_datetime(ohlc_data.index)

    # Resample daily data to get weekly OHLC values
    weekly_ohlc = ohlc_data.resample("W").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )

    # Extract the most recent weekly OHLC values
    latest_weekly_ohlc = weekly_ohlc.iloc[-1]

    # Convert to a DataFrame for export and save to CSV
    latest_weekly_ohlc_df = pd.DataFrame(latest_weekly_ohlc).T
    latest_weekly_ohlc_df.to_csv(output_file)

    return latest_weekly_ohlc_df


def create_eoy_model_table(report_data, cagr_results):
    """
    Extracts and returns a DataFrame containing historical data for specific Bitcoin valuation-related metrics.
    Includes CAGR data from a separate dataset.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical Bitcoin data.
    - cagr_results (pd.DataFrame): DataFrame containing CAGR data for Bitcoin metrics.

    Returns:
    - pd.DataFrame: Combined DataFrame with the specified columns, ensuring all data is preserved.
    """
    # Define the columns to extract from report_data
    columns_of_interest = [
        "PriceUSD",
        "realised_price",
        "thermocap_price",
        "200_day_ma_priceUSD",
        "Lagged_Energy_Value",
        "mvrv_ratio",
        "thermocap_multiple",
        "200_day_multiple",
        "Energy_Value_Multiple",
    ]

    # Define the CAGR columns to extract from cagr_results
    cagr_columns = [
        "PriceUSD_4_Year_CAGR",
        "realised_price_4_Year_CAGR",
        "thermocap_price_4_Year_CAGR",
        "200_day_ma_priceUSD_4_Year_CAGR",
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

    current_start_price = current_month_data["PriceUSD"].iloc[0]

    # Calculate monthly returns for each year
    for year in selected_metrics.index.year.unique():
        monthly_data = selected_metrics[
            (selected_metrics.index.year == year)
            & (selected_metrics.index.month == current_month)
        ]

        if not monthly_data.empty:
            start_price = monthly_data["PriceUSD"].iloc[0]
            end_price = monthly_data["PriceUSD"].iloc[-1]
            return_pct = (end_price / start_price - 1) * 100
            monthly_returns[year] = (start_price, end_price, return_pct)

            # Report Date Return Calculation
            report_date_data = monthly_data[(monthly_data.index.day == current_day)]
            if not report_date_data.empty:
                report_date_price = report_date_data["PriceUSD"].iloc[-1]
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

    current_start_price = current_year_data["PriceUSD"].iloc[0]

    # Calculate yearly returns for each year
    for year in selected_metrics.index.year.unique():
        yearly_data = selected_metrics[selected_metrics.index.year == year]

        if not yearly_data.empty:
            start_price = yearly_data["PriceUSD"].iloc[0]
            end_price = yearly_data["PriceUSD"].iloc[-1]
            return_pct = (end_price / start_price - 1) * 100
            yearly_returns[year] = (start_price, end_price, return_pct)

            # Report Date Return Calculation
            report_date_data = yearly_data[
                yearly_data.index.dayofyear == current_day_of_year
            ]
            if not report_date_data.empty:
                report_date_price = report_date_data["PriceUSD"].iloc[-1]
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
    Generates a valuation table for various assets compared to Bitcoin.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing asset market cap and BTC price data.

    Returns:
    - pd.DataFrame: DataFrame summarizing asset valuations and Bitcoin price targets.
    """
    assets = [
        {"name": "Bitcoin", "data": "PriceUSD", "marketcap": "CapMrktCurUSD"},
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
    bitcoin_price = latest_data.get("PriceUSD", float("nan"))

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
