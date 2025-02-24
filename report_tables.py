import datapane as dp
import pandas as pd
import seaborn as sns
from datetime import date


### Difficulty Adjustment Tables


def create_difficulty_update_table(
    report_data, difficulty_report, report_date, difficulty_period_changes
):
    """
    Creates a summary table of Bitcoin's network difficulty metrics and related financial data.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical Bitcoin data, indexed by date.
    - difficulty_report (dict): Dictionary containing details of the latest difficulty change.
    - report_date (pd.Timestamp or str): Specific date for the report.
    - difficulty_period_changes (pd.DataFrame): DataFrame with Bitcoin price change data for the difficulty period.

    Returns:
    - pd.DataFrame: DataFrame containing the difficulty update data for the specified report date.
    """
    # Extract relevant data from report_data for the given date
    report_date_data = report_data.loc[report_date].name
    bitcoin_supply = report_data.loc[report_date, "SplyCur"]
    HashRate = report_data.loc[report_date, "7_day_ma_HashRate"]
    PriceUSD = report_data.loc[report_date, "PriceUSD"]
    Marketcap = report_data.loc[report_date, "CapMrktCurUSD"]
    sats_per_dollar = 100000000 / report_data.loc[report_date, "PriceUSD"]
    difficulty_period_return = difficulty_period_changes.loc["PriceUSD"]

    # Extract difficulty-related data from difficulty_report
    block_height = difficulty_report["last_difficulty_change"][0]["block_height"]
    difficulty = difficulty_report["last_difficulty_change"][0]["difficulty"]
    difficulty_change = difficulty_report["difficulty_change_percentage"][0]

    # Create a dictionary with all the collected data
    difficulty_update_data = {
        "Report Date": report_date_data,
        "Bitcoin Supply": bitcoin_supply,
        "7 Day Average Hashrate": HashRate,
        "Network Difficulty": difficulty,
        "Last Difficulty Adjustment Block Height": block_height,
        "Last Difficulty Change": difficulty_change,
        "Price USD": PriceUSD,
        "Marketcap": Marketcap,
        "Sats Per Dollar": sats_per_dollar,
        "Bitcoin Price Change Difficulty Period": difficulty_period_return,
    }

    # Convert the dictionary into a DataFrame for easier reporting
    difficulty_update_df = pd.DataFrame([difficulty_update_data])
    return difficulty_update_df


def format_value(value, format_type):
    """
    Formats a given value based on the specified type.

    Parameters:
    - value (any): The value to format.
    - format_type (str): The type of formatting to apply ('percentage', 'integer', 'float', 'currency', 'date').

    Returns:
    - str: Formatted string of the input value.
    """
    if format_type == "percentage":
        return f"{value:.2f}%"
    elif format_type == "integer":
        return f"{int(value):,}"
    elif format_type == "float":
        return f"{value:,.0f}"
    elif format_type == "currency":
        return f"${value:,.0f}"
    elif format_type == "date":
        return value.strftime("%Y-%m-%d")
    else:
        return str(value)


def create_difficulty_big_numbers(difficulty_update_df):
    """
    Creates a set of large-format numbers (BigNumbers) for display of Bitcoin difficulty metrics.

    Parameters:
    - difficulty_update_df (pd.DataFrame): DataFrame containing difficulty update data.

    Returns:
    - dp.Group: Datapane Group object containing BigNumbers, arranged in 3 columns for display.
    """
    # Define formatting rules for each metric in difficulty_update_df
    format_rules = {
        "Report Date": "date",
        "Bitcoin Supply": "integer",
        "Last Difficulty Adjustment Block Height": "float",
        "Network Difficulty": "float",
        "Last Difficulty Change": "percentage",
        "7 Day Average Hashrate": "float",
        "Price USD": "currency",
        "Marketcap": "currency",
        "Sats Per Dollar": "float",
    }

    # Initialize a list to store BigNumber objects
    big_numbers = []

    # Iterate through each metric in the DataFrame row
    for column, value in difficulty_update_df.iloc[0].items():
        # Skip 'Bitcoin Price Change Difficulty Period' as specified
        if column == "Bitcoin Price Change Difficulty Period":
            continue

        # Format the value based on the format_rules dictionary
        formatted_value = format_value(value, format_rules.get(column, ""))

        # If the metric is 'Difficulty Change', set direction for the BigNumber (upward/downward)
        if column == "Difficulty Change":
            is_upward = value >= 0
            big_numbers.append(
                dp.BigNumber(
                    heading=column,
                    value=formatted_value,
                    is_upward_change=is_upward,
                )
            )
        else:
            # Add BigNumber without directional change for other metrics
            big_numbers.append(
                dp.BigNumber(
                    heading=column,
                    value=formatted_value,
                )
            )

    # Arrange BigNumbers in a Group with 3 columns and return the result
    return dp.Group(*big_numbers, columns=3)


def create_performance_table(
    report_data,
    difficulty_period_changes,
    report_date,
    weekly_high_low,
    cagr_results,
    sharpe_results,
    correlation_results,
):
    """
    Creates a performance table for various assets, including Bitcoin, Nasdaq, S&P500,
    and others, showing key metrics such as price, returns, and correlations.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing report data for the assets.
    - difficulty_period_changes (pd.Series): Series with the asset's price changes over
      the latest difficulty adjustment period.
    - report_date (str or pd.Timestamp): Date for which the report is generated.
    - weekly_high_low (dict): Dictionary containing the 52-week high and low values
      for each asset.
    - cagr_results (pd.DataFrame): DataFrame containing Compound Annual Growth Rate (CAGR)
      data for each asset over 4 years.
    - sharpe_results (dict): Dictionary with Sharpe ratios for each asset, indexed by
      time periods (e.g., 4 years).
    - correlation_results (dict): Dictionary of correlation matrices for each period
      (e.g., 90 days) with BTC as a baseline.

    Returns:
    - pd.DataFrame: DataFrame summarizing asset performance metrics.
    """
    # Define the structure for performance metrics
    performance_metrics_dict = {
        "Bitcoin": {
            "Asset": "Bitcoin",
            "Price": report_data.loc[report_date, "PriceUSD"],
            "7 Day Return": report_data.loc[report_date, "PriceUSD_7_change"],
            "Difficulty Period Return": difficulty_period_changes.loc["PriceUSD"],
            "MTD Return": report_data.loc[report_date, "PriceUSD_MTD_change"],
            "90 Day Return": report_data.loc[report_date, "PriceUSD_90_change"],
            "YTD Return": report_data.loc[report_date, "PriceUSD_YTD_change"],
            "4 Year CAGR": cagr_results.loc[report_date, "PriceUSD_4_Year_CAGR"],
            "4 Year Sharpe": sharpe_results["PriceUSD"]["4_year"].loc[report_date],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "PriceUSD"
            ],
            "52 Week Low": weekly_high_low["PriceUSD"]["52_week_low"],
            "52 Week High": weekly_high_low["PriceUSD"]["52_week_high"],
        },
        "Nasdaq": {
            "Asset": "Nasdaq",
            "Price": report_data.loc[report_date, "^IXIC_close"],
            "7 Day Return": report_data.loc[report_date, "^IXIC_close_7_change"],
            "Difficulty Period Return": difficulty_period_changes.loc["^IXIC_close"],
            "MTD Return": report_data.loc[report_date, "^IXIC_close_MTD_change"],
            "90 Day Return": report_data.loc[report_date, "^IXIC_close_90_change"],
            "YTD Return": report_data.loc[report_date, "^IXIC_close_YTD_change"],
            "4 Year CAGR": cagr_results.loc[report_date, "^IXIC_close_4_Year_CAGR"],
            "4 Year Sharpe": sharpe_results["^IXIC_close"]["4_year"].loc[report_date],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "^IXIC_close"
            ],
            "52 Week Low": weekly_high_low["^IXIC_close"]["52_week_low"],
            "52 Week High": weekly_high_low["^IXIC_close"]["52_week_high"],
        },
        "S&P500": {
            "Asset": "S&P500",
            "Price": report_data.loc[report_date, "^GSPC_close"],
            "7 Day Return": report_data.loc[report_date, "^GSPC_close_7_change"],
            "Difficulty Period Return": difficulty_period_changes.loc["^GSPC_close"],
            "MTD Return": report_data.loc[report_date, "^GSPC_close_MTD_change"],
            "90 Day Return": report_data.loc[report_date, "^GSPC_close_90_change"],
            "YTD Return": report_data.loc[report_date, "^GSPC_close_YTD_change"],
            "4 Year CAGR": cagr_results.loc[report_date, "^GSPC_close_4_Year_CAGR"],
            "4 Year Sharpe": sharpe_results["^GSPC_close"]["4_year"].loc[report_date],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "^GSPC_close"
            ],
            "52 Week Low": weekly_high_low["^GSPC_close"]["52_week_low"],
            "52 Week High": weekly_high_low["^GSPC_close"]["52_week_high"],
        },
        "XLF": {
            "Asset": "XLF Financials ETF",
            "Price": report_data.loc[report_date, "XLF_close"],
            "7 Day Return": report_data.loc[report_date, "XLF_close_7_change"],
            "Difficulty Period Return": difficulty_period_changes.loc["XLF_close"],
            "MTD Return": report_data.loc[report_date, "XLF_close_MTD_change"],
            "90 Day Return": report_data.loc[report_date, "XLF_close_90_change"],
            "YTD Return": report_data.loc[report_date, "XLF_close_YTD_change"],
            "4 Year CAGR": cagr_results.loc[report_date, "XLF_close_4_Year_CAGR"],
            "4 Year Sharpe": sharpe_results["XLF_close"]["4_year"].loc[report_date],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "XLF_close"
            ],
            "52 Week Low": weekly_high_low["XLF_close"]["52_week_low"],
            "52 Week High": weekly_high_low["XLF_close"]["52_week_high"],
        },
        "^BCOM": {
            "Asset": "Bloomberg Commodity Index",
            "Price": report_data.loc[report_date, "^BCOM_close"],
            "7 Day Return": report_data.loc[report_date, "^BCOM_close_7_change"],
            "Difficulty Period Return": difficulty_period_changes.loc["^BCOM_close"],
            "MTD Return": report_data.loc[report_date, "^BCOM_close_MTD_change"],
            "90 Day Return": report_data.loc[report_date, "^BCOM_close_90_change"],
            "YTD Return": report_data.loc[report_date, "^BCOM_close_YTD_change"],
            "4 Year CAGR": cagr_results.loc[report_date, "^BCOM_close_4_Year_CAGR"],
            "4 Year Sharpe": sharpe_results["^BCOM_close"]["4_year"].loc[report_date],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "^BCOM_close"
            ],
            "52 Week Low": weekly_high_low["^BCOM_close"]["52_week_low"],
            "52 Week High": weekly_high_low["^BCOM_close"]["52_week_high"],
        },
        "FANG+": {
            "Asset": "FANG+ ETF",
            "Price": report_data.loc[report_date, "FANG.AX_close"],
            "7 Day Return": report_data.loc[report_date, "FANG.AX_close_7_change"],
            "Difficulty Period Return": difficulty_period_changes.loc["FANG.AX_close"],
            "MTD Return": report_data.loc[report_date, "FANG.AX_close_MTD_change"],
            "90 Day Return": report_data.loc[report_date, "FANG.AX_close_90_change"],
            "YTD Return": report_data.loc[report_date, "FANG.AX_close_YTD_change"],
            "4 Year CAGR": cagr_results.loc[report_date, "FANG.AX_close_2_Year_CAGR"],
            "4 Year Sharpe": sharpe_results["FANG.AX_close"]["2_year"].loc[report_date],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "FANG.AX_close"
            ],
            "52 Week Low": weekly_high_low["FANG.AX_close"]["52_week_low"],
            "52 Week High": weekly_high_low["FANG.AX_close"]["52_week_high"],
        },
        "BITQ": {
            "Asset": "BITQ Crypto Industry ETF",
            "Price": report_data.loc[report_date, "BITQ_close"],
            "7 Day Return": report_data.loc[report_date, "BITQ_close_7_change"],
            "Difficulty Period Return": difficulty_period_changes.loc["BITQ_close"],
            "MTD Return": report_data.loc[report_date, "BITQ_close_MTD_change"],
            "90 Day Return": report_data.loc[report_date, "BITQ_close_90_change"],
            "YTD Return": report_data.loc[report_date, "BITQ_close_YTD_change"],
            "4 Year CAGR": cagr_results.loc[report_date, "BITQ_close_2_Year_CAGR"],
            "4 Year Sharpe": sharpe_results["BITQ_close"]["2_year"].loc[report_date],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "BITQ_close"
            ],
            "52 Week Low": weekly_high_low["BITQ_close"]["52_week_low"],
            "52 Week High": weekly_high_low["BITQ_close"]["52_week_high"],
        },
        "Gold Futures": {
            "Asset": "Gold",
            "Price": report_data.loc[report_date, "GC=F_close"],
            "7 Day Return": report_data.loc[report_date, "GC=F_close_7_change"],
            "Difficulty Period Return": difficulty_period_changes.loc["GC=F_close"],
            "MTD Return": report_data.loc[report_date, "GC=F_close_MTD_change"],
            "90 Day Return": report_data.loc[report_date, "GC=F_close_90_change"],
            "YTD Return": report_data.loc[report_date, "GC=F_close_YTD_change"],
            "4 Year CAGR": cagr_results.loc[report_date, "GC=F_close_4_Year_CAGR"],
            "4 Year Sharpe": sharpe_results["GC=F_close"]["4_year"].loc[report_date],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "GC=F_close"
            ],
            "52 Week Low": weekly_high_low["GC=F_close"]["52_week_low"],
            "52 Week High": weekly_high_low["GC=F_close"]["52_week_high"],
        },
        "US Dollar Futures": {
            "Asset": "US Dollar Index",
            "Price": report_data.loc[report_date, "DX=F_close"],
            "7 Day Return": report_data.loc[report_date, "DX=F_close_7_change"],
            "Difficulty Period Return": difficulty_period_changes.loc["DX=F_close"],
            "MTD Return": report_data.loc[report_date, "DX=F_close_MTD_change"],
            "90 Day Return": report_data.loc[report_date, "DX=F_close_90_change"],
            "YTD Return": report_data.loc[report_date, "DX=F_close_YTD_change"],
            "4 Year CAGR": cagr_results.loc[report_date, "DX=F_close_4_Year_CAGR"],
            "4 Year Sharpe": sharpe_results["DX=F_close"]["4_year"].loc[report_date],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "DX=F_close"
            ],
            "52 Week Low": weekly_high_low["DX=F_close"]["52_week_low"],
            "52 Week High": weekly_high_low["DX=F_close"]["52_week_high"],
        },
        "TLT": {
            "Asset": "TLT Treasury Bond ETF",
            "Price": report_data.loc[report_date, "TLT_close"],
            "7 Day Return": report_data.loc[report_date, "TLT_close_7_change"],
            "Difficulty Period Return": difficulty_period_changes.loc["TLT_close"],
            "MTD Return": report_data.loc[report_date, "TLT_close_MTD_change"],
            "90 Day Return": report_data.loc[report_date, "TLT_close_90_change"],
            "4 Year CAGR": cagr_results.loc[report_date, "TLT_close_4_Year_CAGR"],
            "4 Year Sharpe": sharpe_results["TLT_close"]["4_year"].loc[report_date],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "TLT_close"
            ],
            "YTD Return": report_data.loc[report_date, "TLT_close_YTD_change"],
            "52 Week Low": weekly_high_low["TLT_close"]["52_week_low"],
            "52 Week High": weekly_high_low["TLT_close"]["52_week_high"],
        },
    }

    # Convert the dictionary to a DataFrame
    performance_table_df = pd.DataFrame(list(performance_metrics_dict.values()))

    return performance_table_df


def style_performance_table_weekly(performance_table):
    """
    Styles a DataFrame containing weekly performance metrics for various assets.
    The function formats numerical values, applies conditional coloring based on
    value positivity/negativity, and sets other stylistic properties for improved readability.

    Parameters:
    - performance_table (pd.DataFrame): The DataFrame containing performance metrics for
      various assets on a weekly basis.

    Returns:
    - pd.io.formats.style.Styler: A styled DataFrame with formatted columns and conditional
      colors applied for easy interpretation.
    """
    # Define formatting rules for different columns in the performance table
    format_dict = {
        "Asset": "{}",
        "Price": "${:,.0f}",
        "7 Day Return": "{:.2%}",
        "Difficulty Period Return": "{:.2}%",
        "MTD Return": "{:.2}%",
        "90 Day Return": "{:.2%}",
        "YTD Return": "{:.2}%",
        "4 Year CAGR": "{:.2f}%",
        "4 Year Sharpe": "{:,.2f}",
        "90 Day BTC Correlation": "{:,.2f}",
        "52 Week Low": "${:,.0f}",
        "52 Week High": "${:,.0f}",
    }

    # Define color maps for diverging and background color gradients
    diverging_cm = sns.diverging_palette(
        100, 133, as_cmap=True
    )  # Diverging palette for color variation
    bg_colormap = sns.light_palette(
        "white", as_cmap=True
    )  # Background light color palette

    # Helper function to apply conditional text color based on the value sign
    def color_values(val):
        """
        Determines color based on value sign:
        - Green for positive values
        - Red for negative values
        - Black for zero values
        """
        color = "green" if val > 0 else ("red" if val < 0 else "black")
        return f"color: {color}"

    # Specify columns to apply the gradient and conditional text coloring
    gradient_columns = [
        "7 Day Return",
        "Difficulty Period Return",
        "MTD Return",
        "90 Day Return",
        "YTD Return",
        "4 Year CAGR",
        "4 Year Sharpe",
        "90 Day BTC Correlation",
    ]

    # Apply formatting, conditional coloring, and styling
    styled_table = (
        performance_table.style.format(format_dict)  # Apply the format dictionary
        .applymap(color_values, subset=gradient_columns)  # Conditional text color
        .hide_index()  # Hide DataFrame index for cleaner presentation
        .set_properties(**{"white-space": "nowrap"})  # Prevent content wrapping
    )

    return styled_table


def create_bitcoin_fundamentals_table(
    report_data, difficulty_period_changes, weekly_high_low, report_date, cagr_results
):
    """
    Creates a summary table of Bitcoin's fundamental metrics, including current values, changes,
    and calculated CAGR.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing Bitcoin metrics data.
    - difficulty_period_changes (pd.Series): Series containing the change in Bitcoin metrics over the last difficulty period.
    - weekly_high_low (dict): Dictionary containing the 52-week high and low values for each metric.
    - report_date (str or pd.Timestamp): The specific date for which to generate the report.
    - cagr_results (pd.DataFrame): DataFrame containing Compound Annual Growth Rate (CAGR) for each metric.

    Returns:
    - pd.DataFrame: DataFrame summarizing Bitcoin's fundamental metrics, changes, CAGR, and 52-week highs and lows.
    """
    # Extract data from report_data for the specific date
    HashRate = report_data.loc[report_date, "7_day_ma_HashRate"]
    TxCnt = report_data.loc[report_date, "7_day_ma_TxCnt"]
    TxTfrValAdjUSD = report_data.loc[report_date, "7_day_ma_TxTfrValAdjUSD"]
    TxTfrValMeanUSD = report_data.loc[report_date, "7_day_ma_TxTfrValMeanUSD"]
    RevUSD = report_data.loc[report_date, "RevUSD"]
    AdrActCnt = report_data.loc[report_date, "AdrActCnt"]
    AdrBalUSD10Cnt = report_data.loc[report_date, "AdrBalUSD10Cnt"]
    FeeTotUSD = report_data.loc[report_date, "FeeTotUSD"]
    supply_pct_1_year_plus = report_data.loc[report_date, "supply_pct_1_year_plus"]
    VelCur1yr = report_data.loc[report_date, "VelCur1yr"]

    HashRate_MTD = report_data.loc[report_date, "HashRate_MTD_change"]
    TxCnt_MTD = report_data.loc[report_date, "TxCnt_MTD_change"]
    TxTfrValAdjUSD_MTD = report_data.loc[report_date, "TxTfrValAdjUSD_MTD_change"]
    TxTfrValMeanUSD_MTD = report_data.loc[
        report_date, "7_day_ma_TxTfrValMeanUSD_MTD_change"
    ]
    RevUSD_MTD = report_data.loc[report_date, "RevUSD_MTD_change"]
    AdrActCnt_MTD = report_data.loc[report_date, "AdrActCnt_MTD_change"]
    AdrBalUSD10Cnt_MTD = report_data.loc[report_date, "AdrBalUSD10Cnt_MTD_change"]
    FeeTotUSD_MTD = report_data.loc[report_date, "FeeTotUSD_MTD_change"]
    supply_pct_1_year_plus_MTD = report_data.loc[
        report_date, "supply_pct_1_year_plus_MTD_change"
    ]
    VelCur1yr_MTD = report_data.loc[report_date, "VelCur1yr_MTD_change"]

    HashRate_YTD = report_data.loc[report_date, "HashRate_YTD_change"]
    TxCnt_YTD = report_data.loc[report_date, "TxCnt_YTD_change"]
    TxTfrValAdjUSD_YTD = report_data.loc[report_date, "TxTfrValAdjUSD_YTD_change"]
    TxTfrValMeanUSD_YTD = report_data.loc[
        report_date, "7_day_ma_TxTfrValMeanUSD_YTD_change"
    ]
    RevUSD_YTD = report_data.loc[report_date, "RevUSD_YTD_change"]
    AdrActCnt_YTD = report_data.loc[report_date, "AdrActCnt_YTD_change"]
    AdrBalUSD10Cnt_YTD = report_data.loc[report_date, "AdrBalUSD10Cnt_YTD_change"]
    FeeTotUSD_YTD = report_data.loc[report_date, "FeeTotUSD_YTD_change"]
    supply_pct_1_year_plus_YTD = report_data.loc[
        report_date, "supply_pct_1_year_plus_YTD_change"
    ]
    VelCur1yr_YTD = report_data.loc[report_date, "VelCur1yr_YTD_change"]

    HashRate_90 = report_data.loc[report_date, "HashRate_90_change"]
    TxCnt_90 = report_data.loc[report_date, "TxCnt_90_change"]
    TxTfrValAdjUSD_90 = report_data.loc[report_date, "TxTfrValAdjUSD_90_change"]
    TxTfrValMeanUSD_90 = report_data.loc[
        report_date, "7_day_ma_TxTfrValMeanUSD_90_change"
    ]
    RevUSD_90 = report_data.loc[report_date, "RevUSD_90_change"]
    AdrActCnt_90 = report_data.loc[report_date, "AdrActCnt_90_change"]
    AdrBalUSD10Cnt_90 = report_data.loc[report_date, "AdrBalUSD10Cnt_90_change"]
    FeeTotUSD_90 = report_data.loc[report_date, "FeeTotUSD_90_change"]
    supply_pct_1_year_plus_90 = report_data.loc[
        report_date, "supply_pct_1_year_plus_90_change"
    ]
    VelCur1yr_90 = report_data.loc[report_date, "VelCur1yr_90_change"]

    HashRate_7 = report_data.loc[report_date, "HashRate_7_change"]
    TxCnt_7 = report_data.loc[report_date, "TxCnt_7_change"]
    TxTfrValAdjUSD_7 = report_data.loc[report_date, "TxTfrValAdjUSD_7_change"]
    TxTfrValMeanUSD_7 = report_data.loc[
        report_date, "7_day_ma_TxTfrValMeanUSD_7_change"
    ]
    RevUSD_7 = report_data.loc[report_date, "RevUSD_7_change"]
    AdrActCnt_7 = report_data.loc[report_date, "AdrActCnt_7_change"]
    AdrBalUSD10Cnt_7 = report_data.loc[report_date, "AdrBalUSD10Cnt_7_change"]
    FeeTotUSD_7 = report_data.loc[report_date, "FeeTotUSD_7_change"]
    supply_pct_1_year_plus_7 = report_data.loc[
        report_date, "supply_pct_1_year_plus_7_change"
    ]
    VelCur1yr_7 = report_data.loc[report_date, "VelCur1yr_7_change"]

    HashRate_CAGR = cagr_results.loc[report_date, "HashRate_4_Year_CAGR"]
    TxCnt_CAGR = cagr_results.loc[report_date, "TxCnt_4_Year_CAGR"]
    TxTfrValAdjUSD_CAGR = cagr_results.loc[report_date, "TxTfrValAdjUSD_4_Year_CAGR"]
    TxTfrValMeanUSD_CAGR = cagr_results.loc[report_date, "TxTfrValMeanUSD_4_Year_CAGR"]
    RevUSD_CAGR = cagr_results.loc[report_date, "RevUSD_4_Year_CAGR"]
    AdrActCnt_CAGR = cagr_results.loc[report_date, "AdrActCnt_4_Year_CAGR"]
    AdrBalUSD10Cnt_CAGR = cagr_results.loc[report_date, "AdrBalUSD10Cnt_4_Year_CAGR"]
    FeeTotUSD_CAGR = cagr_results.loc[report_date, "FeeTotUSD_4_Year_CAGR"]
    supply_pct_1_year_plus_CAGR = cagr_results.loc[
        report_date, "supply_pct_1_year_plus_4_Year_CAGR"
    ]
    VelCur1yr_CAGR = cagr_results.loc[report_date, "VelCur1yr_4_Year_CAGR"]

    # Fetch 52-week high and low for each metric
    HashRate_52_high = weekly_high_low["7_day_ma_HashRate"]["52_week_high"]
    TxCnt_52_high = weekly_high_low["7_day_ma_TxCnt"]["52_week_high"]
    TxTfrValAdjUSD_52_high = weekly_high_low["7_day_ma_TxTfrValAdjUSD"]["52_week_high"]
    TxTfrValMeanUSD_52_high = weekly_high_low["7_day_ma_TxTfrValMeanUSD"][
        "52_week_high"
    ]
    RevUSD_52_high = weekly_high_low["RevUSD"]["52_week_high"]
    AdrActCnt_52_high = weekly_high_low["AdrActCnt"]["52_week_high"]
    AdrBalUSD10Cnt_52_high = weekly_high_low["AdrBalUSD10Cnt"]["52_week_high"]
    FeeTotUSD_52_high = weekly_high_low["FeeTotUSD"]["52_week_high"]
    supply_pct_1_year_plus_52_high = weekly_high_low["supply_pct_1_year_plus"][
        "52_week_high"
    ]
    VelCur1yr_52_high = weekly_high_low["VelCur1yr"]["52_week_high"]

    HashRate_52_low = weekly_high_low["7_day_ma_HashRate"]["52_week_low"]
    TxCnt_52_low = weekly_high_low["7_day_ma_TxCnt"]["52_week_low"]
    TxTfrValAdjUSD_52_low = weekly_high_low["7_day_ma_TxTfrValAdjUSD"]["52_week_low"]
    TxTfrValMeanUSD_52_low = weekly_high_low["7_day_ma_TxTfrValMeanUSD"]["52_week_low"]
    RevUSD_52_low = weekly_high_low["RevUSD"]["52_week_low"]
    AdrActCnt_52_low = weekly_high_low["AdrActCnt"]["52_week_low"]
    AdrBalUSD10Cnt_52_low = weekly_high_low["AdrBalUSD10Cnt"]["52_week_low"]
    FeeTotUSD_52_low = weekly_high_low["FeeTotUSD"]["52_week_low"]
    supply_pct_1_year_plus_52_low = weekly_high_low["supply_pct_1_year_plus"][
        "52_week_low"
    ]
    VelCur1yr_52_low = weekly_high_low["VelCur1yr"]["52_week_low"]

    HashRate_difficulty_change = difficulty_period_changes.loc["7_day_ma_HashRate"]
    TxCnt_difficulty_change = difficulty_period_changes.loc["TxCnt"]
    TxTfrValAdjUSD_difficulty_change = difficulty_period_changes.loc["TxTfrValAdjUSD"]
    TxTfrValMeanUSD_difficulty_change = difficulty_period_changes.loc[
        "7_day_ma_TxTfrValMeanUSD"
    ]
    RevUSD_difficulty_change = difficulty_period_changes.loc["RevUSD"]
    AdrActCnt_difficulty_change = difficulty_period_changes.loc["AdrActCnt"]
    AdrBalUSD10Cnt_difficulty_change = difficulty_period_changes.loc["AdrBalUSD10Cnt"]
    FeeTotUSD_difficulty_change = difficulty_period_changes.loc["FeeTotUSD"]
    supply_pct_1_year_plus_difficulty_change = difficulty_period_changes.loc[
        "supply_pct_1_year_plus"
    ]
    VelCur1yr_difficulty_change = difficulty_period_changes.loc["VelCur1yr"]

    # Create a dictionary with the extracted values
    bitcoin_fundamentals_data = {
        "Metrics Name": [
            "Hashrate",
            "Transaction Count",
            "Transaction Volume",
            "Avg Transaction Size",
            "Active Address Count",
            "+$10 USD Address",
            "Miner Revenue",
            "Fees In USD",
            "1+ Year Supply %",
            "1 Year Velocity",
        ],
        "Value": [
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
            HashRate_7,
            TxCnt_7,
            TxTfrValAdjUSD_7,
            TxTfrValMeanUSD_7,
            AdrActCnt_7,
            AdrBalUSD10Cnt_7,
            RevUSD_7,
            FeeTotUSD_7,
            supply_pct_1_year_plus_7,
            VelCur1yr_7,
        ],
        "Difficulty Period Change": [
            HashRate_difficulty_change,
            TxCnt_difficulty_change,
            TxTfrValAdjUSD_difficulty_change,
            TxTfrValMeanUSD_difficulty_change,
            AdrActCnt_difficulty_change,
            AdrBalUSD10Cnt_difficulty_change,
            RevUSD_difficulty_change,
            FeeTotUSD_difficulty_change,
            supply_pct_1_year_plus_difficulty_change,
            VelCur1yr_difficulty_change,
        ],
        "MTD Change": [
            HashRate_MTD,
            TxCnt_MTD,
            TxTfrValAdjUSD_MTD,
            TxTfrValMeanUSD_MTD,
            AdrActCnt_MTD,
            AdrBalUSD10Cnt_MTD,
            RevUSD_MTD,
            FeeTotUSD_MTD,
            supply_pct_1_year_plus_MTD,
            VelCur1yr_MTD,
        ],
        "90 Day Change": [
            HashRate_90,
            TxCnt_90,
            TxTfrValAdjUSD_90,
            TxTfrValMeanUSD_90,
            AdrActCnt_90,
            AdrBalUSD10Cnt_90,
            RevUSD_90,
            FeeTotUSD_90,
            supply_pct_1_year_plus_90,
            VelCur1yr_90,
        ],
        "YTD Change": [
            HashRate_YTD,
            TxCnt_YTD,
            TxTfrValAdjUSD_YTD,
            TxTfrValMeanUSD_YTD,
            AdrActCnt_YTD,
            AdrBalUSD10Cnt_YTD,
            RevUSD_YTD,
            FeeTotUSD_YTD,
            supply_pct_1_year_plus_YTD,
            VelCur1yr_YTD,
        ],
        "4 Year CAGR": [
            HashRate_CAGR,
            TxCnt_CAGR,
            TxTfrValAdjUSD_CAGR,
            TxTfrValMeanUSD_CAGR,
            AdrActCnt_CAGR,
            AdrBalUSD10Cnt_CAGR,
            RevUSD_CAGR,
            FeeTotUSD_CAGR,
            supply_pct_1_year_plus_CAGR,
            VelCur1yr_CAGR,
        ],
        "52 Week Low": [
            HashRate_52_low,
            TxCnt_52_low,
            TxTfrValAdjUSD_52_low,
            TxTfrValMeanUSD_52_low,
            AdrActCnt_52_low,
            AdrBalUSD10Cnt_52_low,
            RevUSD_52_low,
            FeeTotUSD_52_low,
            supply_pct_1_year_plus_52_low,
            VelCur1yr_52_low,
        ],
        "52 Week High": [
            HashRate_52_high,
            TxCnt_52_high,
            TxTfrValAdjUSD_52_high,
            TxTfrValMeanUSD_52_high,
            AdrActCnt_52_high,
            AdrBalUSD10Cnt_52_high,
            RevUSD_52_high,
            FeeTotUSD_52_high,
            supply_pct_1_year_plus_52_high,
            VelCur1yr_52_high,
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
    styled_table_colors = (
        fundamentals_table.style.format(format_dict_fundamentals)
        .applymap(color_values, subset=gradient_columns)
        .hide_index()
        .set_properties(**{"white-space": "nowrap"})
    )

    return styled_table_colors


def create_bitcoin_valuation_table(
    report_data, difficulty_period_changes, weekly_high_low, valuation_data, report_date
):
    """
    Generates a valuation table for Bitcoin using different valuation models.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing report data for Bitcoin.
    - difficulty_period_changes (pd.Series): Series of changes in valuation model metrics for the latest difficulty period.
    - weekly_high_low (dict): Dictionary with 52-week highs and lows for each metric.
    - valuation_data (dict): Dictionary of target valuations for buy/sell decisions.
    - report_date (str or pd.Timestamp): The specific date for the valuation report.

    Returns:
    - pd.DataFrame: DataFrame summarizing various valuation metrics, model prices, buy/sell targets, and fair values.
    """
    # Extract BTC Value
    btc_value = report_data.loc[report_date, "PriceUSD"]

    # Extraction for "NVTAdj"
    nvt_price_multiple = report_data.loc[report_date, "nvt_price"]
    nvt_difficulty_change = difficulty_period_changes.loc["nvt_price"]
    nvt_buy_target = valuation_data["nvt_price_multiple_buy_target"]
    nvt_sell_target = valuation_data["nvt_price_multiple_sell_target"]
    nvt_pct_from_fair_value = (nvt_price_multiple - btc_value) / btc_value
    nvt_return_to_target = (nvt_sell_target - btc_value) / btc_value
    nvt_return_to_buy_target = (nvt_buy_target - btc_value) / btc_value

    # Extraction for "200_day_multiple"
    day_200_price = report_data.loc[report_date, "200_day_ma_priceUSD"]
    day_200_difficulty_change = difficulty_period_changes.loc["200_day_multiple"]
    day_200_buy_target = valuation_data["200_day_multiple_buy_target"]
    day_200_sell_target = valuation_data["200_day_multiple_sell_target"]
    day_200_pct_from_fair_value = (day_200_price - btc_value) / btc_value
    day_200_return_to_target = (day_200_sell_target - btc_value) / btc_value
    day_200_return_to_buy_target = (day_200_buy_target - btc_value) / btc_value

    # Extraction for "mvrv_ratio"
    mvrv_price = report_data.loc[report_date, "realised_price"]
    mvrv_difficulty_change = difficulty_period_changes.loc["realised_price"]
    mvrv_buy_target = valuation_data["mvrv_ratio_buy_target"]
    mvrv_sell_target = valuation_data["mvrv_ratio_sell_target"]
    mvrv_pct_from_fair_value = (mvrv_price - btc_value) / btc_value
    mvrv_return_to_target = (mvrv_sell_target - btc_value) / btc_value
    mvrv_return_to_buy_target = (mvrv_buy_target - btc_value) / btc_value

    # Extraction for "thermocap_multiple"
    thermo_price = report_data.loc[report_date, "thermocap_price_multiple_8"]
    thermo_difficulty_change = difficulty_period_changes.loc[
        "thermocap_price_multiple_8"
    ]
    thermo_buy_target = valuation_data["thermocap_multiple_buy_target"]
    thermo_sell_target = valuation_data["thermocap_multiple_sell_target"]
    thermo_pct_from_fair_value = (thermo_price - btc_value) / btc_value
    thermo_return_to_target = (thermo_sell_target - btc_value) / btc_value
    thermo_return_to_buy_target = (thermo_buy_target - btc_value) / btc_value

    # Extraction for "stocktoflow"
    sf_price = report_data.loc[report_date, "SF_Predicted_Price"]
    sf_difficulty_change = difficulty_period_changes.loc["SF_Predicted_Price"]
    sf_buy_target = valuation_data["SF_Multiple_buy_target"]
    sf_sell_target = valuation_data["SF_Multiple_sell_target"]
    sf_pct_from_fair_value = (sf_price - btc_value) / btc_value
    sf_return_to_target = (sf_sell_target - btc_value) / btc_value
    sf_return_to_buy_target = (sf_buy_target - btc_value) / btc_value

    # Extraction for "appl_marketcap"
    aapl_price = valuation_data["AAPL_MarketCap_bull_present_value"]
    aapl_difficulty_change = difficulty_period_changes.loc["AAPL_mc_btc_price"]
    aapl_buy_target = valuation_data["AAPL_MarketCap_base_present_value"]
    aapl_sell_target = report_data.loc[report_date, "AAPL_mc_btc_price"]
    aapl_pct_from_fair_value = (aapl_price - btc_value) / btc_value
    aapl_return_to_target = (aapl_sell_target - btc_value) / btc_value
    aapl_return_to_buy_target = (aapl_buy_target - btc_value) / btc_value

    # Extraction for "gold_marketcap_billion_usd"
    gold_price = valuation_data["gold_marketcap_billion_usd_bull_present_value"]
    gold_difficulty_change = difficulty_period_changes.loc["gold_marketcap_billion_usd"]
    gold_buy_target = valuation_data["gold_marketcap_billion_usd_base_present_value"]
    gold_sell_target = (
        report_data.loc[report_date, "gold_marketcap_billion_usd"]
        / report_data.loc[report_date, "SplyExpFut10yr"]
    )
    gold_pct_from_fair_value = (gold_price - btc_value) / btc_value
    gold_return_to_target = (gold_sell_target - btc_value) / btc_value
    gold_return_to_buy_target = (gold_buy_target - btc_value) / btc_value

    # Extraction for "silver_marketcap_billion_usd"
    silver_price = valuation_data["silver_marketcap_billion_usd_bull_present_value"]
    silver_difficulty_change = difficulty_period_changes.loc[
        "silver_marketcap_billion_usd"
    ]
    silver_buy_target = valuation_data[
        "silver_marketcap_billion_usd_base_present_value"
    ]
    silver_sell_target = (
        report_data.loc[report_date, "silver_marketcap_billion_usd"]
        / report_data.loc[report_date, "SplyExpFut10yr"]
    )
    silver_pct_from_fair_value = (silver_price - btc_value) / btc_value
    silver_return_to_target = (silver_sell_target - btc_value) / btc_value
    silver_return_to_buy_target = (silver_buy_target - btc_value) / btc_value

    # Extraction for "United_States_btc_price"
    us_btc_price = valuation_data["United_States_cap_bull_present_value"]
    us_difficulty_change = difficulty_period_changes.loc["United_States_btc_price"]
    us_buy_target = valuation_data["United_States_cap_base_present_value"]
    us_sell_target = report_data.loc[report_date, "United_States_btc_price"]
    us_pct_from_fair_value = (us_btc_price - btc_value) / btc_value
    us_return_to_target = (us_sell_target - btc_value) / btc_value
    us_return_to_buy_target = (us_buy_target - btc_value) / btc_value

    # Extraction for "United_Kingdom_btc_price"
    uk_btc_price = valuation_data["United_Kingdom_cap_bull_present_value"]
    uk_difficulty_change = difficulty_period_changes.loc["United_Kingdom_btc_price"]
    uk_buy_target = valuation_data["United_Kingdom_cap_base_present_value"]
    uk_sell_target = report_data.loc[report_date, "United_Kingdom_btc_price"]
    uk_pct_from_fair_value = (uk_btc_price - btc_value) / btc_value
    uk_return_to_target = (uk_sell_target - btc_value) / btc_value
    uk_return_to_buy_target = (uk_buy_target - btc_value) / btc_value

    # Update the dictionary with the extracted values
    bitcoin_valuation_data = {
        "Valuation Model": [
            "200 Day Moving Average",
            "NVT Price",
            "Realized Price",
            "ThermoCap Price",
            "Stock To Flow Price",
            "Silver Market Cap",
            "UK M0 Price",
            "Apple Market Cap",
            "US M0 Price",
            "Gold Market Cap",
        ],
        "Model Price": [
            day_200_price,
            nvt_price_multiple,
            mvrv_price,
            thermo_price,
            sf_price,
            silver_price,
            uk_btc_price,
            aapl_price,
            us_btc_price,
            gold_price,
        ],
        "Difficulty Period Change": [
            day_200_difficulty_change,
            nvt_difficulty_change,
            mvrv_difficulty_change,
            thermo_difficulty_change,
            sf_difficulty_change,
            silver_difficulty_change,
            uk_difficulty_change,
            aapl_difficulty_change,
            us_difficulty_change,
            gold_difficulty_change,
        ],
        "BTC Price": [
            btc_value,
            btc_value,
            btc_value,
            btc_value,
            btc_value,
            btc_value,
            btc_value,
            btc_value,
            btc_value,
            btc_value,
        ],
        "Buy Target": [
            day_200_buy_target,
            nvt_buy_target,
            mvrv_buy_target,
            thermo_buy_target,
            sf_buy_target,
            silver_buy_target,
            uk_buy_target,
            aapl_buy_target,
            us_buy_target,
            gold_buy_target,
        ],
        "Sell Target": [
            day_200_sell_target,
            nvt_sell_target,
            mvrv_sell_target,
            thermo_sell_target,
            sf_sell_target,
            silver_sell_target,
            uk_sell_target,
            aapl_sell_target,
            us_sell_target,
            gold_sell_target,
        ],
        "% To Buy Target": [
            day_200_return_to_buy_target,
            nvt_return_to_buy_target,
            mvrv_return_to_buy_target,
            thermo_return_to_buy_target,
            sf_return_to_buy_target,
            silver_return_to_buy_target,
            uk_return_to_buy_target,
            aapl_return_to_buy_target,
            us_return_to_buy_target,
            gold_return_to_buy_target,
        ],
        "% To Model Price": [
            day_200_pct_from_fair_value,
            nvt_pct_from_fair_value,
            mvrv_pct_from_fair_value,
            thermo_pct_from_fair_value,
            sf_pct_from_fair_value,
            silver_pct_from_fair_value,
            uk_pct_from_fair_value,
            aapl_pct_from_fair_value,
            us_pct_from_fair_value,
            gold_pct_from_fair_value,
        ],
        "% To Sell Target": [
            day_200_return_to_target,
            nvt_return_to_target,
            mvrv_return_to_target,
            thermo_return_to_target,
            sf_return_to_target,
            silver_return_to_target,
            uk_return_to_target,
            aapl_return_to_target,
            us_return_to_target,
            gold_return_to_target,
        ],
    }

    # Create and return the "Bitcoin Valuation" DataFrame
    bitcoin_valuation_df = pd.DataFrame(bitcoin_valuation_data)
    # Convert columns from 'object' to 'float64'
    bitcoin_valuation_df["Difficulty Period Change"] = pd.to_numeric(
        bitcoin_valuation_df["Difficulty Period Change"], errors="coerce"
    )
    bitcoin_valuation_df["Sell Target"] = pd.to_numeric(
        bitcoin_valuation_df["Sell Target"], errors="coerce"
    )
    bitcoin_valuation_df["% To Sell Target"] = pd.to_numeric(
        bitcoin_valuation_df["% To Sell Target"], errors="coerce"
    )
    return bitcoin_valuation_df


def style_bitcoin_valuation_table(bitcoin_valuation_table):
    """
    Styles a DataFrame containing valuation metrics for Bitcoin. The function applies
    specific formatting rules to monetary values, percentages, and other numeric data.
    Conditional coloring is used to differentiate positive and negative values, and
    additional styling adjustments improve readability.

    Parameters:
    - bitcoin_valuation_table (pd.DataFrame): DataFrame containing valuation metrics for Bitcoin.

    Returns:
    - pd.io.formats.style.Styler: A styled DataFrame with formatted columns, conditional
      coloring, and layout enhancements for readability.
    """
    # Define formatting rules for each column in the valuation table
    format_dict_valuation = {
        "Valuation Model": "{}",
        "Model Price": lambda x: "${:,.0f}".format(x)
        if pd.notnull(x)
        else x,  # Ensure numeric formatting
        "Difficulty Period Change": "{:.2f}%",  # Format percentage values
        "BTC Price": lambda x: "${:,.0f}".format(x) if pd.notnull(x) else x,
        "Buy Target": lambda x: "${:,.0f}".format(x) if pd.notnull(x) else x,
        "Sell Target": lambda x: "${:,.0f}".format(x) if pd.notnull(x) else x,
        "% To Buy Target": "{:.2%}",
        "% To Model Price": "{:.2%}",
        "% To Sell Target": "{:.2%}",
    }

    # Define color palettes for conditional formatting (not applied directly but can be used in extensions)
    diverging_cm = sns.diverging_palette(100, 133, as_cmap=True)  # Diverging palette
    bg_colormap = sns.light_palette("white", as_cmap=True)  # Light background palette

    # Helper function to apply conditional text color based on value
    def color_values(val):
        """
        Determines color based on value sign:
        - Green for positive values
        - Red for negative values
        - Black for zero or NaN values
        """
        try:
            if val > 0:
                color = "green"
            elif val < 0:
                color = "red"
            else:
                color = "black"
        except ValueError:
            color = "black"  # Default color if comparison fails
        return f"color: {color}"

    # Columns to apply the gradient and conditional text coloring
    gradient_columns = [
        "Difficulty Period Change",
        "% To Model Price",
        "% To Sell Target",
        "% To Buy Target",
    ]

    # Apply formatting, conditional coloring, and styling to the table
    styled_table_colors = (
        bitcoin_valuation_table.style.format(
            format_dict_valuation
        )  # Apply the format dictionary
        .applymap(color_values, subset=gradient_columns)  # Conditional text color
        .hide_index()  # Hide DataFrame index for cleaner presentation
        .set_properties(**{"white-space": "nowrap"})  # Prevent content wrapping
        .set_table_styles(
            [{"selector": "th", "props": [("white-space", "nowrap")]}]
        )  # Set table header styling
    )

    return styled_table_colors


### Weekly Market Summary Tables


def create_weekly_summary_table(report_data, report_date):
    """
    Generates a weekly summary table for Bitcoin's key metrics.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical Bitcoin data, indexed by date.
    - report_date (str or pd.Timestamp): Specific date for which the summary is generated.

    Returns:
    - pd.DataFrame: DataFrame containing a summary of Bitcoin metrics for the specified report date.
    """

    # Extract the necessary data for the specified date from report_data
    report_date_data = report_data.loc[report_date].name
    bitcoin_supply = report_data.loc[report_date, "SplyCur"]
    hash_rate = report_data.loc[report_date, "7_day_ma_HashRate"]
    price_usd = report_data.loc[report_date, "PriceUSD"]
    market_cap = report_data.loc[report_date, "CapMrktCurUSD"]
    sats_per_dollar = 100000000 / price_usd  # Calculates satoshis per dollar
    btc_dominance = report_data.loc[report_date, "bitcoin_dominance"]
    btc_volume = report_data.loc[report_date, "btc_trading_volume"]

    # Placeholder values for market sentiment, trend, and valuation; can be dynamically assigned in the future
    fear_greed = "Neutral"  # Example sentiment
    bitcoin_trend = "Bullish"  # Example trend
    bitcoin_valuation = "Fair Value"  # Example valuation status

    # Create a dictionary to hold all the extracted metrics
    weekly_update_data = {
        "Bitcoin Price USD": price_usd,
        "Report Date": report_date_data,
        "Bitcoin Marketcap": market_cap,
        "Sats Per Dollar": sats_per_dollar,
        "Bitcoin Dominance": btc_dominance,
        "24HR Bitcoin Trading Volume": btc_volume,
        "Bitcoin Market Sentiment": fear_greed,
        "Bitcoin Market Trend": bitcoin_trend,
        "Bitcoin Valuation": bitcoin_valuation,
    }

    # Create a DataFrame from the dictionary for clear and structured reporting
    weekly_summary_df = pd.DataFrame([weekly_update_data])

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


def create_weekly_summary_big_numbers(weekly_summary_df):
    """
    Generates a series of BigNumbers for each metric in the weekly summary table, applying
    formatting rules based on metric type and indicating directional changes where applicable.

    Parameters:
    - weekly_summary_df (pd.DataFrame): DataFrame containing weekly summary metrics for Bitcoin.

    Returns:
    - dp.Group: A Datapane Group containing BigNumbers for display, arranged in 3 columns.
    """

    # Define formatting rules for each metric in the weekly summary
    format_rules = {
        "Bitcoin Price USD": "currency",
        "Report Date": "string",
        "Bitcoin Marketcap": "currency",
        "Sats Per Dollar": "float",
        "Bitcoin Dominance": "percentage",
        "24HR Bitcoin Trading Volume": "currency",
        "Bitcoin Market Sentiment": "string",
        "Bitcoin Market Trend": "string",
        "Bitcoin Valuation": "string",
    }

    # Initialize a list to store BigNumber elements
    big_numbers = []

    # Loop through each metric and apply formatting and directional styling if needed
    for column, value in weekly_summary_df.iloc[0].items():
        # Skip irrelevant or specifically excluded metrics
        if column == "Bitcoin Price Change Difficulty Period":
            continue

        # Format the metric value based on its type as per format_rules
        formatted_value = format_value(value, format_rules.get(column, ""))

        # Handle metrics with directional indication, assuming "Difficulty Change" implies a trend
        if column == "Difficulty Change":
            is_upward = value >= 0  # Positive values indicate upward trend
            big_numbers.append(
                dp.BigNumber(
                    heading=column,
                    value=formatted_value,
                    is_upward_change=is_upward,
                )
            )
        else:
            # Add BigNumber without directional indication for other metrics
            big_numbers.append(
                dp.BigNumber(
                    heading=column,
                    value=formatted_value,
                )
            )

    # Organize BigNumbers into a Group with 3 columns and return the result for display
    return dp.Group(*big_numbers, columns=3)


def create_crypto_performance_table(
    report_data, data, report_date, correlation_results
):
    """
    Creates a performance table summarizing key metrics for major cryptocurrencies, including
    Bitcoin, Ethereum, Ripple, Dogecoin, Binance Coin, and Tether. Metrics include price, market cap,
    returns over multiple periods, and correlations with Bitcoin.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for cryptocurrencies.
    - data (dict): Additional data related to cryptocurrencies (not directly used here but kept for compatibility).
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (pd.DataFrame): DataFrame with correlation values between cryptocurrencies and Bitcoin.

    Returns:
    - pd.DataFrame: A DataFrame containing the performance metrics for the selected cryptocurrencies.
    """

    # Define the structure and data sources for each cryptocurrency's performance metrics
    performance_metrics_dict = {
        "Bitcoin": {
            "Asset": "BTC",
            "Price": report_data.loc[report_date, "PriceUSD"],
            "Market Cap": report_data.loc[report_date, "CapMrktCurUSD"],
            "Week To Date Return": report_data.loc[
                report_date, "PriceUSD_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "PriceUSD_MTD_change"],
            "YTD": report_data.loc[report_date, "PriceUSD_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "PriceUSD"
            ],
        },
        "Ethereum": {
            "Asset": "ETH",
            "Price": report_data.loc[report_date, "ethereum_close"],
            "Market Cap": report_data.loc[report_date, "ethereum_market_cap"],
            "Week To Date Return": report_data.loc[
                report_date, "ethereum_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "ethereum_close_MTD_change"],
            "YTD": report_data.loc[report_date, "ethereum_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "ethereum_close"
            ],
        },
        "Ripple": {
            "Asset": "XRP",
            "Price": report_data.loc[report_date, "ripple_close"],
            "Market Cap": report_data.loc[report_date, "ripple_market_cap"],
            "Week To Date Return": report_data.loc[
                report_date, "ripple_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "ripple_close_MTD_change"],
            "YTD": report_data.loc[report_date, "ripple_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "ripple_close"
            ],
        },
        "Dogecoin": {
            "Asset": "DOGE",
            "Price": report_data.loc[report_date, "dogecoin_close"],
            "Market Cap": report_data.loc[report_date, "dogecoin_market_cap"],
            "Week To Date Return": report_data.loc[
                report_date, "dogecoin_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "dogecoin_close_MTD_change"],
            "YTD": report_data.loc[report_date, "dogecoin_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "dogecoin_close"
            ],
        },
        "Binance Coin": {
            "Asset": "BNB",
            "Price": report_data.loc[report_date, "binancecoin_close"],
            "Market Cap": report_data.loc[report_date, "binancecoin_market_cap"],
            "Week To Date Return": report_data.loc[
                report_date, "binancecoin_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "binancecoin_close_MTD_change"],
            "YTD": report_data.loc[report_date, "binancecoin_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "binancecoin_close"
            ],
        },
        "Tether": {
            "Asset": "USDT",
            "Price": report_data.loc[report_date, "tether_close"],
            "Market Cap": report_data.loc[report_date, "tether_market_cap"],
            "Week To Date Return": report_data.loc[
                report_date, "tether_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "tether_close_MTD_change"],
            "YTD": report_data.loc[report_date, "tether_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "tether_close"
            ],
        },
    }

    # Convert the dictionary into a DataFrame for easier data manipulation and display
    performance_table_df = pd.DataFrame(list(performance_metrics_dict.values()))

    return performance_table_df


def create_index_performance_table(report_data, data, report_date, correlation_results):
    """
    Creates a performance table summarizing key metrics for major financial indices and ETFs,
    including Bitcoin, Nasdaq, S&P500, XLF Financials ETF, XLE Energy ETF, and FANG+ ETF.
    Metrics include price, returns over multiple periods, and correlations with Bitcoin.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for various financial indices and ETFs.
    - data (dict): Additional data related to indices and ETFs (not directly used here but included for compatibility).
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (pd.DataFrame): DataFrame with correlation values between indices/ETFs and Bitcoin.

    Returns:
    - pd.DataFrame: A DataFrame containing the performance metrics for the selected indices and ETFs.
    """

    # Define the structure and data sources for each index/ETF's performance metrics
    performance_metrics_dict = {
        "Bitcoin": {
            "Asset": "Bitcoin",
            "Price": report_data.loc[report_date, "PriceUSD"],
            "Week To Date Return": report_data.loc[
                report_date, "PriceUSD_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "PriceUSD_MTD_change"],
            "YTD": report_data.loc[report_date, "PriceUSD_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "PriceUSD"
            ],
        },
        "Nasdaq": {
            "Asset": "Nasdaq",
            "Price": report_data.loc[report_date, "^IXIC_close"],
            "Week To Date Return": report_data.loc[
                report_date, "^IXIC_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "^IXIC_close_MTD_change"],
            "YTD": report_data.loc[report_date, "^IXIC_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "^IXIC_close"
            ],
        },
        "S&P500": {
            "Asset": "S&P500",
            "Price": report_data.loc[report_date, "^GSPC_close"],
            "Week To Date Return": report_data.loc[
                report_date, "^GSPC_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "^GSPC_close_MTD_change"],
            "YTD": report_data.loc[report_date, "^GSPC_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "^GSPC_close"
            ],
        },
        "XLF": {
            "Asset": "XLF Financials ETF",
            "Price": report_data.loc[report_date, "XLF_close"],
            "Week To Date Return": report_data.loc[
                report_date, "XLF_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "XLF_close_MTD_change"],
            "YTD": report_data.loc[report_date, "XLF_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "XLF_close"
            ],
        },
        "XLE": {
            "Asset": "XLE Energy ETF",
            "Price": report_data.loc[report_date, "XLE_close"],
            "Week To Date Return": report_data.loc[
                report_date, "XLE_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "XLE_close_MTD_change"],
            "YTD": report_data.loc[report_date, "XLE_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "XLE_close"
            ],
        },
        "FANG+": {
            "Asset": "FANG+ ETF",
            "Price": report_data.loc[report_date, "FANG.AX_close"],
            "Week To Date Return": report_data.loc[
                report_date, "FANG.AX_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "FANG.AX_close_MTD_change"],
            "YTD": report_data.loc[report_date, "FANG.AX_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "FANG.AX_close"
            ],
        },
    }

    # Convert the dictionary into a DataFrame for easier data manipulation and display
    performance_table_df = pd.DataFrame(list(performance_metrics_dict.values()))

    return performance_table_df


def create_macro_performance_table(report_data, data, report_date, correlation_results):
    """
    Generates a performance table summarizing key metrics for macroeconomic assets, including Bitcoin,
    the US Dollar Index, Gold Futures, Crude Oil Futures, 20+ Year Treasury Bond ETF, and the Bloomberg Commodity Index.
    Metrics include asset price, returns over multiple periods, and correlations with Bitcoin.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for various macroeconomic assets.
    - data (dict): Additional data related to assets (not directly used here but included for compatibility).
    - report_date (str or pd.Timestamp): The date for which the performance metrics are retrieved.
    - correlation_results (pd.DataFrame): DataFrame containing correlation values between assets and Bitcoin.

    Returns:
    - pd.DataFrame: A DataFrame containing performance metrics for the selected macroeconomic assets.
    """

    # Define the structure and data sources for each macroeconomic asset's performance metrics
    performance_metrics_dict = {
        "Bitcoin": {
            "Asset": "Bitcoin",
            "Price": report_data.loc[report_date, "PriceUSD"],
            "Week To Date Return": report_data.loc[
                report_date, "PriceUSD_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "PriceUSD_MTD_change"],
            "YTD": report_data.loc[report_date, "PriceUSD_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "PriceUSD"
            ],
        },
        "US Dollar Index": {
            "Asset": "US Dollar Index",
            "Price": report_data.loc[report_date, "DX=F_close"],
            "Week To Date Return": report_data.loc[
                report_date, "DX=F_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "DX=F_close_MTD_change"],
            "YTD": report_data.loc[report_date, "DX=F_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "DX=F_close"
            ],
        },
        "Gold Futures": {
            "Asset": "Gold Futures",
            "Price": report_data.loc[report_date, "GC=F_close"],
            "Week To Date Return": report_data.loc[
                report_date, "GC=F_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "GC=F_close_MTD_change"],
            "YTD": report_data.loc[report_date, "GC=F_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "GC=F_close"
            ],
        },
        "Crude Oil Futures": {
            "Asset": "Crude Oil Futures",
            "Price": report_data.loc[report_date, "CL=F_close"],
            "Week To Date Return": report_data.loc[
                report_date, "CL=F_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "CL=F_close_MTD_change"],
            "YTD": report_data.loc[report_date, "CL=F_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "CL=F_close"
            ],
        },
        "20+ Year Treasury Bond ETF": {
            "Asset": "20+ Year Treasury Bond ETF",
            "Price": report_data.loc[report_date, "TLT_close"],
            "Week To Date Return": report_data.loc[
                report_date, "TLT_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "TLT_close_MTD_change"],
            "YTD": report_data.loc[report_date, "TLT_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "TLT_close"
            ],
        },
        "Bloomberg Commodity Index": {
            "Asset": "Bloomberg Commodity Index",
            "Price": report_data.loc[report_date, "^BCOM_close"],
            "Week To Date Return": report_data.loc[
                report_date, "^BCOM_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, "^BCOM_close_MTD_change"],
            "YTD": report_data.loc[report_date, "^BCOM_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "^BCOM_close"
            ],
        },
    }

    # Convert the dictionary into a DataFrame for easier data manipulation and display
    macro_performance_table_df = pd.DataFrame(list(performance_metrics_dict.values()))

    return macro_performance_table_df


def create_equities_performance_table(
    report_data, data, report_date, correlation_results
):
    """
    Generates a performance table summarizing key metrics for Bitcoin and select equities
    involved in the cryptocurrency ecosystem. Metrics include asset price, market capitalization,
    returns over various periods, and correlations with Bitcoin.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for Bitcoin and equities.
    - data (pd.DataFrame): DataFrame containing additional data, such as market capitalization, for equities.
    - report_date (str or pd.Timestamp): The date for which the performance metrics are retrieved.
    - correlation_results (pd.DataFrame): DataFrame containing correlation values between Bitcoin and equities.

    Returns:
    - pd.DataFrame: A DataFrame containing performance metrics for Bitcoin and selected equities.
    """

    # Define a list of equity tickers in the crypto ecosystem (excluding Bitcoin)
    equities = ["COIN", "XYZ", "MSTR", "MARA", "RIOT"]

    # Initialize a dictionary to store performance metrics for each asset
    performance_metrics_dict = {}

    # Add Bitcoin's performance metrics with direct values
    performance_metrics_dict["Bitcoin"] = {
        "Asset": "Bitcoin",
        "Price": report_data.loc[report_date, "PriceUSD"],
        "Market Cap": report_data.loc[report_date, "CapMrktCurUSD"],
        "Week To Date Return": report_data.loc[
            report_date, "PriceUSD_trading_week_change"
        ],
        "MTD": report_data.loc[report_date, "PriceUSD_MTD_change"],
        "YTD": report_data.loc[report_date, "PriceUSD_YTD_change"],
        "90 Day BTC Correlation": 1.0,  # Bitcoin's correlation with itself is always 1
    }

    # Iterate over each equity in the list to collect its performance data
    for equity in equities:
        performance_metrics_dict[equity] = {
            "Asset": equity,
            "Price": report_data.loc[report_date, f"{equity}_close"],
            "Market Cap": data.loc[report_date, f"{equity}_MarketCap"],
            "Week To Date Return": report_data.loc[
                report_date, f"{equity}_close_trading_week_change"
            ],
            "MTD": report_data.loc[report_date, f"{equity}_close_MTD_change"],
            "YTD": report_data.loc[report_date, f"{equity}_close_YTD_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", f"{equity}_close"
            ],
        }

    # Convert the dictionary to a DataFrame for easier display and manipulation
    performance_table_df = pd.DataFrame(list(performance_metrics_dict.values()))

    return performance_table_df


def style_performance_table(performance_table):
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
        "Price": "${:,.2f}",
        "Week To Date Return": "{:.2%}",
        "MTD": "{:.2}%",
        "YTD": "{:.2}%",
        "90 Day BTC Correlation": "{:,.2f}",
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
    gradient_columns = ["Week To Date Return", "MTD", "YTD", "90 Day BTC Correlation"]

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


## New Tables


def create_bitcoin_model_table(report_data, report_date, cagr_results):
    """
    Constructs a table with various valuation metrics for Bitcoin based on financial models,
    comparing BTC against different market multiples and external asset price levels.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing Bitcoin and asset market data by date.
    - report_date (str or datetime): Specific date to extract metrics.
    - cagr_results (pd.DataFrame): DataFrame with calculated Compound Annual Growth Rate (CAGR) values.

    Returns:
    - pd.DataFrame: DataFrame with Bitcoin valuation metrics including market multiples,
      traditional assets, and country-level comparisons.
    """
    # Extract BTC Value
    btc_value = report_data.loc[report_date, "PriceUSD"]

    # Extraction for Values"
    four_year_cagr = (cagr_results.loc[report_date, "PriceUSD_4_Year_CAGR"],)
    sf_multiple = report_data.loc[report_date, "SF_Multiple"]
    day_200_price_multiple = report_data.loc[report_date, "200_day_multiple"]
    realized_price_multiple = report_data.loc[report_date, "mvrv_ratio"]
    thermocap_multiple = report_data.loc[report_date, "thermocap_multiple"]
    production_price_multiple = report_data.loc[report_date, "Energy_Value_Multiple"]

    # Extraction for "appl_marketcap"
    silver_price = (
        report_data.loc[report_date, "silver_marketcap_billion_usd"]
        / report_data.loc[report_date, "SplyExpFut10yr"]
    )
    gold_price_country = report_data.loc[
        report_date, "gold_official_country_holdings_marketcap_btc_price"
    ]
    gold_price_private = report_data.loc[
        report_date, "gold_private_investment_marketcap_btc_price"
    ]
    gold_price = (
        report_data.loc[report_date, "gold_marketcap_billion_usd"]
        / report_data.loc[report_date, "SplyExpFut10yr"]
    )
    META_price = report_data.loc[report_date, "META_mc_btc_price"]
    AMZN_price = report_data.loc[report_date, "AMZN_mc_btc_price"]
    GOOGL_price = report_data.loc[report_date, "GOOGL_mc_btc_price"]
    MSFT_price = report_data.loc[report_date, "MSFT_mc_btc_price"]
    AAPL_price = report_data.loc[report_date, "AAPL_mc_btc_price"]
    uk_price = report_data.loc[report_date, "United_Kingdom_btc_price"]
    japan_price = report_data.loc[report_date, "Japan_btc_price"]
    china_price = report_data.loc[report_date, "China_btc_price"]
    us_price = report_data.loc[report_date, "United_States_btc_price"]
    eu_price = report_data.loc[report_date, "Eurozone_btc_price"]

    # Update the dictionary with the extracted values
    bitcoin_model_data = {
        "Model": [
            "Bitcoin Price",
            "4 Year CAGR",
            "Stock-Flow Multiple",
            "200 Day Price Multiple",
            "Realized Price Multiple",
            "Thermocap Multiple",
            "Production Price Multiple",
            "Silver Price Level",
            "Gold Country Price Level",
            "Gold Private Ownership Price Level",
            "Total Gold Price Level",
            "META Price Level",
            "Amazon Price Level",
            "Google Price Level",
            "Microsoft Price Level",
            "Apple Price Level",
            "UK Price Level",
            "Japan Price Level",
            "China Price Level",
            "US Price Level",
            "EU Price Level",
        ],
        "Model Multiple / Value": [
            btc_value,
            four_year_cagr,
            sf_multiple,
            day_200_price_multiple,
            realized_price_multiple,
            thermocap_multiple,
            production_price_multiple,
            silver_price,
            gold_price_country,
            gold_price_private,
            gold_price,
            META_price,
            AMZN_price,
            GOOGL_price,
            MSFT_price,
            AAPL_price,
            uk_price,
            japan_price,
            china_price,
            us_price,
            eu_price,
        ],
    }

    # Create and return the "Bitcoin Valuation" DataFrame
    bitcoin_model_df = pd.DataFrame(bitcoin_model_data)

    return bitcoin_model_df


def style_bitcoin_model_table(bitcoin_model_table):
    """
    Applies styling to the Bitcoin model table to improve readability, including
    custom number formatting and conditional coloring.

    Parameters:
    - bitcoin_model_table (pd.DataFrame): DataFrame containing Bitcoin model values
      and multiple valuation metrics.

    Returns:
    - pd.io.formats.style.Styler: A styled DataFrame with formatted values and color styling.
    """
    # Define formatting rules for the columns in the model table
    format_dict_valuation = {
        "Valuation Model": "{}",
        "Model Multiple / Value": "{:,.0f}",
    }

    # Define custom colormaps for the table
    diverging_cm = sns.diverging_palette(100, 133, as_cmap=True)
    bg_colormap = sns.light_palette("white", as_cmap=True)

    def color_values(val):
        """
        Returns CSS color styling based on the sign of the value:
        - Green for positive values
        - Red for negative values
        - Black for zero or null values

        Parameters:
        - val (float or int): Numeric value to style.

        Returns:
        - str: CSS style string for color.
        """
        color = "green" if val > 0 else ("red" if val < 0 else "black")
        return f"color: {color}"

    # Apply the formatting rules and style configurations
    styled_table = (
        bitcoin_model_table.style.format(format_dict_valuation)
        .hide_index()  # Hide the DataFrame index for cleaner presentation
        .set_properties(**{"white-space": "nowrap"})  # Prevents content wrapping
        .set_table_styles([{"selector": "th", "props": [("white-space", "nowrap")]}])
    )

    return styled_table


## Weekly Bitcoin Recap Summary Tables


def create_summary_table_weekly_bitcoin_recap(report_data, report_date):
    """
    Generates a weekly summary table for Bitcoin's key metrics.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical Bitcoin data, indexed by date.
    - report_date (str or pd.Timestamp): Specific date for which the summary is generated.

    Returns:
    - pd.DataFrame: DataFrame containing a summary of Bitcoin metrics for the specified report date.
    """

    # Extract the necessary data for the specified date from report_data
    price_usd = report_data.loc[report_date, "PriceUSD"]
    market_cap = report_data.loc[report_date, "CapMrktCurUSD"]
    sats_per_dollar = 100000000 / price_usd  # Calculates satoshis per dollar

    bitcoin_supply = report_data.loc[report_date, "SplyCur"]
    miner_revenue_30d = report_data.loc[report_date, "30_day_ma_RevUSD"]
    tx_volume_30d = report_data.loc[report_date, "30_day_ma_TxTfrValAdjUSD"]
    btc_dominance = report_data["bitcoin_dominance"].iloc[-1]    # Placeholder values for market sentiment, trend, and valuation; can be dynamically assigned in the future
    fear_greed = "Neutral"  # Example sentiment
    bitcoin_valuation = "Fair Value"  # Example valuation status

    # Create a dictionary to hold all the extracted metrics
    weekly_update_data = {
        "Bitcoin Price USD": price_usd,
        "Bitcoin Marketcap": market_cap,
        "Sats Per Dollar": sats_per_dollar,
        "Bitcoin Supply": bitcoin_supply,
        "Bitcoin Miner Revenue": miner_revenue_30d,
        "Bitcoin Transaction Volume": tx_volume_30d,
        "Bitcoin Dominance": btc_dominance,
        "Bitcoin Market Sentiment": fear_greed,
        "Bitcoin Valuation": bitcoin_valuation,
    }

    # Create a DataFrame from the dictionary for clear and structured reporting
    weekly_summary_df = pd.DataFrame([weekly_update_data])

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


def create_summary_big_numbers_weekly_bitcoin_recap(weekly_summary_df):
    """
    Generates a series of BigNumbers for each metric in the weekly summary table, applying
    formatting rules based on metric type and indicating directional changes where applicable.

    Parameters:
    - weekly_summary_df (pd.DataFrame): DataFrame containing weekly summary metrics for Bitcoin.

    Returns:
    - dp.Group: A Datapane Group containing BigNumbers for display, arranged in 3 columns.
    """

    # Define formatting rules for each metric in the weekly summary
    format_rules = {
        "Bitcoin Price USD": "currency",
        "Bitcoin Marketcap": "currency",
        "Sats Per Dollar": "float",
        "Bitcoin Supply": "float",
        "Bitcoin Miner Revenue": "currency",
        "Bitcoin Transaction Volume": "currency",
        "Bitcoin Dominance": "percentage",
        "Bitcoin Market Sentiment": "string",
        "Bitcoin Valuation": "string",
    }

    # Initialize a list to store BigNumber elements
    big_numbers = []

    # Loop through each metric and apply formatting and directional styling if needed
    for column, value in weekly_summary_df.iloc[0].items():
        # Skip irrelevant or specifically excluded metrics
        if column == "Bitcoin Price Change Difficulty Period":
            continue

        # Format the metric value based on its type as per format_rules
        formatted_value = format_value(value, format_rules.get(column, ""))

        # Handle metrics with directional indication, assuming "Difficulty Change" implies a trend
        if column == "Difficulty Change":
            is_upward = value >= 0  # Positive values indicate upward trend
            big_numbers.append(
                dp.BigNumber(
                    heading=column,
                    value=formatted_value,
                    is_upward_change=is_upward,
                )
            )
        else:
            # Add BigNumber without directional indication for other metrics
            big_numbers.append(
                dp.BigNumber(
                    heading=column,
                    value=formatted_value,
                )
            )

    # Organize BigNumbers into a Group with 3 columns and return the result for display
    return dp.Group(*big_numbers, columns=3)


def create_equity_performance_table(report_data, report_date, correlation_results):
    """
    Creates a performance table summarizing key metrics for selected assets, including Bitcoin (BTC),
    SPY (S&P 500 ETF), QQQ (Nasdaq-100 ETF), VTI (Total Stock Market ETF), and VXUS (International Stock ETF).
    Metrics include price, 7-day return, MTD return, YTD return, 90-day return, and correlation with Bitcoin.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for the assets.
    - data (dict): Additional data related to assets (not directly used here but kept for compatibility).
    - report_date (str or pd.Timestamp): Date for which the performance metrics are retrieved.
    - correlation_results (pd.DataFrame): DataFrame with correlation values between assets and Bitcoin.

    Returns:
    - pd.DataFrame: A DataFrame containing the performance metrics for the selected assets.
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
        "SPY": {
            "Asset": "S&P 500 Index ETF - [SPY]",
            "Price": report_data.loc[report_date, "SPY_close"],
            "7 Day Return": report_data.loc[report_date, "SPY_close_7_change"],
            "MTD Return": report_data.loc[report_date, "SPY_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "SPY_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "SPY_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "SPY_close"
            ],
        },
        "QQQ": {
            "Asset": "Nasdaq-100 ETF - [QQQ]",
            "Price": report_data.loc[report_date, "QQQ_close"],
            "7 Day Return": report_data.loc[report_date, "QQQ_close_7_change"],
            "MTD Return": report_data.loc[report_date, "QQQ_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "QQQ_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "QQQ_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "QQQ_close"
            ],
        },
        "VTI": {
            "Asset": "US Total Stock Market ETF - [VTI]",
            "Price": report_data.loc[report_date, "VTI_close"],
            "7 Day Return": report_data.loc[report_date, "VTI_close_7_change"],
            "MTD Return": report_data.loc[report_date, "VTI_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "VTI_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "VTI_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "VTI_close"
            ],
        },
        "VXUS": {
            "Asset": "International Stock ETF - [VXUS]",
            "Price": report_data.loc[report_date, "VXUS_close"],
            "7 Day Return": report_data.loc[report_date, "VXUS_close_7_change"],
            "MTD Return": report_data.loc[report_date, "VXUS_close_MTD_change"],
            "YTD Return": report_data.loc[report_date, "VXUS_close_YTD_change"],
            "90 Day Return": report_data.loc[report_date, "VXUS_close_90_change"],
            "90 Day BTC Correlation": correlation_results["priceusd_90_days"].loc[
                "PriceUSD", "VXUS_close"
            ],
        },
    }

    # Convert the dictionary into a DataFrame for easier data manipulation and display
    performance_table_df = pd.DataFrame(list(performance_metrics_dict.values()))

    return performance_table_df


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
    gradient_columns = ["7 Day Return", "MTD Return", "YTD Return", "90 Day Return", "90 Day BTC Correlation"]

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
    difficulty_period_changes,
    report_date,
    weekly_high_low,
    cagr_results,
    sharpe_results,
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

def get_eoy_model_data(report_data, cagr_results):
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
    available_columns = [col for col in columns_of_interest if col in report_data.columns]
    available_cagr_columns = [col for col in cagr_columns if col in cagr_results.columns]

    # Extract the relevant data from both datasets
    report_data_filtered = report_data[available_columns]
    cagr_results_filtered = cagr_results[available_cagr_columns]

    # Merge both datasets on the index (assuming they share the same date index)
    full_data = report_data_filtered.merge(cagr_results_filtered, left_index=True, right_index=True, how="left")

    return full_data
