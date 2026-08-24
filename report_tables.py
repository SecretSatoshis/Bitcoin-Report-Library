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
    - Temporal Analysis: Monthly/yearly return comparisons, OHLC exports, heatmaps
    - Valuation Models: Relative value comparisons and end-of-year projections

Output Format:
    All functions return unstyled pandas DataFrames ready for CSV export. Formatting is
    minimal and data-focused to ensure compatibility with downstream analysis tools.
"""

import pandas as pd
from datetime import timedelta
import numpy as np
from pandas.tseries.offsets import MonthEnd
import calendar
from data_definitions import ELECTRICITY_TARIFFS_USD_PER_KWH, SATS_PER_BTC


def create_network_model_metrics(report_data, report_date=None):
    """Build the detailed Metcalfe, power-law, and hash-ribbon export."""
    columns = {
        "price_close": "BTC Price",
        "market_cap": "Bitcoin Market Cap",
        "supply": "Bitcoin Supply",
        "addr_count": "Non-Zero Address Count",
        "addrs_over_100k_sats_addr_count": "Addresses Holding 0.001+ BTC",
        "addrs_over_1m_sats_addr_count": "Addresses Holding 0.01+ BTC",
        "addrs_over_10m_sats_addr_count": "Addresses Holding 0.1+ BTC",
        "metcalfe_value": "Metcalfe Value (Any Balance)",
        "metcalfe_value_0p001_btc": "Metcalfe Value (0.001+ BTC)",
        "metcalfe_value_0p01_btc": "Metcalfe Value (0.01+ BTC)",
        "metcalfe_value_0p1_btc": "Metcalfe Value (0.1+ BTC)",
        "metcalfe_scale_any_balance": "Metcalfe Scale (Any Balance)",
        "metcalfe_scale_0p001_btc": "Metcalfe Scale (0.001+ BTC)",
        "metcalfe_scale_0p01_btc": "Metcalfe Scale (0.01+ BTC)",
        "metcalfe_scale_0p1_btc": "Metcalfe Scale (0.1+ BTC)",
        "metcalfe_price_multiple": "BTC Price / Metcalfe Value",
        "power_law_price": "Power Law Price",
        "power_law_price_multiple": "BTC Price / Power Law Price",
        "days_since_genesis": "Days Since Genesis",
        "power_law_exponent": "Power Law Exponent",
        "power_law_scale": "Power Law Scale",
        "hash_rate": "Hash Rate (H/s)",
        "30_day_ma_hash_rate": "Hash Rate 30-Day MA (H/s)",
        "60_day_ma_hash_rate": "Hash Rate 60-Day MA (H/s)",
        "hash_ribbon_ratio": "Hash Ribbon 30D / 60D",
        "hash_ribbon_capitulation": "Hash Ribbon Capitulation",
    }
    missing = sorted(set(columns).difference(report_data.columns))
    if missing:
        raise ValueError(f"Network model export is missing required columns: {missing}")

    result = report_data.loc[:, list(columns)].copy()
    if report_date is not None:
        result = result.loc[: pd.to_datetime(report_date).normalize()]
    result = result.loc[pd.to_numeric(result["price_close"], errors="coerce") > 0]
    result = result.rename(columns=columns)
    result.index.name = "date"
    return result


def create_electricity_cost_scenarios(report_data, report_date=None):
    """Build the daily power-expense scenario export.

    The scenario columns price one shared network-energy estimate at several
    electricity tariffs and divide by observed BTC paid to miners (subsidy plus
    fees). Legacy PUE/subsidy-only, production-cost, Hayes, and Energy Value
    series are included for comparison but retain their distinct definitions.
    """
    required = {
        "price_close",
        "cm_efficiency_j_gh",
        "network_power_watts",
        "daily_electricity_consumption_kwh",
        "subsidy_sum_24h",
        "fees_sum_24h",
        "miner_revenue_btc",
        "power_only_breakeven_tariff_usd_per_kwh",
        "Electricity_Cost_PUE_Subsidy_Only",
        "Bitcoin_Production_Cost",
        "Hayes_Network_Price_Per_BTC",
        "Energy_Value",
    }
    tariff_columns = {
        f"Electricity_Cost_{int(round(tariff * 100))}c":
        f"Power Expense (${tariff:.2f}/kWh)"
        for tariff in ELECTRICITY_TARIFFS_USD_PER_KWH
    }
    required.update(tariff_columns)
    missing = sorted(required.difference(report_data.columns))
    if missing:
        raise ValueError(
            f"Electricity scenario input is missing required columns: {missing}"
        )

    columns = {
        "price_close": "BTC Price",
        "cm_efficiency_j_gh": "Fleet Efficiency (J/GH)",
        "network_power_watts": "Network Power Draw (W)",
        "daily_electricity_consumption_kwh": "Daily Electricity Consumption (kWh)",
        "subsidy_sum_24h": "Subsidy (BTC)",
        "fees_sum_24h": "Fees (BTC)",
        "miner_revenue_btc": "Miner Revenue (BTC)",
        **tariff_columns,
        "power_only_breakeven_tariff_usd_per_kwh":
        "Power-Only Break-Even Tariff ($/kWh)",
        "Electricity_Cost_PUE_Subsidy_Only": "Legacy PUE/Subsidy-Only Cost",
        "Bitcoin_Production_Cost": "Bitcoin Production Cost",
        "Hayes_Network_Price_Per_BTC": "Hayes Network Price",
        "Energy_Value": "Energy Value",
    }
    result = report_data.loc[:, list(columns)].copy()
    if report_date is not None:
        result = result.loc[: pd.to_datetime(report_date).normalize()]
    result = result.dropna(
        subset=["price_close", "daily_electricity_consumption_kwh", "miner_revenue_btc"]
    ).rename(columns=columns)
    result.index.name = "date"
    return result


def calculate_price_buckets(data, bucket_size, report_date=None):
    """
    Calculates the number of unique trading days the price spent in each bucket range.

    Parameters:
    data (pd.DataFrame): DataFrame with DateTime index and a 'price_close' column.
    bucket_size (int): The size of each price bucket.
    report_date (str or datetime, optional): As-of date. Rows after this date are
                                             excluded from both counts and the
                                             current-price marker.

    Returns:
    pd.DataFrame: DataFrame containing counts of unique trading days in each price bucket.
    """
    if bucket_size <= 0:
        raise ValueError("bucket_size must be greater than zero.")

    # Ensure the DataFrame is sorted by time without mutating the caller's data.
    data = data.sort_index(ascending=True).copy()

    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")

    if report_date is not None:
        report_date = pd.to_datetime(report_date).normalize()
        data = data.loc[data.index <= report_date]

    # Zeroes in the source are pre-price-history placeholders, not trading prices.
    data = data.dropna(subset=["price_close"])
    data = data.loc[data["price_close"] > 0]
    if data.empty:
        raise ValueError(
            "No positive price data is available on or before the report date."
        )

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

    # Create a DataFrame for bucket counts with formatted price ranges
    bucket_counts_df = bucket_days_count.reset_index()
    bucket_counts_df.columns = ["Price Range Interval", "Count"]
    bucket_counts_df["Is Current Bucket"] = (
        bucket_counts_df["Price Range Interval"] == current_bucket
    )
    bucket_counts_df["Current Price"] = current_price
    bucket_counts_df["Price Range ($)"] = bucket_counts_df["Price Range Interval"].apply(
        lambda x: f"${int(x.left / 1000)}K-${int(x.right / 1000)}K"
    )
    bucket_counts_df = bucket_counts_df[
        ["Price Range ($)", "Count", "Is Current Bucket", "Current Price"]
    ]

    return bucket_counts_df


def _positive_price_series(price_series):
    """Return sorted, daily, positive prices without changing the source object."""
    prices = pd.to_numeric(price_series, errors="coerce").sort_index()
    prices = prices.dropna().loc[lambda values: values > 0]
    return prices.groupby(prices.index.normalize()).last()


def _last_positive_before(price_series, boundary):
    """Return the final positive price strictly before a calendar boundary."""
    prior = price_series.loc[price_series.index < pd.Timestamp(boundary)]
    return prior.iloc[-1] if not prior.empty else np.nan


def create_indexed_returns_history(
    price_series, report_date, period, min_year=2014
):
    """Create a wide, price-indexed MTD or YTD history through an as-of date.

    Each calendar year's period return is measured from the final positive close
    strictly before its month/year boundary, then rebased to the current period's
    equivalent boundary close. Row 0 records that shared baseline, preserving the
    first day's move at row 1. Historical years remain complete while the current
    year is capped at ``report_date``. YTD rows use a common 365-day calendar ordinal:
    February 29 is omitted and dates after it are shifted back one position, keeping
    the same month/day aligned across leap and common years.

    Parameters:
    price_series (pd.Series): Daily prices with a DatetimeIndex.
    report_date (str or datetime): Canonical report as-of date.
    period (str): Either ``"mtd"`` or ``"ytd"``.
    min_year (int): Earliest calendar year to include.

    Returns:
    pd.DataFrame: Year columns plus historical Median and Average columns, indexed by
                  day of month (MTD) or common day of year (YTD).
    """
    if not isinstance(price_series, pd.Series):
        raise TypeError("price_series must be a pandas Series.")
    if not isinstance(price_series.index, pd.DatetimeIndex):
        raise ValueError("price_series index must be a DatetimeIndex.")

    period = period.lower()
    if period not in {"mtd", "ytd"}:
        raise ValueError("period must be either 'mtd' or 'ytd'.")

    report_date = pd.to_datetime(report_date).normalize()
    index_name = "day" if period == "mtd" else "day_of_year"

    prices = _positive_price_series(price_series)
    prices = prices.loc[prices.index <= report_date]
    if prices.empty:
        empty = pd.DataFrame()
        empty.index.name = index_name
        return empty

    # Collapse any duplicate/intraday rows so each calendar position has one value.
    prices = prices.groupby(prices.index.normalize()).last()
    current_year = report_date.year
    current_month = report_date.month

    def period_for_year(year):
        mask = prices.index.year == year
        if period == "mtd":
            mask &= prices.index.month == current_month
        selected = prices.loc[mask]
        if period == "ytd":
            selected = selected.loc[
                ~((selected.index.month == 2) & (selected.index.day == 29))
            ]
        return selected

    def boundary_for_year(year):
        month = current_month if period == "mtd" else 1
        return pd.Timestamp(year=year, month=month, day=1)

    current_period = period_for_year(current_year)
    base_price = _last_positive_before(
        prices, boundary_for_year(current_year)
    )
    if current_period.empty or pd.isna(base_price):
        empty = pd.DataFrame()
        empty.index.name = index_name
        return empty

    indexed_years = {}
    for year in sorted(prices.index.year.unique()):
        if year < min_year:
            continue

        year_period = period_for_year(year)
        if year_period.empty:
            continue
        period_start_price = _last_positive_before(
            prices, boundary_for_year(year)
        )
        if pd.isna(period_start_price):
            continue

        indexed = (year_period / period_start_price) * base_price
        if period == "mtd":
            indexed.index = indexed.index.day
        else:
            common_ordinal = indexed.index.dayofyear.to_numpy(copy=True)
            after_february = indexed.index.month > 2
            common_ordinal[
                indexed.index.is_leap_year & after_february
            ] -= 1
            indexed.index = common_ordinal
        indexed_years[str(year)] = pd.concat(
            [pd.Series([base_price], index=[0]), indexed]
        )

    result = pd.DataFrame(indexed_years).sort_index()
    result.index.name = index_name
    historical_columns = [
        column for column in result.columns if int(column) < current_year
    ]
    if historical_columns:
        result["Median"] = result[historical_columns].median(axis=1)
        result["Average"] = result[historical_columns].mean(axis=1)
    return result


def create_summary_history(
    report_data, report_date, metrics, comparison_days=30
):
    """Create long-form daily metric history over an inclusive calendar window.

    A 30-day comparison requires both endpoints, so a complete daily source produces
    31 observations per metric. The window is anchored to the latest available row on
    or before ``report_date`` and never includes later data.
    """
    if not isinstance(report_data.index, pd.DatetimeIndex):
        raise ValueError("report_data index must be a DatetimeIndex.")
    if comparison_days < 0:
        raise ValueError("comparison_days must be non-negative.")

    report_date = pd.to_datetime(report_date).normalize()
    available_dates = report_data.index[report_data.index <= report_date]
    if len(available_dates) == 0:
        return pd.DataFrame(columns=["Metric", "date", "Value"])

    end_date = available_dates.max()
    start_date = end_date.normalize() - pd.Timedelta(days=comparison_days)
    history = report_data.loc[
        (report_data.index >= start_date) & (report_data.index <= end_date)
    ]

    rows = []
    for label, column in metrics.items():
        if column not in history.columns:
            continue
        for date_index, value in history[column].dropna().items():
            rows.append(
                {
                    "Metric": label,
                    "date": date_index.strftime("%Y-%m-%d"),
                    "Value": value,
                }
            )
    return pd.DataFrame(rows, columns=["Metric", "date", "Value"])


def calculate_roi_table(data, report_date, price_column="price_close"):
    """
    Calculates the return on investment (ROI) for Bitcoin over various time frames from the report date.

    Parameters:
    data (pd.DataFrame): DataFrame containing price data with a DateTime index.
    report_date (str or datetime): The date for which to calculate ROI.
    price_column (str): The column name for Bitcoin price data.

    Returns:
    pd.DataFrame: DataFrame containing Time Frame, ROI (%), Start Date, and BTC Price.
    """
    if price_column not in data.columns:
        raise ValueError(
            f"The price column '{price_column}' does not exist in the data."
        )

    if data.empty:
        raise ValueError("The input data is empty.")

    period_offsets = {
        "1 day": pd.DateOffset(days=1),
        "3 day": pd.DateOffset(days=3),
        "7 day": pd.DateOffset(days=7),
        "30 day": pd.DateOffset(days=30),
        "90 day": pd.DateOffset(days=90),
        "1 Year": pd.DateOffset(years=1),
        "2 Year": pd.DateOffset(years=2),
        "4 Year": pd.DateOffset(years=4),
        "5 Year": pd.DateOffset(years=5),
        "10 Year": pd.DateOffset(years=10),
    }

    data = data.sort_index()
    report_date = pd.to_datetime(report_date).normalize()
    available_dates = data.index[data.index <= report_date]
    if len(available_dates) == 0:
        raise ValueError("No data available on or before the report date.")
    current_date = available_dates.max()
    current_price = data.loc[current_date, price_column]

    # Pre-compute the 'Start Date' and 'BTC Price' for each period
    start_dates = {
        period: current_date - offset
        for period, offset in period_offsets.items()
    }

    btc_prices = {}
    roi_data = {}
    for period, start_date in start_dates.items():
        prior_dates = data.index[data.index <= start_date]
        if len(prior_dates) == 0:
            btc_prices[period] = None
            roi_data[period] = np.nan
            continue

        actual_start_date = prior_dates.max()
        start_price = data.loc[actual_start_date, price_column]
        btc_prices[period] = start_price
        roi_data[period] = (
            ((current_price / start_price) - 1) * 100
            if pd.notna(start_price) and start_price != 0
            else np.nan
        )
        start_dates[period] = actual_start_date

    # Combine the ROI, Start Dates, and BTC Prices into a DataFrame
    roi_table = pd.DataFrame(
        {
            "Time Frame": period_offsets.keys(),
            "ROI (%)": [roi_data[period] for period in period_offsets],
            "Start Date": [start_dates[period] for period in period_offsets],
            "BTC Price": [btc_prices[period] for period in period_offsets],
        }
    )
    return roi_table


def _format_fundamental_value(value, format_type):
    """Format fundamentals table values for report-ready CSV output."""
    if pd.isna(value):
        return ""

    if format_type == "currency":
        return f"${value:,.0f}"
    if format_type == "hashrate_ehs":
        return f"{value / 1e18:,.2f} EH/s"
    if format_type == "difficulty_t":
        return f"{value / 1e12:,.2f}T"
    if format_type == "percent_ratio":
        return f"{value * 100:.2f}%"
    if format_type in {"percent", "percent_point"}:
        return f"{value:.2f}%"
    if format_type == "number2":
        return f"{value:,.2f}"
    if format_type == "number":
        return f"{value:,.0f}"

    return str(value)


def create_fundamentals_table(df, metrics_template, report_date=None):
    """
    Generates a fundamentals metrics table with current/prior values, week-over-week
    change, daily Monday–Sunday breakdown, and 52-week range for each metric.

    Each metric is grouped by section (Network Performance, Network Security, etc.).
    Pre-formatted strings are used for value columns (since metrics have varied
    format types — number, currency, percent), while the 7 Day % Change is kept
    numeric so the dashboard can color-code it as a delta.

    Parameters:
    df (pd.DataFrame): DataFrame with DatetimeIndex containing all columns specified
                       in metrics_template. Must have at least 14 days of data.
    metrics_template (dict): {section_name: {metric_label: (column_name, format_type)}}
    report_date (str or datetime, optional): As-of date. Anchors this table to the same
                       row every other report table uses; without it the table reads the
                       latest available row and can publish a different "current" price
                       than summary/performance in the same run.

    Returns:
    pd.DataFrame: Columns:
        - Section: Group label (Network Performance, etc.)
        - Metric: Display name
        - Current Value: Report-date formatted value
        - 7 Days Ago: Value 7 calendar days before the report date
        - 7 Day Change (%): (current / 7 days ago - 1) * 100, in percentage points
                            (e.g. 6.83 = 6.83%)
        - Monday..Sunday: Daily values for the report week
        - 52W Low: Min over last 365 days
        - 52W High: Max over last 365 days
    """
    table_data = []

    df = df.sort_index()
    if report_date is not None:
        df = df.loc[: pd.to_datetime(report_date).normalize()]

    latest_date = df.index.max()
    start_of_week = latest_date - timedelta(days=latest_date.weekday())
    weekly_index = pd.date_range(start=start_of_week.normalize(), periods=7, freq="D")

    for section, metrics in metrics_template.items():
        for metric_display_name, (column_name, format_type) in metrics.items():
            series = df[column_name].dropna()
            if len(series) == 0:
                continue

            current = series.iloc[-1]

            # Look the prior value up by calendar date, not by row offset: after dropna()
            # a row offset of 8 means "8 observations back", which drifts away from seven
            # days whenever a metric has gaps.
            prior_dates = series.index[series.index <= series.index[-1] - timedelta(days=7)]
            seven_days_ago = series.loc[prior_dates[-1]] if len(prior_dates) else np.nan

            # Spot-to-spot change between the two values displayed beside it. This column
            # previously compared a 7-day mean against the prior 7-day mean, which is a
            # legitimate statistic but not the delta of the adjacent columns — it disagreed
            # in sign on noisy series (e.g. transaction count up 13% spot, mean down 0.5%)
            # while being rendered as those columns' delta.
            # Percentage points, matching every other published "(%)" column
            # (Return (%), ROI (%), drawdown_pct). See csv/SCHEMA.md.
            pct_change = (
                ((current / seven_days_ago) - 1) * 100
                if pd.notna(seven_days_ago) and seven_days_ago != 0
                else np.nan
            )

            year_window = series.tail(365)
            low_52w = year_window.min()
            high_52w = year_window.max()

            weekly_values = df[column_name].reindex(weekly_index).tolist()

            table_data.append(
                {
                    "Section": section,
                    "Metric": metric_display_name,
                    "Current Value": _format_fundamental_value(current, format_type),
                    "7 Days Ago": _format_fundamental_value(seven_days_ago, format_type),
                    "7 Day Change (%)": pct_change,
                    "Monday": _format_fundamental_value(weekly_values[0], format_type),
                    "Tuesday": _format_fundamental_value(weekly_values[1], format_type),
                    "Wednesday": _format_fundamental_value(weekly_values[2], format_type),
                    "Thursday": _format_fundamental_value(weekly_values[3], format_type),
                    "Friday": _format_fundamental_value(weekly_values[4], format_type),
                    "Saturday": _format_fundamental_value(weekly_values[5], format_type),
                    "Sunday": _format_fundamental_value(weekly_values[6], format_type),
                    "52W Low": _format_fundamental_value(low_52w, format_type),
                    "52W High": _format_fundamental_value(high_52w, format_type),
                }
            )

    return pd.DataFrame(table_data)


## Summary and Performance Tables


def _row_asof(df, report_date):
    report_date = pd.to_datetime(report_date).normalize()
    df = df.sort_index()
    available_dates = df.index[df.index <= report_date]
    if len(available_dates) == 0:
        raise ValueError("No data available on or before the report date.")
    return df.loc[available_dates.max()]


def _classify_fear_greed(value):
    if pd.isna(value):
        return ""
    if value <= 24:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 54:
        return "Neutral"
    if value <= 74:
        return "Greed"
    return "Extreme Greed"


def _classify_bitcoin_valuation(mvrv_ratio):
    if pd.isna(mvrv_ratio):
        return ""
    if mvrv_ratio < 1:
        return "Undervalued"
    if mvrv_ratio < 2:
        return "Fair Value"
    if mvrv_ratio < 3:
        return "Overvalued"
    return "Extremely Overvalued"


def create_summary_table(report_data, report_date):
    """
    Generates a summary table for Bitcoin's key metrics with categorized column headers.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical Bitcoin data, indexed by date.
    - report_date (str or pd.Timestamp): Specific date for which the summary is generated.

    Returns:
    - pd.DataFrame: DataFrame containing a summary of Bitcoin metrics for the specified report date.
    """

    latest = _row_asof(report_data, report_date)

    # Extract key metrics from report_data
    price_usd = latest["price_close"]
    market_cap = latest["market_cap"]
    sats_per_dollar = SATS_PER_BTC / price_usd

    bitcoin_supply = latest["supply"]
    # These labels also identify raw daily series in summary_history.csv. Keep the
    # snapshot definition identical instead of silently substituting a 30-day mean.
    miner_revenue = latest["coinbase_sum_24h_usd"]
    tx_volume = latest["transfer_volume_sum_24h_usd"]
    btc_dominance = latest.get("bitcoin_dominance", np.nan)
    if pd.isna(btc_dominance):
        raise RuntimeError(
            "Bitcoin dominance is required for the report-date summary snapshot"
        )

    fear_greed_value = latest.get("fear_greed_value", np.nan)
    fear_greed = latest.get("fear_greed_classification", "")
    if pd.isna(fear_greed) or not fear_greed:
        fear_greed = _classify_fear_greed(fear_greed_value)
    bitcoin_valuation = _classify_bitcoin_valuation(latest.get("mvrv_ratio", np.nan))

    # Define categories for organization
    categorized_data = {
        "Market Data": {
            "Bitcoin Price USD": price_usd,
            "Bitcoin Marketcap": market_cap,
            "Sats Per Dollar": sats_per_dollar,
        },
        "On-chain Data": {
            "Bitcoin Supply": bitcoin_supply,
            "Bitcoin Miner Revenue": miner_revenue,
            "Bitcoin Transaction Volume": tx_volume,
        },
        "Investor Sentiment": {
            "Bitcoin Dominance": btc_dominance,
            "Bitcoin Fear & Greed Index": fear_greed_value,
            "Bitcoin Market Sentiment": fear_greed,
            "Bitcoin Valuation": bitcoin_valuation,
        },
    }

    summary_rows = []
    for category, metrics in categorized_data.items():
        for metric, value in metrics.items():
            summary_rows.append(
                {"Metric": metric, "Value": value, "Category": category}
            )

    weekly_summary_df = pd.DataFrame(summary_rows)

    return weekly_summary_df


def _build_performance_table(
    report_data: pd.DataFrame,
    report_date,
    correlation_results: dict,
    asset_configs: list,
    category: str,
) -> pd.DataFrame:
    """
    Generic performance table builder for any asset category.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing historical data for all assets.
    - report_date (str or pd.Timestamp): As-of cutoff; metrics use the newest
      available row on or before this date.
    - correlation_results (dict): Dictionary with correlation DataFrames for different periods.
    - asset_configs (list): List of dicts with 'name', 'label', 'ticker' keys.
                            Example: [{"name": "BTC", "label": "Bitcoin - [BTC]", "ticker": "price_close"}]

    Returns:
    - pd.DataFrame: Performance metrics for the specified assets.
    """
    performance_metrics = []

    # Resolve the as-of row once. Market closures or an upstream gap can leave no
    # exact report-date label; every current value and the 52-week window must then
    # use the same newest observation on or before the requested cutoff.
    report_date = pd.to_datetime(report_date).normalize()
    report_data = report_data.sort_index()
    available_dates = report_data.index[report_data.index <= report_date]
    if len(available_dates) == 0:
        raise ValueError("No data available on or before the report date.")
    actual_report_date = available_dates.max()
    latest = report_data.loc[actual_report_date]
    if isinstance(latest, pd.DataFrame):
        latest = latest.iloc[-1]

    # 52-week window for high/low calculations, anchored to that same as-of row.
    year_ago = actual_report_date - pd.Timedelta(days=365)

    for config in asset_configs:
        ticker = config["ticker"]
        # Handle special case for Bitcoin price_close column
        price_col = ticker if ticker == "price_close" else f"{ticker}_close"
        corr_col = ticker if ticker == "price_close" else f"{ticker}_close"

        # Compute 52-week high/low from the last 365 days of close prices
        window = report_data.loc[year_ago:actual_report_date, price_col].dropna()
        high_52w = window.max() if len(window) else None
        low_52w = window.min() if len(window) else None

        metrics = {
            "Category": category,
            "Asset": config["label"],
            "Price": latest[price_col],
            "7 Day Return (%)": latest[f"{price_col}_7_change"],
            "MTD Return (%)": latest[f"{price_col}_MTD_change"],
            "YTD Return (%)": latest[f"{price_col}_YTD_change"],
            "90 Day Return (%)": latest[f"{price_col}_90_change"],
            "52 Week High": high_52w,
            "52 Week Low": low_52w,
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
    return _build_performance_table(
        report_data,
        report_date,
        correlation_results,
        asset_configs,
        "Equity Market Indexes",
    )


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
    return _build_performance_table(
        report_data,
        report_date,
        correlation_results,
        asset_configs,
        "Sectors",
    )


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
        {"name": "DXY", "label": "US Dollar Index - [DXY]", "ticker": "DX-Y.NYB"},
        {"name": "GLD", "label": "Gold ETF - [GLD]", "ticker": "GLD"},
        {"name": "AGG", "label": "Aggregate Bond ETF - [AGG]", "ticker": "AGG"},
        {
            "name": "SPGSCI",
            "label": "S&P GSCI Commodity Index - [SPGSCI]",
            "ticker": "^SPGSCI",
        },
    ]
    return _build_performance_table(
        report_data,
        report_date,
        correlation_results,
        asset_configs,
        "Macro Asset Classes",
    )


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
    return _build_performance_table(
        report_data,
        report_date,
        correlation_results,
        asset_configs,
        "Bitcoin Industry Performance",
    )


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


def monthly_heatmap(data, report_date=None, export_csv=True):
    """
    Creates monthly and yearly Bitcoin returns heatmap data with statistical aggregations.

    This function generates a matrix of monthly returns organized by year (rows) and month (columns),
    with an additional yearly return column. It includes statistical rows (4-year average, median,
    average) and handles incomplete current month data by calculating month-to-date returns.

    Parameters:
    data (pd.DataFrame): DataFrame with DatetimeIndex and 'price_close' column. Data is filtered
                         to start from 2012-01-01 within the function.
    report_date (str or datetime, optional): As-of date used to cap current-period returns.
    export_csv (bool): If True, exports heatmap data to csv/monthly_heatmap_data.csv. Default: True.

    Returns:
    pd.DataFrame: Heatmap matrix with:
        - Rows: Years (2012+), plus "4-Year Average", "Median", "Average"
        - Columns: Month names (Jan-Dec) plus "Yearly"
        - Values: Percentage points (5.0 = 5% gain, -3.0 = 3% loss)
        - Yearly matches ytd_return_comparison.csv (prior year close -> latest)
        - Current incomplete month shows MTD return
        - Statistical rows exclude incomplete current-period data
    """
    data = data.sort_index()
    if report_date is not None:
        report_date = pd.to_datetime(report_date).normalize()
        data = data.loc[:report_date]

    # Retain pre-2012 prices for the January 2012/year-2012 boundary lookup, but only
    # publish period rows from 2012 onward.
    all_prices = _positive_price_series(data["price_close"])
    display_prices = all_prices.loc[all_prices.index >= pd.Timestamp("2012-01-01")]
    if display_prices.empty:
        raise ValueError("No positive price data is available from 2012 onward.")

    monthly_returns = {}
    for (year, month), month_prices in display_prices.groupby(
        [display_prices.index.year, display_prices.index.month]
    ):
        boundary = pd.Timestamp(year=year, month=month, day=1)
        prior_month_close = _last_positive_before(all_prices, boundary)
        if pd.notna(prior_month_close):
            monthly_returns[(year, month)] = (
                month_prices.iloc[-1] / prior_month_close
            ) - 1

    heatmap_data = pd.Series(monthly_returns).unstack().reindex(columns=range(1, 13))

    # Get the last date in the data to check if the current month is complete
    last_date = display_prices.index[-1]
    current_year, current_month = last_date.year, last_date.month

    # Check if the current month is incomplete
    is_incomplete_month = last_date.day != (last_date + MonthEnd(0)).day

    # Yearly uses the same prior-calendar-close denominator as the monthly cells and
    # ytd_return_comparison.csv. Consequently, compounding a complete year's monthly
    # cells now agrees with its Yearly value.
    yearly_returns = {}
    for year, year_prices in display_prices.groupby(display_prices.index.year):
        prior_year_close = _last_positive_before(
            all_prices, pd.Timestamp(year=year, month=1, day=1)
        )
        if pd.notna(prior_year_close):
            yearly_returns[year] = (year_prices.iloc[-1] / prior_year_close) - 1
    heatmap_data[13] = pd.Series(yearly_returns)

    # Create a copy excluding incomplete current-period data for the statistical rows
    heatmap_data_excluded = heatmap_data.copy()
    if current_year in heatmap_data.index:
        if is_incomplete_month:
            heatmap_data_excluded.loc[current_year, current_month] = pd.NA
        # The current year is unfinished by definition, so its partial yearly figure must
        # not be averaged in with completed years.
        heatmap_data_excluded.loc[current_year, 13] = pd.NA

    # Add the "4-Year Average" row — the four most recent years that actually have data
    # for each column. Slicing the last four *rows* instead averages only three values
    # for any month the current year has not reached yet, while still labelling the
    # result a four-year average.
    heatmap_data.loc["4-Year Average"] = heatmap_data_excluded.apply(
        lambda col: col.dropna().tail(4).mean(), axis=0
    )

    # Add the "Median" row, excluding the incomplete month
    heatmap_data.loc["Median"] = heatmap_data_excluded.apply(
        lambda col: col[~col.isna()].median(), axis=0
    )

    # Add the "Average" row, excluding the incomplete month
    heatmap_data.loc["Average"] = heatmap_data_excluded.apply(
        lambda col: col[~col.isna()].mean(), axis=0
    )

    # Publish in percentage points, matching every other percentage column in csv/.
    heatmap_data = heatmap_data * 100

    # Rename columns to month names
    month_names = [calendar.month_abbr[i] for i in range(1, 13)] + ["Yearly"]
    heatmap_data.columns = month_names
    heatmap_data.index.name = "time"

    # Optionally export the heatmap data to CSV
    if export_csv:
        heatmap_data.to_csv("csv/monthly_heatmap_data.csv")

    return heatmap_data


## CSV Exports


def calculate_ohlc(ohlc_data, output_file="csv/ohlc_data.csv"):
    """
    Saves BRK weekly OHLC data to CSV.

    BRK week1 rows are week-start labels and include the latest available
    current-week candle. For an open candle, Close represents the latest
    available price, not a finalized weekly close.

    Parameters:
    ohlc_data (pd.DataFrame): DataFrame with DatetimeIndex and columns: 'Open', 'High', 'Low', 'Close'.
                              Index must be datetime-compatible for export.
    output_file (str): Path for CSV export. Default: "csv/ohlc_data.csv"

    Returns:
    pd.DataFrame: Weekly OHLC DataFrame with columns:
        - Open: First open price of the week
        - High: Highest price during the week
        - Low: Lowest price during the week
        - Close: Last close price of the week
        Index is BRK week-start date labels.
    """
    required_columns = ["Open", "High", "Low", "Close"]
    if ohlc_data is None or ohlc_data.empty:
        raise ValueError("Weekly OHLC data is empty; refusing to overwrite output")
    missing = [column for column in required_columns if column not in ohlc_data.columns]
    if missing:
        raise ValueError(
            f"Weekly OHLC data is missing required columns {missing}; refusing to overwrite output"
        )

    # Ensure the index is a datetime index before export without mutating the caller.
    ohlc_data = ohlc_data.copy()
    ohlc_data.index = pd.to_datetime(ohlc_data.index)
    weekly_ohlc = (
        ohlc_data[required_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .sort_index()
        .dropna()
    )
    if weekly_ohlc.empty:
        raise ValueError(
            "Weekly OHLC data has no complete numeric candles; refusing to overwrite output"
        )

    # Export full history to CSV.
    weekly_ohlc.to_csv(output_file)

    return weekly_ohlc


def create_report_ohlc_summary(
    daily_ohlc_data, report_date, output_file="csv/report_ohlc_summary.csv"
):
    """
    Create report-date OHLC context from daily candles.

    The daily close is the canonical report-date close used in narrative/report
    logic. The weekly fields are week-to-date values derived from daily candles
    through the report date, so they provide weekly context without pulling in
    post-report-date movement from an open weekly candle.

    Parameters:
    daily_ohlc_data (pd.DataFrame): Daily OHLC DataFrame with DatetimeIndex.
    report_date (str or datetime): Canonical report as-of date.
    output_file (str): Path for CSV export.

    Returns:
    pd.DataFrame: One-row report OHLC summary.
    """
    required_columns = ["Open", "High", "Low", "Close"]
    if daily_ohlc_data is None or daily_ohlc_data.empty:
        raise ValueError("Daily OHLC data is empty; refusing to overwrite output")
    missing = [
        column for column in required_columns if column not in daily_ohlc_data.columns
    ]
    if missing:
        raise ValueError(
            f"Daily OHLC data is missing required columns {missing}; refusing to overwrite output"
        )

    daily_ohlc_data = daily_ohlc_data[required_columns].copy()
    daily_ohlc_data = daily_ohlc_data.apply(pd.to_numeric, errors="coerce")
    daily_ohlc_data = daily_ohlc_data.replace([np.inf, -np.inf], np.nan)
    daily_ohlc_data.index = pd.to_datetime(daily_ohlc_data.index).normalize()
    daily_ohlc_data = daily_ohlc_data.sort_index().dropna()
    if daily_ohlc_data.empty:
        raise ValueError(
            "Daily OHLC data has no complete numeric candles; refusing to overwrite output"
        )

    report_date = pd.to_datetime(report_date).normalize()
    available_dates = daily_ohlc_data.index[daily_ohlc_data.index <= report_date]
    if len(available_dates) == 0:
        raise ValueError("No daily OHLC data available on or before the report date.")

    actual_report_date = available_dates.max()
    daily_row = daily_ohlc_data.loc[actual_report_date]

    week_start = actual_report_date - pd.Timedelta(days=actual_report_date.weekday())
    week_to_date = daily_ohlc_data.loc[week_start:actual_report_date]

    out = pd.DataFrame(
        [
            {
                "Report Date": actual_report_date.strftime("%Y-%m-%d"),
                "Daily Open": daily_row["Open"],
                "Daily High": daily_row["High"],
                "Daily Low": daily_row["Low"],
                "Daily Close": daily_row["Close"],
                "Week Start": week_start.strftime("%Y-%m-%d"),
                "Week-to-Date Open": week_to_date["Open"].iloc[0],
                "Week-to-Date High": week_to_date["High"].max(),
                "Week-to-Date Low": week_to_date["Low"].min(),
                "Week-to-Date Close": daily_row["Close"],
                "Week-to-Date Days": len(week_to_date),
            }
        ]
    )
    out.to_csv(output_file, index=False)
    return out


def create_eoy_model_table(report_data, cagr_results, report_date=None):
    """
    Generates end-of-year price model projection table combining valuation metrics and CAGR data.

    This function creates a dataset used for projecting Bitcoin's end-of-year price based on
    multiple valuation models (Realized Price, Thermocap, 200-day MA, Energy Value) and their
    historical 4-year compound annual growth rates. The output is used for price modeling and
    forecasting analysis.

    Parameters:
    report_data (pd.DataFrame): DataFrame with DatetimeIndex containing Bitcoin valuation metrics:
                                price_close, realized_price, thermocap_price, 200_day_ma_price_close,
                                Lagged_Energy_Value, mvrv_ratio, thermocap_multiple, 200_day_multiple,
                                Energy_Value_Multiple.
    cagr_results (pd.DataFrame): DataFrame with DatetimeIndex containing 4-year CAGR calculations for
                                 the valuation models (output from calculate_rolling_cagr_for_all_metrics).
    report_date (str or pd.Timestamp, optional): As-of cutoff. When supplied, rows
                                                 after this date are excluded.

    Returns:
    pd.DataFrame: Combined DataFrame with columns:
        - Current values: price_close, realized_price, thermocap_price, 200_day_ma_price_close,
                         Lagged_Energy_Value
        - Multiples: mvrv_ratio, thermocap_multiple, 200_day_multiple, Energy_Value_Multiple
        - Growth rates: *_4_Year_CAGR for each valuation model
        Merged on DatetimeIndex with a left join, preserving selected report_data
        dates through ``report_date`` when a cutoff is supplied.
    """
    # Define the columns to extract from report_data
    columns_of_interest = [
        "price_close",
        "realized_price",
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
        "realized_price_4_Year_CAGR",
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

    if report_date is not None:
        cutoff = pd.to_datetime(report_date).normalize()
        report_data_filtered = report_data_filtered.loc[
            report_data_filtered.index <= cutoff
        ]
        cagr_results_filtered = cagr_results_filtered.loc[
            cagr_results_filtered.index <= cutoff
        ]

    # Merge both datasets on the index (assuming they share the same date index)
    full_data = report_data_filtered.merge(
        cagr_results_filtered, left_index=True, right_index=True, how="left"
    )

    return full_data


def create_monthly_returns_table(selected_metrics, report_date=None):
    """
    Generates a month-to-date (MTD) return comparison table indexed to the current month.

    This function compares Bitcoin's performance for the current month across all historical
    years where data is available. It calculates returns from the final positive close
    before the current month to both the current date and month end, providing historical
    context for current performance.

    Parameters:
    selected_metrics (pd.DataFrame): DataFrame with DatetimeIndex containing at minimum
                                      a 'price_close' column. Data filtered to 2014-01-01+.

    Returns:
    pd.DataFrame: Table with columns:
        - Year: The calendar year
        - Start Price ($): Final positive price before the month began for that year
        - End Price ($): Price at end of current month for that year
        - Return (%): Full month return percentage
        - Report Date Return (%): Return from month start to current date
        Final rows include current year data and median historical projection.
    """
    today = (
        pd.to_datetime(report_date).date()
        if report_date is not None
        else pd.to_datetime(selected_metrics.index.max()).date()
    )
    current_year = today.year
    current_month = today.month
    current_day = today.day

    selected_metrics = selected_metrics.sort_index()
    if report_date is not None:
        selected_metrics = selected_metrics.loc[: pd.to_datetime(report_date).normalize()]

    # Keep earlier rows available for the January 2014/prior-month boundary lookup;
    # the publication-year cutoff is applied to the loop, not the source history.
    all_prices = _positive_price_series(selected_metrics["price_close"])
    publication_years = [
        year for year in all_prices.index.year.unique() if year >= 2014
    ]

    monthly_returns = {}
    report_date_returns = {}

    # Get the starting price for the current month of the current year
    current_month_data = all_prices.loc[
        (all_prices.index.year == current_year)
        & (all_prices.index.month == current_month)
    ]
    current_start_price = _last_positive_before(
        all_prices, pd.Timestamp(current_year, current_month, 1)
    )
    if current_month_data.empty or pd.isna(current_start_price):
        return None  # No data for current month

    # Calculate monthly returns for each year
    for year in publication_years:
        monthly_data = all_prices.loc[
            (all_prices.index.year == year)
            & (all_prices.index.month == current_month)
        ]

        if not monthly_data.empty:
            start_price = _last_positive_before(
                all_prices, pd.Timestamp(year, current_month, 1)
            )
            if pd.isna(start_price):
                continue
            end_price = monthly_data.iloc[-1]
            return_pct = (end_price / start_price - 1) * 100
            monthly_returns[year] = (start_price, end_price, return_pct)

            # Report Date Return Calculation
            report_date_data = monthly_data[(monthly_data.index.day == current_day)]
            if not report_date_data.empty:
                report_date_price = report_date_data.iloc[-1]
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

    # The projection is a historical benchmark for the current period to be measured
    # against, so it must exclude the current period. Including the in-progress month
    # folds today's partial return into the very median it is being compared to.
    historical = df.drop(index=current_year, errors="ignore")

    # Calculate the historical median return
    median_return = historical["Return (%)"].median()
    median_end_price = current_start_price * (1 + median_return / 100)

    # Calculate the median return at the current date (not full period)
    median_report_date_return = historical["Report Date Return (%)"].median()

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


def create_yearly_returns_table(selected_metrics, report_date=None):
    """
    Generates a year-to-date (YTD) return comparison table indexed to the current day of year.

    This function compares Bitcoin's performance for the current year across all historical
    years where data is available. It calculates returns from the final positive close
    before January 1 to both the current calendar date and year end, providing historical
    context for current YTD performance.

    Parameters:
    selected_metrics (pd.DataFrame): DataFrame with DatetimeIndex containing at minimum
                                      a 'price_close' column. Data filtered to 2014-01-01+.

    Returns:
    pd.DataFrame: Table with columns:
        - Year: The calendar year
        - Start Price ($): Final positive price before January 1 for that year
        - End Price ($): Price on December 31st for that year
        - Return (%): Full year return percentage
        - Report Date Return (%): Return from January 1st to current day of year
        Final rows include current year data and median historical projection.
    """
    today = (
        pd.to_datetime(report_date).date()
        if report_date is not None
        else pd.to_datetime(selected_metrics.index.max()).date()
    )
    current_year = today.year
    selected_metrics = selected_metrics.sort_index()
    if report_date is not None:
        selected_metrics = selected_metrics.loc[: pd.to_datetime(report_date).normalize()]

    # Keep earlier rows available for the January 2014 boundary lookup.
    all_prices = _positive_price_series(selected_metrics["price_close"])
    publication_years = [
        year for year in all_prices.index.year.unique() if year >= 2014
    ]

    yearly_returns = {}
    report_date_returns = {}

    # Get the starting price for the current year
    current_year_data = all_prices.loc[all_prices.index.year == current_year]
    current_start_price = _last_positive_before(
        all_prices, pd.Timestamp(current_year, 1, 1)
    )
    if current_year_data.empty or pd.isna(current_start_price):
        return None  # No data for current year

    # Calculate yearly returns for each year
    for year in publication_years:
        yearly_data = all_prices.loc[all_prices.index.year == year]

        if not yearly_data.empty:
            start_price = _last_positive_before(
                all_prices, pd.Timestamp(year, 1, 1)
            )
            if pd.isna(start_price):
                continue
            end_price = yearly_data.iloc[-1]
            return_pct = (end_price / start_price - 1) * 100
            yearly_returns[year] = (start_price, end_price, return_pct)

            # Report Date Return Calculation.
            # Match on calendar date, not day-of-year: after February, a leap year's
            # ordinal day is one ahead of a common year's, so comparing dayofyear lines
            # today up against the previous calendar day in every leap year.
            report_date_data = yearly_data[
                (yearly_data.index.month == today.month)
                & (yearly_data.index.day == today.day)
            ]
            if not report_date_data.empty:
                report_date_price = report_date_data.iloc[-1]
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

    # The projection is a historical benchmark for the current year to be measured
    # against, so it must exclude the current year. Including the in-progress year folds
    # today's partial return into the very median it is being compared to, and drags the
    # projected year-end price toward the current year's performance.
    historical = df.drop(index=current_year, errors="ignore")

    # Calculate the historical median return
    median_return = historical["Return (%)"].median()
    median_end_price = current_start_price * (1 + median_return / 100)

    # Median return as of the same calendar date in prior years
    median_report_date_return_pct = historical["Report Date Return (%)"].dropna().median()

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


def create_asset_valuation_table(report_data, report_date=None):
    """
    Generates relative valuation comparison table showing Bitcoin price if it matched other asset market caps.

    This function calculates what Bitcoin's price would be if its market cap equaled various
    benchmark assets including stocks (AAPL, NVDA, META, AMZN), precious metals (gold, silver),
    and fiat money supplies (US M0, UK M0, gold reserves). It shows the percentage move required
    for Bitcoin to reach each valuation milestone.

    Parameters:
    report_data (pd.DataFrame): DataFrame with report-date row containing:
        - price_close: Current Bitcoin price
        - market_cap: Current Bitcoin market cap
        - *_mc_btc_price: Calculated BTC price if matching each asset's market cap
        - *_MarketCap or *_cap: Market cap values for comparison assets in USD
        - gold/silver ``*_marketcap_billion_usd`` columns: legacy-named columns
          whose stored values are absolute USD, not values to rescale by one billion

    Returns:
    pd.DataFrame: Table with columns:
        - Asset: Name of comparison asset (Bitcoin, stocks, gold, fiat currencies)
        - Market Cap (USD): Current market cap of the asset in USD
        - Market Cap BTC Price: What Bitcoin price would be at that market cap
        - BTC % Move to Marketcap BTC Price: Percentage points of gain/loss needed to
          reach that market cap (e.g. 1937.0 = +1937%)
        Numeric throughout; missing values are NaN. Sorted by market cap, descending.
    """
    assets = [
        {"name": "Bitcoin", "data": "price_close", "marketcap": "market_cap"},
        # Fiat money (M0)
        {
            "name": "Switzerland M0",
            "data": "Switzerland_btc_price",
            "marketcap": "Switzerland_cap",
        },
        {
            "name": "UK M0",
            "data": "United_Kingdom_btc_price",
            "marketcap": "United_Kingdom_cap",
        },
        {
            "name": "US M0",
            "data": "United_States_btc_price",
            "marketcap": "United_States_cap",
        },
        # Precious metals
        {
            "name": "Total Silver Market",
            "data": "silver_marketcap_btc_price",
            "marketcap": "silver_marketcap_billion_usd",
        },
        {
            "name": "Total Gold Market",
            "data": "gold_marketcap_btc_price",
            "marketcap": "gold_marketcap_billion_usd",
        },
        # Mega-cap stocks
        {"name": "Apple", "data": "AAPL_mc_btc_price", "marketcap": "AAPL_MarketCap"},
        {"name": "Amazon", "data": "AMZN_mc_btc_price", "marketcap": "AMZN_MarketCap"},
        {"name": "Meta", "data": "META_mc_btc_price", "marketcap": "META_MarketCap"},
        {"name": "NVIDIA", "data": "NVDA_mc_btc_price", "marketcap": "NVDA_MarketCap"},
        {"name": "Broadcom", "data": "AVGO_mc_btc_price", "marketcap": "AVGO_MarketCap"},
        {"name": "Tesla", "data": "TSLA_mc_btc_price", "marketcap": "TSLA_MarketCap"},
        {"name": "Eli Lilly", "data": "LLY_mc_btc_price", "marketcap": "LLY_MarketCap"},
        {"name": "Micron", "data": "MU_mc_btc_price", "marketcap": "MU_MarketCap"},
        {"name": "TSMC", "data": "TSM_mc_btc_price", "marketcap": "TSM_MarketCap"},
        {"name": "SpaceX", "data": "SPCX_mc_btc_price", "marketcap": "SPCX_MarketCap"},
        {"name": "Saudi Aramco", "data": "2222.SR_mc_btc_price", "marketcap": "2222.SR_MarketCap"},
        {"name": "Samsung Electronics", "data": "005930.KS_mc_btc_price", "marketcap": "005930.KS_MarketCap"},
        {"name": "Berkshire Hathaway Class B", "data": "BRK-B_mc_btc_price", "marketcap": "BRK-B_MarketCap"},
        # Financials
        {"name": "JPMorgan", "data": "JPM_mc_btc_price", "marketcap": "JPM_MarketCap"},
        {"name": "Visa", "data": "V_mc_btc_price", "marketcap": "V_MarketCap"},
    ]

    latest_data = (
        _row_asof(report_data, report_date)
        if report_date is not None
        else report_data.sort_index().iloc[-1]
    )
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
            percent_move = np.nan

        # Published as numbers, not pre-formatted strings. Formatting belongs to the
        # consumer: strings forced every reader to strip "$", "," and "%" before doing
        # arithmetic, and rounding the percentage to whole points here made Bitcoin's own
        # reference row indistinguishable from any asset within half a point of it.
        valuation_data.append(
            {
                "Asset": asset["name"],
                "Market Cap (USD)": marketcap_value,
                "Market Cap BTC Price": marketcap_btc_price,
                "BTC % Move to Marketcap BTC Price": percent_move,
            }
        )

    valuation_df = pd.DataFrame(valuation_data)
    valuation_df = (
        valuation_df.sort_values("Market Cap (USD)", ascending=False)
        .reset_index(drop=True)
    )

    return valuation_df
