"""Every published report table, built from the merged daily frame.

    - Summary: snapshot, 30-day history, investor sentiment
    - Fundamentals: network metrics with weekly detail and 52-week range
    - Performance: Bitcoin against equity, sector, macro and Bitcoin-industry assets
    - Returns: ROI, MTD/YTD comparisons, indexed return paths, monthly heatmap
    - OHLC: weekly candles and the report-date summary
    - Relative value: Bitcoin's price at other assets' market caps

main.py writes the tables. Values are numeric except in the fundamentals table, whose
rows mix units and are published as formatted text.
"""

import pandas as pd
from datetime import timedelta
import numpy as np
from pandas.tseries.offsets import MonthEnd
import calendar
from data_validation import OHLC_COLUMNS, assert_ohlc_usable
from metrics import create_correlation_matrix_data
from data_definitions import (
    ELECTRICITY_BASE_TARIFF_USD_PER_KWH,
    NUPL_SENTIMENT_WINDOW_DAYS,
    NUPL_SENTIMENT_ZONES,
    POWER_LAW_VALUATION_BANDS,
    SATS_PER_BTC,
    FIAT_MONEY_SUPPLY,
)


# Earliest year in the MTD/YTD comparisons and return paths; earlier data is too thin.
RETURN_HISTORY_MIN_YEAR = 2014


def _positive_price_series(price_series):
    """Return sorted, daily, positive prices without changing the source object."""
    prices = pd.to_numeric(price_series, errors="coerce").sort_index()
    prices = prices.dropna().loc[lambda values: values > 0]
    return prices.groupby(prices.index.normalize()).last()


def _last_positive_before(price_series, boundary):
    """Return the final positive price strictly before a calendar boundary."""
    prior = price_series.loc[price_series.index < pd.Timestamp(boundary)]
    return prior.iloc[-1] if not prior.empty else np.nan


# Simple moving averages over calendar-day windows; a window missing any close stays
# empty. The dashboard draws the 3-month, 1-year and 200-week lines.
PRICE_MOVING_AVERAGES = {
    "50-day MA": 50,
    "3-month MA": 90,
    "200-day MA": 200,
    "1-year MA": 364,
    "200-week MA": 1400,
}


# Daily series published in summary_history.csv, by label.
HEADLINE_METRICS = {
    "Bitcoin Price USD": "price_close",
    "Bitcoin Market Cap": "market_cap",
    "Sats Per Dollar": "sat_per_dollar",
    "Bitcoin Supply": "supply",
    "Bitcoin Miner Revenue": "coinbase_sum_24h_usd",
    "Bitcoin Transaction Volume": "transfer_volume_sum_24h_usd",
}

# Model columns published in onchain_price_models.csv, with their published names.
ONCHAIN_PRICE_MODEL_COLUMNS = {
    "price_close": "BTC Price",
    f"electricity_cost_{round(ELECTRICITY_BASE_TARIFF_USD_PER_KWH * 100)}c": "Electricity Cost",
    "metcalfe_value": "Metcalfe Value",
    "power_law_price": "Power Law Price",
    "sth_realized_price": "STH Realized Price",
    "lth_realized_price": "LTH Realized Price",
    "realized_price": "Realized Price",
}


def create_onchain_price_models(report_data, report_date):
    """Daily BTC price, valuation models, 3x realized price and price moving averages."""
    models = report_data.loc[:report_date, list(ONCHAIN_PRICE_MODEL_COLUMNS)].dropna(
        subset=["price_close"]
    )
    models["3x Realized Price"] = models["realized_price"] * 3
    models = add_price_moving_averages(models.rename(columns=ONCHAIN_PRICE_MODEL_COLUMNS))
    models.index.name = "date"
    return models


def add_price_moving_averages(frame, price_column="BTC Price"):
    """Return a copy of a date-indexed frame with the published price moving averages."""
    result = frame.copy()
    prices = pd.to_numeric(result[price_column], errors="coerce")
    for column, days in PRICE_MOVING_AVERAGES.items():
        window = prices.rolling(f"{days}D")
        result[column] = window.mean().where(window.count() == days)
    return result


def create_price_paths(
    price_series, report_date, period, min_year=RETURN_HISTORY_MIN_YEAR
):
    """Each year's MTD or YTD price path, rebased to the current period's starting price.

    Every year starts from its last close before the period began; row 0 is that shared
    baseline. The current year stops at ``report_date``. YTD rows use a 365-day calendar
    (February 29 dropped) so dates line up across leap years. Returns one column per year,
    indexed by day of month or day of year.
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
    return result


def create_summary_history(
    report_data, report_date, metrics, comparison_days=30
):
    """Long-form daily values of each metric over the `comparison_days` before the report
    date, both endpoints included (31 rows per metric for 30 days)."""
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
    """Bitcoin's return from 1 day to 10 years before the report date: Time Frame, ROI (%),
    Start Date and Start Price."""
    if price_column not in data.columns:
        raise ValueError(
            f"The price column '{price_column}' does not exist in the data."
        )

    if data.empty:
        raise ValueError("The input data is empty.")

    period_offsets = {
        "1 Day": pd.DateOffset(days=1),
        "3 Day": pd.DateOffset(days=3),
        "7 Day": pd.DateOffset(days=7),
        "30 Day": pd.DateOffset(days=30),
        "90 Day": pd.DateOffset(days=90),
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

    roi_table = pd.DataFrame(
        {
            "Time Frame": period_offsets.keys(),
            "ROI (%)": [roi_data[period] for period in period_offsets],
            "Start Date": [start_dates[period] for period in period_offsets],
            "Start Price": [btc_prices[period] for period in period_offsets],
        }
    )
    return roi_table


def _format_fundamental_value(value, format_type):
    """Format one fundamentals value for display; blank when missing."""
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
    """One row per `metrics_template` metric: Section, Metric, Current Value, 7 Days Ago,
    7 Day Change (%), a Monday-Sunday column for each day of the report week, 52W Low
    and 52W High.

    Values are formatted text because rows mix units; 7 Day Change (%) stays numeric so
    the dashboard can colour it. Without `report_date`, the latest row is used.
    """
    table_data = []

    df = df.sort_index()
    if report_date is not None:
        df = df.loc[: pd.to_datetime(report_date).normalize()]

    latest_date = pd.to_datetime(report_date).normalize() if report_date is not None else df.index.max()
    start_of_week = latest_date - timedelta(days=latest_date.weekday())
    weekly_index = pd.date_range(start=start_of_week.normalize(), periods=7, freq="D")

    for section, metrics in metrics_template.items():
        for metric_display_name, (column_name, format_type) in metrics.items():
            series = df[column_name]
            current = series.get(latest_date, np.nan)
            if pd.isna(current):
                raise ValueError(f"Fundamental {column_name} is missing on {latest_date.date()}")
            seven_days_ago = series.get(latest_date - timedelta(days=7), np.nan)

            # Change between the two values shown beside it, in percentage points.
            pct_change = (
                ((current / seven_days_ago) - 1) * 100
                if pd.notna(seven_days_ago) and seven_days_ago != 0
                else np.nan
            )

            year_window = series.loc[latest_date - timedelta(days=364):latest_date]
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


# --- Summary and performance tables ---


def _row_asof(df, report_date):
    """The latest row on or before the report date."""
    report_date = pd.to_datetime(report_date).normalize()
    df = df.sort_index()
    available_dates = df.index[df.index <= report_date]
    if len(available_dates) == 0:
        raise ValueError("No data available on or before the report date.")
    return df.loc[available_dates.max()]


def _band_label(value, bands):
    """Label of the first (upper bound, label) band the value is below."""
    return next(label for upper, label in bands if value < upper)


def _nupl_sentiment(report_data, report_date):
    """Market sentiment label: the NUPL zone of the trailing 7-day average NUPL."""
    report_date = pd.to_datetime(report_date).normalize()
    window = pd.to_numeric(
        report_data["nupl"].sort_index().loc[:report_date], errors="coerce"
    ).tail(NUPL_SENTIMENT_WINDOW_DAYS)
    if len(window) < NUPL_SENTIMENT_WINDOW_DAYS or window.isna().any():
        raise RuntimeError(
            f"NUPL needs {NUPL_SENTIMENT_WINDOW_DAYS} daily values through the report date"
        )
    return _band_label(window.mean(), NUPL_SENTIMENT_ZONES)


def _power_law_valuation(power_law_multiple):
    """Valuation label: the POWER_LAW_VALUATION_BANDS band of the power-law multiple."""
    if pd.isna(power_law_multiple) or power_law_multiple <= 0:
        raise RuntimeError("Power-law price multiple is required for the valuation label")
    return _band_label(power_law_multiple, POWER_LAW_VALUATION_BANDS)


def create_summary_table(report_data, report_date):
    """Report-date snapshot of the headline metrics: Metric, Value, Label and Category.

    Value holds the numbers; the two sentiment rows carry text in Label instead.
    """
    latest = _row_asof(report_data, report_date)

    price_usd = latest["price_close"]
    market_cap = latest["market_cap"]
    sats_per_dollar = SATS_PER_BTC / price_usd

    bitcoin_supply = latest["supply"]
    # Daily values, matching the same series in summary_history.csv.
    miner_revenue = latest["coinbase_sum_24h_usd"]
    tx_volume = latest["transfer_volume_sum_24h_usd"]
    supply_in_profit_pct = latest["supply_in_profit"] / latest["supply"] * 100
    if pd.isna(supply_in_profit_pct) or not 0 <= supply_in_profit_pct <= 100:
        raise RuntimeError("Supply in profit is required for the report-date summary snapshot")
    market_sentiment = _nupl_sentiment(report_data, report_date)
    bitcoin_valuation = _power_law_valuation(latest.get("power_law_price_multiple", np.nan))

    categorized_data = {
        "Market Data": {
            "Bitcoin Price USD": price_usd,
            "Bitcoin Market Cap": market_cap,
            "Sats Per Dollar": sats_per_dollar,
        },
        "On-chain Data": {
            "Bitcoin Supply": bitcoin_supply,
            "Bitcoin Miner Revenue": miner_revenue,
            "Bitcoin Transaction Volume": tx_volume,
        },
        "Investor Sentiment": {
            "Bitcoin Supply in Profit (%)": supply_in_profit_pct,
            "Bitcoin Market Sentiment": market_sentiment,
            "Bitcoin Valuation": bitcoin_valuation,
        },
    }

    summary_rows = []
    for category, metrics in categorized_data.items():
        for metric, value in metrics.items():
            is_label = isinstance(value, str)
            summary_rows.append({
                "Metric": metric,
                "Value": np.nan if is_label else value,
                "Label": value if is_label else None,
                "Category": category,
            })
    return pd.DataFrame(summary_rows, columns=["Metric", "Value", "Label", "Category"])


# Performance table rows by category, in published order. Consumers show the Bitcoin row
# alongside each group.
PERFORMANCE_GROUPS = {
    "Bitcoin": [("Bitcoin - [BTC]", "price_close")],
    "Equity Market Indexes": [
        ("S&P 500 Index ETF - [SPY]", "SPY"),
        ("Nasdaq-100 ETF - [QQQ]", "QQQ"),
        ("Russell 2000 Small-Cap ETF - [IWM]", "IWM"),
        ("International Stock ETF - [VXUS]", "VXUS"),
    ],
    "Sectors": [
        ("Technology Sector ETF - [XLK]", "XLK"),
        ("Financials Sector ETF - [XLF]", "XLF"),
        ("Energy Sector ETF - [XLE]", "XLE"),
        ("Real Estate Sector ETF - [XLRE]", "XLRE"),
    ],
    "Macro Asset Classes": [
        ("US Dollar Index - [DXY]", "DX-Y.NYB"),
        ("Gold ETF - [GLD]", "GLD"),
        ("Aggregate Bond ETF - [AGG]", "AGG"),
        ("S&P GSCI Commodity Index - [SPGSCI]", "^SPGSCI"),
    ],
    "Bitcoin Industry Performance": [
        ("MicroStrategy - [MSTR]", "MSTR"),
        ("Block - [XYZ]", "XYZ"),
        ("Coinbase - [COIN]", "COIN"),
        ("Bitcoin Miners ETF - [WGMI]", "WGMI"),
    ],
}


CORRELATION_ASSETS = [
    (category, asset, "BTC" if ticker == "price_close" else ticker,
     "price_close" if ticker == "price_close" else f"{ticker}_close")
    for category, assets in PERFORMANCE_GROUPS.items()
    for asset, ticker in assets
]
CORRELATION_MATRIX_PERIODS = (30, 90, 365)


def create_correlation_matrix_table(correlations_data, report_date):
    """30/90/365-day matrices for Bitcoin and the performance-table assets, in window and group order."""
    columns = [column for _, _, _, column in CORRELATION_ASSETS]
    tickers = [ticker for _, _, ticker, _ in CORRELATION_ASSETS]
    tables = []
    for period in CORRELATION_MATRIX_PERIODS:
        matrix = create_correlation_matrix_data(report_date, columns, correlations_data, period)
        matrix.columns = tickers
        metadata = pd.DataFrame([
            {"Report Date": str(pd.Timestamp(report_date).date()), "Window Days": period,
             "Category": category, "Asset": asset, "Ticker": ticker}
            for category, asset, ticker, _ in CORRELATION_ASSETS
        ])
        tables.append(pd.concat([metadata, matrix.reset_index(drop=True)], axis=1))
    return pd.concat(tables, ignore_index=True)


def _build_performance_table(
    report_data: pd.DataFrame,
    report_date,
    correlation_results: dict,
    asset_groups: dict,
) -> pd.DataFrame:
    """One row per asset, in group order: price, 7-day/MTD/YTD/90-day returns, 52-week
    range and 90-day correlation with Bitcoin. Ticker "price_close" is Bitcoin."""
    # Every value and the 52-week window use the same latest row on or before the cutoff.
    report_date = pd.to_datetime(report_date).normalize()
    report_data = report_data.sort_index()
    available_dates = report_data.index[report_data.index <= report_date]
    if len(available_dates) == 0:
        raise ValueError("No data available on or before the report date.")
    actual_report_date = available_dates.max()
    latest = report_data.loc[actual_report_date]
    if isinstance(latest, pd.DataFrame):
        latest = latest.iloc[-1]

    year_ago = actual_report_date - pd.Timedelta(days=365)

    rows = []
    for category, assets in asset_groups.items():
        for label, ticker in assets:
            price_col = ticker if ticker == "price_close" else f"{ticker}_close"
            window = report_data.loc[year_ago:actual_report_date, price_col].dropna()
            rows.append({
                "Category": category,
                "Asset": label,
                "Price": latest[price_col],
                "7 Day Return (%)": latest[f"{price_col}_7d_change"],
                "MTD Return (%)": latest[f"{price_col}_mtd_change"],
                "YTD Return (%)": latest[f"{price_col}_ytd_change"],
                "90 Day Return (%)": latest[f"{price_col}_90d_change"],
                "52 Week High": window.max() if len(window) else None,
                "52 Week Low": window.min() if len(window) else None,
                "90 Day BTC Correlation": 1 if ticker == "price_close" else
                    correlation_results["price_close_90_days"].loc["price_close", price_col],
            })
    return pd.DataFrame(rows)


def create_full_performance_table(report_data, report_date, correlation_results):
    """Performance rows for Bitcoin and every asset in PERFORMANCE_GROUPS."""
    return _build_performance_table(
        report_data, report_date, correlation_results, PERFORMANCE_GROUPS
    )


def monthly_heatmap(data, report_date=None):
    """Bitcoin's monthly and yearly returns from 2012, in percentage points.

    Rows are years plus "4-Year Average", "Median" and "Average"; columns are Jan-Dec and
    "Yearly". Each return runs from the prior period's last close, so the current month
    and year show MTD and YTD. The summary rows leave out incomplete periods.
    """
    data = data.sort_index()
    if report_date is not None:
        report_date = pd.to_datetime(report_date).normalize()
        data = data.loc[:report_date]

    # Pre-2012 prices are kept only as the starting close for January 2012.
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

    last_date = display_prices.index[-1]
    current_year, current_month = last_date.year, last_date.month
    is_incomplete_month = last_date.day != (last_date + MonthEnd(0)).day

    # Same prior-close basis as the monthly cells, so a full year's months compound to it.
    yearly_returns = {}
    for year, year_prices in display_prices.groupby(display_prices.index.year):
        prior_year_close = _last_positive_before(
            all_prices, pd.Timestamp(year=year, month=1, day=1)
        )
        if pd.notna(prior_year_close):
            yearly_returns[year] = (year_prices.iloc[-1] / prior_year_close) - 1
    heatmap_data[13] = pd.Series(yearly_returns)

    # The summary rows leave out the incomplete month and year.
    heatmap_data_excluded = heatmap_data.copy()
    if current_year in heatmap_data.index:
        if is_incomplete_month:
            heatmap_data_excluded.loc[current_year, current_month] = pd.NA
        if (last_date.month, last_date.day) != (12, 31):
            heatmap_data_excluded.loc[current_year, 13] = pd.NA

    # The four most recent years with a value in each column, not the last four rows.
    heatmap_data.loc["4-Year Average"] = heatmap_data_excluded.apply(
        lambda col: col.dropna().tail(4).mean(), axis=0
    )

    heatmap_data.loc["Median"] = heatmap_data_excluded.apply(
        lambda col: col[~col.isna()].median(), axis=0
    )

    heatmap_data.loc["Average"] = heatmap_data_excluded.apply(
        lambda col: col[~col.isna()].mean(), axis=0
    )

    heatmap_data = heatmap_data * 100
    month_names = [calendar.month_abbr[i] for i in range(1, 13)] + ["Yearly"]
    heatmap_data.columns = month_names
    heatmap_data.index.name = "Year"

    return heatmap_data


# --- OHLC and period return tables ---


def create_report_ohlc_summary(daily_ohlc, report_date):
    """One row: the report-date daily candle and the week-to-date candle through it.

    Raises if the report date or any earlier day of its week is missing.
    """
    assert_ohlc_usable(daily_ohlc, "Daily OHLC")
    daily = daily_ohlc[OHLC_COLUMNS].copy()
    daily.index = pd.to_datetime(daily.index).normalize()
    daily = daily.sort_index()

    report_date = pd.to_datetime(report_date).normalize()
    if report_date not in daily.index:
        raise ValueError("Daily OHLC is missing the report date")
    daily_row = daily.loc[report_date]

    week_start = report_date - pd.Timedelta(days=report_date.weekday())
    week_to_date = daily.loc[week_start:report_date]
    if not week_to_date.index.equals(pd.date_range(week_start, report_date)):
        raise ValueError("Daily OHLC is missing a day in the report week")

    return pd.DataFrame(
        [
            {
                "date": report_date.strftime("%Y-%m-%d"),
                "daily_open": daily_row["Open"],
                "daily_high": daily_row["High"],
                "daily_low": daily_row["Low"],
                "daily_close": daily_row["Close"],
                "week_start": week_start.strftime("%Y-%m-%d"),
                "week_to_date_open": week_to_date["Open"].iloc[0],
                "week_to_date_high": week_to_date["High"].max(),
                "week_to_date_low": week_to_date["Low"].min(),
                "week_to_date_close": daily_row["Close"],
                "week_to_date_days": len(week_to_date),
            }
        ]
    )


def create_period_returns_table(report_data, report_date, period):
    """Current-year MTD or YTD return and a "Median Projection" row.

    Returns run from the last close before the period began. The projection applies the
    median full-period return of past years to this period's start price. Report Date
    Return (%) is the return to the report's calendar date.
    """
    if period not in {"mtd", "ytd"}:
        raise ValueError("period must be either 'mtd' or 'ytd'.")
    report_date = pd.to_datetime(report_date).normalize()
    prices = _positive_price_series(report_data.sort_index().loc[:report_date, "price_close"])

    def period_prices(year):
        mask = prices.index.year == year
        if period == "mtd":
            mask &= prices.index.month == report_date.month
        return prices.loc[mask]

    def period_start_price(year):
        month = report_date.month if period == "mtd" else 1
        return _last_positive_before(prices, pd.Timestamp(year, month, 1))

    current_start_price = period_start_price(report_date.year)
    if period_prices(report_date.year).empty or pd.isna(current_start_price):
        raise ValueError(f"No {period.upper()} price history for {report_date.date()}")

    rows = {}
    for year in prices.index.year.unique():
        year_prices = period_prices(year)
        start_price = period_start_price(year)
        if year < RETURN_HISTORY_MIN_YEAR or year_prices.empty or pd.isna(start_price):
            continue
        end_price = year_prices.iloc[-1]
        # Match the calendar date; day-of-year shifts by one after February in leap years.
        same_date = year_prices[
            (year_prices.index.month == report_date.month)
            & (year_prices.index.day == report_date.day)
        ]
        rows[year] = (
            start_price,
            end_price,
            (end_price / start_price - 1) * 100,
            (same_date.iloc[-1] / start_price - 1) * 100 if not same_date.empty else np.nan,
        )

    table = pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=["Start Price ($)", "End Price ($)", "Return (%)", "Report Date Return (%)"],
    )
    table.index.name = "Year"

    # The benchmark excludes the current, partial period.
    historical = table.drop(index=report_date.year, errors="ignore")
    median_return = historical["Return (%)"].median()
    median_row = pd.DataFrame(
        {
            "Year": ["Median Projection"],
            "Start Price ($)": [current_start_price],
            "End Price ($)": [current_start_price * (1 + median_return / 100)],
            "Return (%)": [median_return],
            "Report Date Return (%)": [historical["Report Date Return (%)"].median()],
        }
    )
    return pd.concat([table.loc[[report_date.year]].reset_index(), median_row], ignore_index=True)


def create_asset_valuation_table(report_data, report_date=None):
    """Bitcoin's price if its market cap matched each asset (M0 supplies, gold, silver,
    large stocks), with the move needed to get there.

    Columns: Asset, Market Cap (USD), BTC Price at Market Cap and Move Needed (%). Sorted
    by market cap, largest first.
    """
    fiat_m0_usd = dict(zip(
        FIAT_MONEY_SUPPLY["Country"], FIAT_MONEY_SUPPLY["US Dollar Trillion"] * 1e12
    ))
    assets = [
        {"name": "Bitcoin", "data": "price_close", "market_cap": "market_cap"},
        # Fiat money (M0)
        {
            "name": "Switzerland M0",
            "data": "switzerland_m0_btc_price",
            "market_cap_usd": fiat_m0_usd["Switzerland"],
        },
        {
            "name": "UK M0",
            "data": "united_kingdom_m0_btc_price",
            "market_cap_usd": fiat_m0_usd["United Kingdom"],
        },
        {
            "name": "US M0",
            "data": "united_states_m0_btc_price",
            "market_cap_usd": fiat_m0_usd["United States"],
        },
        # Precious metals
        {
            "name": "Total Silver Market",
            "data": "silver_market_cap_btc_price",
            "market_cap": "silver_market_cap_usd",
        },
        {
            "name": "Total Gold Market",
            "data": "gold_market_cap_btc_price",
            "market_cap": "gold_market_cap_usd",
        },
        # Mega-cap stocks
        {"name": "Apple", "data": "AAPL_market_cap_btc_price", "market_cap": "AAPL_market_cap"},
        {"name": "Amazon", "data": "AMZN_market_cap_btc_price", "market_cap": "AMZN_market_cap"},
        {"name": "Meta", "data": "META_market_cap_btc_price", "market_cap": "META_market_cap"},
        {"name": "NVIDIA", "data": "NVDA_market_cap_btc_price", "market_cap": "NVDA_market_cap"},
        {"name": "Broadcom", "data": "AVGO_market_cap_btc_price", "market_cap": "AVGO_market_cap"},
        {"name": "Tesla", "data": "TSLA_market_cap_btc_price", "market_cap": "TSLA_market_cap"},
        {"name": "Eli Lilly", "data": "LLY_market_cap_btc_price", "market_cap": "LLY_market_cap"},
        {"name": "Micron", "data": "MU_market_cap_btc_price", "market_cap": "MU_market_cap"},
        {"name": "TSMC", "data": "TSM_market_cap_btc_price", "market_cap": "TSM_market_cap"},
        {"name": "SpaceX", "data": "SPCX_market_cap_btc_price", "market_cap": "SPCX_market_cap"},
        {"name": "Saudi Aramco", "data": "2222.SR_market_cap_btc_price", "market_cap": "2222.SR_market_cap"},
        {"name": "Samsung Electronics", "data": "005930.KS_market_cap_btc_price", "market_cap": "005930.KS_market_cap"},
        {"name": "Berkshire Hathaway Class B", "data": "BRK-B_market_cap_btc_price", "market_cap": "BRK-B_market_cap"},
        # Financials
        {"name": "JPMorgan", "data": "JPM_market_cap_btc_price", "market_cap": "JPM_market_cap"},
        {"name": "Visa", "data": "V_market_cap_btc_price", "market_cap": "V_market_cap"},
    ]

    latest_data = (
        _row_asof(report_data, report_date)
        if report_date is not None
        else report_data.sort_index().iloc[-1]
    )
    bitcoin_price = latest_data.get("price_close", float("nan"))

    valuation_data = []
    for asset in assets:
        btc_price_at_market_cap = latest_data.get(asset["data"], float("nan"))
        market_cap_value = asset.get("market_cap_usd")
        if market_cap_value is None:
            market_cap_value = latest_data.get(asset["market_cap"], float("nan"))

        if (
            pd.notna(bitcoin_price)
            and pd.notna(btc_price_at_market_cap)
            and bitcoin_price > 0
        ):
            percent_move = ((btc_price_at_market_cap - bitcoin_price) / bitcoin_price) * 100
        else:
            percent_move = np.nan

        valuation_data.append(
            {
                "Asset": asset["name"],
                "Market Cap (USD)": market_cap_value,
                "BTC Price at Market Cap": btc_price_at_market_cap,
                "Move Needed (%)": percent_move,
            }
        )

    valuation_df = pd.DataFrame(valuation_data)
    valuation_df = (
        valuation_df.sort_values("Market Cap (USD)", ascending=False)
        .reset_index(drop=True)
    )

    return valuation_df
