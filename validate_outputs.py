"""Validate generated report artifacts without fetching or mutating data."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Local configuration only; importing data_definitions performs no I/O.
from data_definitions import REPORT_DATE as CLOCK_REPORT_DATE


@dataclass(frozen=True)
class RowBounds:
    minimum: int
    maximum: int | None = None


# Bounds are intentionally broad for long-form history, and tight for fixed-shape
# report tables. They catch truncation, header-only files, accidental duplication,
# and runaway exports without coupling validation to today's exact history length.
OUTPUT_RULES = {
    "cycle_low_data.csv": RowBounds(1, 100_000),
    "drawdown_data.csv": RowBounds(1, 100_000),
    "fundamentals_table.csv": RowBounds(1, 1_000),
    "halving_data.csv": RowBounds(1, 100_000),
    "master_metrics_data.csv.gz": RowBounds(365, 100_000),
    "monthly_heatmap_data.csv": RowBounds(4, 1_000),
    "mtd_return_comparison.csv": RowBounds(2, 10),
    "mtd_returns_history.csv": RowBounds(29, 32),
    "ohlc_data.csv": RowBounds(52, 10_000),
    "onchain_price_models.csv": RowBounds(365, 100_000),
    "performance_table.csv": RowBounds(1, 1_000),
    "price_outlook.csv": RowBounds(1, 1_000),
    "relative_value_comparison.csv": RowBounds(2, 1_000),
    "report_ohlc_summary.csv": RowBounds(1, 1),
    "roi_table.csv": RowBounds(1, 100),
    "summary_history.csv": RowBounds(31, 1_000),
    "summary_table.csv": RowBounds(1, 100),
    "ytd_return_comparison.csv": RowBounds(2, 10),
    "ytd_returns_history.csv": RowBounds(365, 366),
}


REQUIRED_COLUMNS = {
    "cycle_low_data.csv": {"days_since_cycle_low", "index_value", "Cycle"},
    "drawdown_data.csv": {"days_since_ath", "drawdown_pct", "Cycle"},
    "fundamentals_table.csv": {"Section", "Metric", "Current Value"},
    "halving_data.csv": {"days_since_halving", "index_value", "Era"},
    "master_metrics_data.csv.gz": {
        "time", "price_close", "market_cap", "metcalfe_value",
        "power_law_price", "60_day_ma_hash_rate", "hash_ribbon_capitulation",
        "price_close_4_Year_CAGR",
    },
    "monthly_heatmap_data.csv": {
        "time", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
        "Aug", "Sep", "Oct", "Nov", "Dec", "Yearly",
    },
    "mtd_return_comparison.csv": {
        "Year", "End Price ($)", "Return (%)", "Report Date Return (%)",
    },
    "mtd_returns_history.csv": {"day", "Median", "Average"},
    "ohlc_data.csv": {"Time", "Open", "High", "Low", "Close"},
    "onchain_price_models.csv": {
        "date", "BTC Price", "Electricity Cost", "Metcalfe Value", "Power Law Price",
        "50-day MA", "3-month MA", "200-day MA", "1-year MA", "200-week MA",
    },
    "performance_table.csv": {
        "Category", "Asset", "Price", "MTD Return (%)", "YTD Return (%)",
    },
    "price_outlook.csv": {"label", "price", "type", "color", "outlook_year"},
    "relative_value_comparison.csv": {"Asset", "Market Cap (USD)", "Market Cap BTC Price"},
    "report_ohlc_summary.csv": {"Report Date", "Daily Close"},
    "roi_table.csv": {"Time Frame", "ROI (%)", "Start Date", "BTC Price"},
    "summary_history.csv": {"Metric", "date", "Value"},
    "summary_table.csv": {"Metric", "Value", "Category"},
    "ytd_return_comparison.csv": {
        "Year", "End Price ($)", "Return (%)", "Report Date Return (%)",
    },
    "ytd_returns_history.csv": {"day_of_year", "Median", "Average"},
}


SUMMARY_HISTORY_METRICS = {
    "Bitcoin Price USD",
    "Bitcoin Marketcap",
    "Sats Per Dollar",
    "Bitcoin Supply",
    "Bitcoin Miner Revenue",
    "Bitcoin Transaction Volume",
}


RETAINED_OUTPUTS = {
    "ohlc_data.csv",
    "fundamentals_table.csv",
    "cycle_low_data.csv",
    "halving_data.csv",
    "monthly_heatmap_data.csv",
    "mtd_return_comparison.csv",
    "mtd_returns_history.csv",
    "onchain_price_models.csv",
    "performance_table.csv",
    "report_ohlc_summary.csv",
    "summary_history.csv",
    "summary_table.csv",
    "ytd_return_comparison.csv",
    "ytd_returns_history.csv",
}


_INFINITY_TOKENS = {
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}


def _chunk_has_infinity(chunk: pd.DataFrame) -> bool:
    numeric = chunk.select_dtypes(include=[np.number])
    if not numeric.empty and np.isinf(numeric.to_numpy(dtype=float)).any():
        return True

    for column in chunk.columns.difference(numeric.columns):
        values = chunk[column].dropna().astype(str).str.strip().str.lower()
        if values.isin(_INFINITY_TOKENS).any():
            return True
    return False


def _scan_csv(path: Path, retain: bool) -> tuple[int, set[str], bool, pd.DataFrame | None]:
    row_count = 0
    columns: set[str] = set()
    contains_infinity = False
    retained_chunks = []

    reader = pd.read_csv(path, chunksize=10_000, low_memory=False)
    for chunk in reader:
        columns = set(chunk.columns)
        row_count += len(chunk)
        contains_infinity = contains_infinity or _chunk_has_infinity(chunk)
        if retain:
            retained_chunks.append(chunk)

    frame = None
    if retain:
        frame = (
            pd.concat(retained_chunks, ignore_index=True)
            if retained_chunks
            else pd.DataFrame(columns=sorted(columns))
        )
    return row_count, columns, contains_infinity, frame


def _normalized_dates(
    frame: pd.DataFrame,
    column: str,
    filename: str,
    errors: list[str],
) -> pd.Series | None:
    if column not in frame.columns:
        return None
    dates = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    if dates.isna().any():
        errors.append(f"{filename}: {column!r} contains invalid or missing dates")
        return None
    return dates


def _validate_dated_output(
    frames: dict[str, pd.DataFrame],
    filename: str,
    column: str,
    expected_report_date: pd.Timestamp,
    errors: list[str],
    require_every_row: bool = False,
) -> None:
    frame = frames.get(filename)
    if frame is None or frame.empty:
        return
    dates = _normalized_dates(frame, column, filename, errors)
    if dates is None:
        return

    if require_every_row:
        mismatches = dates.ne(expected_report_date)
        if mismatches.any():
            found = sorted(dates[mismatches].dt.strftime("%Y-%m-%d").unique())
            errors.append(
                f"{filename}: expected every {column!r} to be "
                f"{expected_report_date.date()}, found {found}"
            )
    elif dates.max() != expected_report_date:
        errors.append(
            f"{filename}: latest {column!r} is {dates.max().date()}, "
            f"expected {expected_report_date.date()}"
        )


# Exports too large to retain in memory are checked by streaming their index column
# only. The master is large raw; holding it the way RETAINED_OUTPUTS does would be
# wasteful when the only thing left to assert is the cutoff.
INDEX_CUTOFF_OUTPUTS = {
    "master_metrics_data.csv.gz": "time",
}


def _validate_index_cutoff(
    output_dir: Path,
    filename: str,
    column: str,
    expected_report_date: pd.Timestamp,
    errors: list[str],
) -> None:
    """Assert a large dated export ends exactly on the report date."""
    path = output_dir / filename
    if not path.is_file():
        return
    try:
        index = pd.read_csv(path, usecols=[column])[column]
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        errors.append(f"{filename}: cannot read {column!r} ({exc})")
        return

    parsed = pd.to_datetime(index, errors="coerce")
    if parsed.isna().any():
        errors.append(f"{filename}: {column!r} contains invalid or missing dates")
        return
    dates = parsed.dt.normalize()
    from data_validation import validate_calendar
    try:
        validate_calendar(pd.DatetimeIndex(parsed), filename)
    except (RuntimeError, ValueError) as exc:
        errors.append(str(exc))
    if dates.max() != expected_report_date:
        errors.append(
            f"{filename}: latest {column!r} is {dates.max().date()}, "
            f"expected {expected_report_date.date()}"
        )


def _current_history_position(report_date: pd.Timestamp, period: str) -> int:
    if period == "mtd":
        return report_date.day
    if report_date.month == 2 and report_date.day == 29:
        return 59
    position = report_date.dayofyear
    if report_date.is_leap_year and report_date.month > 2:
        position -= 1
    return position


def _validate_history_position(
    frames: dict[str, pd.DataFrame],
    filename: str,
    index_column: str,
    period: str,
    expected_report_date: pd.Timestamp,
    errors: list[str],
) -> None:
    frame = frames.get(filename)
    if frame is None or frame.empty:
        return
    year_column = str(expected_report_date.year)
    if year_column not in frame.columns:
        errors.append(f"{filename}: missing current-year column {year_column!r}")
        return

    current = frame.loc[frame[year_column].notna(), index_column]
    positions = pd.to_numeric(current, errors="coerce").dropna()
    if positions.empty:
        errors.append(f"{filename}: current-year column {year_column!r} is empty")
        return

    actual = int(positions.max())
    expected = _current_history_position(expected_report_date, period)
    if actual != expected:
        errors.append(
            f"{filename}: current-year series ends at {index_column}={actual}, "
            f"expected {expected} for {expected_report_date.date()}"
        )


def _unique_numeric_values(
    frame: pd.DataFrame,
    value_column: str,
    mask: pd.Series | None = None,
) -> list[float]:
    values = frame.loc[mask, value_column] if mask is not None else frame[value_column]
    return sorted(pd.to_numeric(values, errors="coerce").dropna().unique().tolist())


def _validate_price_agreement(
    frames: dict[str, pd.DataFrame],
    expected_report_date: pd.Timestamp,
    errors: list[str],
) -> None:
    prices: dict[str, float] = {}

    summary = frames.get("summary_table.csv")
    if summary is not None and {"Metric", "Value"}.issubset(summary.columns):
        values = _unique_numeric_values(
            summary, "Value", summary["Metric"].eq("Bitcoin Price USD")
        )
        if len(values) == 1:
            prices["summary_table.csv"] = values[0]
        else:
            errors.append("summary_table.csv: expected one Bitcoin Price USD value")

    performance = frames.get("performance_table.csv")
    if performance is not None and {"Asset", "Price"}.issubset(performance.columns):
        values = _unique_numeric_values(
            performance, "Price", performance["Asset"].eq("Bitcoin - [BTC]")
        )
        if len(values) == 1:
            prices["performance_table.csv"] = values[0]
        else:
            errors.append("performance_table.csv: Bitcoin rows do not share one price")

    dated_price_sources = {
        "onchain_price_models.csv": ("date", "BTC Price"),
        "report_ohlc_summary.csv": ("Report Date", "Daily Close"),
    }
    for filename, (date_column, value_column) in dated_price_sources.items():
        frame = frames.get(filename)
        if frame is None or not {date_column, value_column}.issubset(frame.columns):
            continue
        dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
        values = _unique_numeric_values(
            frame, value_column, dates.eq(expected_report_date)
        )
        if len(values) == 1:
            prices[filename] = values[0]
        else:
            errors.append(
                f"{filename}: expected one price for {expected_report_date.date()}"
            )

    for filename in ("mtd_return_comparison.csv", "ytd_return_comparison.csv"):
        frame = frames.get(filename)
        if frame is None or not {"Year", "End Price ($)"}.issubset(frame.columns):
            continue
        values = _unique_numeric_values(
            frame,
            "End Price ($)",
            frame["Year"].astype(str).eq(str(expected_report_date.year)),
        )
        if len(values) == 1:
            prices[filename] = values[0]
        else:
            errors.append(f"{filename}: expected one current-year end price")

    if len(prices) < 2:
        return
    reference_name, reference_price = next(iter(prices.items()))
    for filename, price in prices.items():
        if not np.isclose(price, reference_price, rtol=1e-9, atol=1e-6):
            errors.append(
                f"{filename}: report-date BTC price {price} disagrees with "
                f"{reference_name} ({reference_price})"
            )


def _validate_return_agreement(
    frames: dict[str, pd.DataFrame],
    expected_report_date: pd.Timestamp,
    errors: list[str],
) -> None:
    returns: dict[str, dict[str, float]] = {"MTD": {}, "YTD": {}}

    performance = frames.get("performance_table.csv")
    performance_columns = {"Asset", "MTD Return (%)", "YTD Return (%)"}
    if performance is not None and performance_columns.issubset(performance.columns):
        bitcoin_rows = performance.loc[
            performance["Asset"].eq("Bitcoin - [BTC]")
        ]
        for period, column in (
            ("MTD", "MTD Return (%)"),
            ("YTD", "YTD Return (%)"),
        ):
            values = _unique_numeric_values(bitcoin_rows, column)
            if len(values) == 1:
                returns[period]["performance_table.csv"] = values[0]
            else:
                errors.append(
                    f"performance_table.csv: Bitcoin rows do not share one {period} return"
                )

    for period, filename in (
        ("MTD", "mtd_return_comparison.csv"),
        ("YTD", "ytd_return_comparison.csv"),
    ):
        frame = frames.get(filename)
        comparison_columns = {"Year", "Return (%)", "Report Date Return (%)"}
        if frame is None or not comparison_columns.issubset(frame.columns):
            continue
        current_rows = frame.loc[
            frame["Year"].astype(str).eq(str(expected_report_date.year))
        ]
        if len(current_rows) != 1:
            errors.append(
                f"{filename}: expected exactly one row for {expected_report_date.year}"
            )
            continue
        for column in ("Return (%)", "Report Date Return (%)"):
            values = _unique_numeric_values(current_rows, column)
            if len(values) == 1:
                returns[period][f"{filename} {column}"] = values[0]
            else:
                errors.append(
                    f"{filename}: current-year {column!r} is missing or non-numeric"
                )

    heatmap = frames.get("monthly_heatmap_data.csv")
    month_column = expected_report_date.strftime("%b")
    heatmap_columns = {"time", month_column, "Yearly"}
    if heatmap is not None and heatmap_columns.issubset(heatmap.columns):
        current_rows = heatmap.loc[
            heatmap["time"].astype(str).eq(str(expected_report_date.year))
        ]
        if len(current_rows) != 1:
            errors.append(
                "monthly_heatmap_data.csv: expected exactly one current-year row"
            )
        else:
            for period, column in (("MTD", month_column), ("YTD", "Yearly")):
                values = _unique_numeric_values(current_rows, column)
                if len(values) == 1:
                    returns[period][f"monthly_heatmap_data.csv {column}"] = values[0]
                else:
                    errors.append(
                        f"monthly_heatmap_data.csv: current-year {column!r} "
                        "is missing or non-numeric"
                    )

    for period, sources in returns.items():
        if len(sources) < 2:
            continue
        reference_name, reference_value = next(iter(sources.items()))
        for source_name, value in sources.items():
            if not np.isclose(value, reference_value, rtol=1e-9, atol=1e-9):
                errors.append(
                    f"{source_name}: BTC {period} return {value} disagrees with "
                    f"{reference_name} ({reference_value})"
                )


def _validate_cycle_contracts(
    frames: dict[str, pd.DataFrame],
    errors: list[str],
) -> None:
    cycle = frames.get("cycle_low_data.csv")
    cycle_columns = {"days_since_cycle_low", "index_value", "Cycle"}
    if cycle is not None and cycle_columns.issubset(cycle.columns):
        for label, group in cycle.groupby("Cycle", sort=False):
            ordered = group.sort_values("days_since_cycle_low")
            days = pd.to_numeric(ordered["days_since_cycle_low"], errors="coerce")
            values = pd.to_numeric(ordered["index_value"], errors="coerce")
            if days.isna().any() or values.isna().any():
                errors.append(f"cycle_low_data.csv: {label!r} has non-numeric rows")
                continue
            if days.iloc[0] != 0 or not np.isclose(values.iloc[0], 1.0):
                errors.append(
                    f"cycle_low_data.csv: {label!r} must start at day 0/index 1.0"
                )
            if values.min() < 1.0 - 1e-12:
                errors.append(
                    f"cycle_low_data.csv: {label!r} falls below its cycle-low "
                    f"baseline ({values.min()})"
                )

    halving = frames.get("halving_data.csv")
    halving_columns = {"days_since_halving", "index_value", "Era"}
    if halving is not None and halving_columns.issubset(halving.columns):
        if halving["Era"].eq("Genesis Era").any():
            errors.append(
                "halving_data.csv: Genesis Era has no positive day-0 source price "
                "and must be omitted"
            )
        for label, group in halving.groupby("Era", sort=False):
            ordered = group.sort_values("days_since_halving")
            days = pd.to_numeric(ordered["days_since_halving"], errors="coerce")
            values = pd.to_numeric(ordered["index_value"], errors="coerce")
            if days.isna().any() or values.isna().any():
                errors.append(f"halving_data.csv: {label!r} has non-numeric rows")
                continue
            if days.iloc[0] != 0 or not np.isclose(values.iloc[0], 1.0):
                errors.append(
                    f"halving_data.csv: {label!r} must start at day 0/index 1.0"
                )




# The Dashboard price chart's simple moving averages, recomputed here independently:
# calendar-day windows of daily closes, empty until every day in the window has a close.
PRICE_MOVING_AVERAGE_DAYS = {
    "50-day MA": 50,
    "3-month MA": 90,
    "200-day MA": 200,
    "1-year MA": 364,
    "200-week MA": 1400,
}


def _validate_price_moving_averages(
    frames: dict[str, pd.DataFrame],
    errors: list[str],
) -> None:
    filename = "onchain_price_models.csv"
    frame = frames.get(filename)
    columns = {"date", "BTC Price", *PRICE_MOVING_AVERAGE_DAYS}
    if frame is None or frame.empty or not columns.issubset(frame.columns):
        return
    prices = pd.Series(
        pd.to_numeric(frame["BTC Price"], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame["date"]),
    ).sort_index()
    for column, days in PRICE_MOVING_AVERAGE_DAYS.items():
        window = prices.rolling(f"{days}D")
        expected = window.mean().where(window.count() == days).to_numpy()
        actual = pd.Series(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame["date"]),
        ).sort_index().to_numpy()
        if not np.allclose(actual, expected, rtol=1e-9, atol=1e-6, equal_nan=True):
            errors.append(
                f"{filename}: {column!r} is not the {days}-day average of BTC Price"
            )




def _validate_report_agreement(
    frames: dict[str, pd.DataFrame],
    expected_report_date: pd.Timestamp,
    errors: list[str],
) -> None:
    _validate_dated_output(
        frames, "onchain_price_models.csv", "date", expected_report_date, errors
    )
    _validate_dated_output(
        frames,
        "report_ohlc_summary.csv",
        "Report Date",
        expected_report_date,
        errors,
        require_every_row=True,
    )


    history = frames.get("summary_history.csv")
    if history is not None and not history.empty:
        found_metrics = set(history["Metric"].dropna()) if "Metric" in history else set()
        missing_metrics = sorted(SUMMARY_HISTORY_METRICS - found_metrics)
        extra_metrics = sorted(found_metrics - SUMMARY_HISTORY_METRICS)
        if missing_metrics:
            errors.append(f"summary_history.csv: missing metrics {missing_metrics}")
        if extra_metrics:
            errors.append(f"summary_history.csv: unexpected metrics {extra_metrics}")

        if {"Metric", "date"}.issubset(history.columns):
            expected_start = expected_report_date - pd.Timedelta(days=30)
            for metric, group in history.groupby("Metric"):
                dates = _normalized_dates(group, "date", "summary_history.csv", errors)
                if dates is None:
                    continue
                if len(group) != 31 or dates.nunique() != 31:
                    errors.append(
                        f"summary_history.csv: {metric!r} has {len(group)} rows/"
                        f"{dates.nunique()} dates; expected 31 daily endpoints"
                    )
                if dates.min() != expected_start or dates.max() != expected_report_date:
                    errors.append(
                        f"summary_history.csv: {metric!r} spans "
                        f"{dates.min().date()} to {dates.max().date()}, expected "
                        f"{expected_start.date()} to {expected_report_date.date()}"
                    )

    _validate_history_position(
        frames,
        "mtd_returns_history.csv",
        "day",
        "mtd",
        expected_report_date,
        errors,
    )
    _validate_history_position(
        frames,
        "ytd_returns_history.csv",
        "day_of_year",
        "ytd",
        expected_report_date,
        errors,
    )
    _validate_price_agreement(frames, expected_report_date, errors)
    _validate_return_agreement(frames, expected_report_date, errors)
    _validate_cycle_contracts(frames, errors)
    _validate_price_moving_averages(frames, errors)


def _validate_investor_sentiment(summary, master_path, report_date, errors):
    """Recompute the three on-chain Investor Sentiment values from the master file."""
    from report_tables import _nupl_sentiment, _power_law_valuation
    try:
        master = pd.read_csv(
            master_path,
            usecols=["time", "nupl", "supply_in_profit", "supply", "power_law_price_multiple"],
            parse_dates=["time"],
        ).set_index("time").loc[:report_date]
        latest = master.iloc[-1]
        expected = {
            "Bitcoin Supply in Profit": latest["supply_in_profit"] / latest["supply"] * 100,
            "Bitcoin Market Sentiment": _nupl_sentiment(master, report_date),
            "Bitcoin Valuation": _power_law_valuation(latest["power_law_price_multiple"]),
        }
        values = summary.set_index("Metric")["Value"]
        for metric, value in expected.items():
            if metric not in values.index:
                raise ValueError(f"missing {metric!r}")
            observed = values[metric]
            matches = (
                np.isclose(float(observed), value, rtol=1e-9)
                if isinstance(value, float) else str(observed) == value
            )
            if not matches:
                raise ValueError(f"{metric!r} is {observed!r}, expected {value!r}")
    except (ValueError, KeyError, RuntimeError, IndexError) as exc:
        errors.append(f"summary_table.csv: investor sentiment does not match its source ({exc})")


PERFORMANCE_VALUE_COLUMNS = [
    "Price", "7 Day Return (%)", "MTD Return (%)", "YTD Return (%)", "90 Day Return (%)",
]


def _validate_performance_rows(frames, errors):
    """Every configured asset is published once, in its category, with a price and returns.

    A market-data outage leaves carried-forward prices blank rather than stale, so without
    this check a table of empty rows would still publish.
    """
    table = frames.get("performance_table.csv")
    if table is None or not {"Category", "Asset"}.issubset(table.columns):
        return
    from report_tables import PERFORMANCE_GROUPS
    expected = [
        (category, label)
        for category, assets in PERFORMANCE_GROUPS.items()
        for label, _ in assets
    ]
    if list(zip(table["Category"], table["Asset"])) != expected:
        errors.append("performance_table.csv: rows do not match the configured assets and categories")
        return
    absent = [column for column in PERFORMANCE_VALUE_COLUMNS if column not in table.columns]
    if absent:
        errors.append(f"performance_table.csv: missing columns {absent}")
        return
    values = table[PERFORMANCE_VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    blank = table.loc[values.isna().any(axis=1), "Asset"].tolist()
    if blank:
        errors.append(
            "performance_table.csv: missing price or returns for " + ", ".join(blank)
        )


def _validate_review_contracts(frames, output_dir, report_date, errors):
    from data_validation import validate_candles, validate_calendar
    weekly = frames.get("ohlc_data.csv")
    if weekly is not None:
        try:
            weekly = weekly.set_index("Time")
            validate_candles(weekly, "ohlc_data.csv")
            validate_calendar(weekly.index, "ohlc_data.csv", step=7)
            # The newest candle is the report week's, never a week the cutoff has not reached.
            report_week = report_date - pd.Timedelta(days=report_date.weekday())
            if pd.Timestamp(weekly.index.max()) != report_week:
                raise ValueError(
                    f"ohlc_data.csv: latest week is {weekly.index.max()}, expected "
                    f"{report_week.date()}"
                )
        except (ValueError, RuntimeError, KeyError) as exc:
            errors.append(str(exc))
    summary = frames.get("report_ohlc_summary.csv")
    if summary is not None:
        try:
            for prefix in ("Daily", "Week-to-Date"):
                candles = summary[[f"{prefix} {c}" for c in ("Open", "High", "Low", "Close")]].copy()
                candles.columns = ["Open", "High", "Low", "Close"]
                validate_candles(candles, "report_ohlc_summary.csv " + prefix)
            row = summary.iloc[0]
            if (pd.Timestamp(row["Week Start"]) != report_date - pd.Timedelta(days=report_date.weekday())
                    or float(row["Week-to-Date Days"]) != report_date.weekday() + 1
                    or float(row["Week-to-Date Close"]) != float(row["Daily Close"])
                    or float(row["Week-to-Date High"]) < float(row["Daily High"])
                    or float(row["Week-to-Date Low"]) > float(row["Daily Low"])):
                raise ValueError("report_ohlc_summary.csv: inconsistent week-to-date candle")
        except (ValueError, KeyError, IndexError) as exc:
            errors.append(f"report_ohlc_summary.csv: {exc}")
    master_path = output_dir / "master_metrics_data.csv.gz"
    summary = frames.get("summary_table.csv")
    if summary is not None and master_path.is_file():
        _validate_investor_sentiment(summary, master_path, report_date, errors)
    fundamentals = frames.get("fundamentals_table.csv")
    if fundamentals is not None and master_path.is_file():
        from data_definitions import FUNDAMENTALS_TEMPLATE
        from report_tables import create_fundamentals_table
        try:
            columns = list(dict.fromkeys(["time"] + [item[0] for group in FUNDAMENTALS_TEMPLATE.values() for item in group.values()]))
            master = pd.read_csv(master_path, usecols=columns, parse_dates=["time"]).set_index("time")
            expected = create_fundamentals_table(master, FUNDAMENTALS_TEMPLATE, report_date)
            change = "7 Day Change (%)"
            if not np.allclose(pd.to_numeric(fundamentals[change], errors="coerce"),
                               expected[change], rtol=1e-10, atol=1e-10, equal_nan=True):
                raise ValueError("7 Day Change (%) differs from source")
            fundamentals = fundamentals.drop(columns=[change])
            expected = expected.drop(columns=[change])
            pd.testing.assert_frame_equal(
                fundamentals.fillna("").astype(str).reset_index(drop=True),
                expected.fillna("").astype(str).reset_index(drop=True), check_dtype=False)
        except (ValueError, KeyError, AssertionError) as exc:
            errors.append(f"fundamentals_table.csv: does not match report-date source values ({str(exc)[:180]})")


def validate_outputs(
    output_dir: str | Path,
    expected_report_date,
    rules: dict[str, RowBounds] | None = None,
    require_release_manifest: bool = False,
) -> list[str]:
    """Return validation errors for generated outputs; an empty list means success."""
    output_dir = Path(output_dir)
    expected_report_date = pd.to_datetime(expected_report_date).normalize()
    rules = OUTPUT_RULES if rules is None else rules
    errors: list[str] = []
    retained_frames: dict[str, pd.DataFrame] = {}

    for filename, bounds in rules.items():
        path = output_dir / filename
        if not path.is_file():
            errors.append(f"{filename}: required output is missing")
            continue
        try:
            row_count, columns, contains_infinity, frame = _scan_csv(
                path, filename in RETAINED_OUTPUTS
            )
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            errors.append(f"{filename}: cannot parse CSV ({exc})")
            continue

        if row_count < bounds.minimum:
            errors.append(
                f"{filename}: {row_count} rows is below minimum {bounds.minimum}"
            )
        if bounds.maximum is not None and row_count > bounds.maximum:
            errors.append(
                f"{filename}: {row_count} rows exceeds maximum {bounds.maximum}"
            )

        missing_columns = REQUIRED_COLUMNS.get(filename, set()) - columns
        if missing_columns:
            errors.append(f"{filename}: missing columns {sorted(missing_columns)}")
        if contains_infinity:
            errors.append(f"{filename}: contains positive or negative infinity")
        if frame is not None:
            retained_frames[filename] = frame

    for filename, column in INDEX_CUTOFF_OUTPUTS.items():
        _validate_index_cutoff(
            output_dir,
            filename,
            column,
            expected_report_date,
            errors,
        )

    _validate_report_agreement(retained_frames, expected_report_date, errors)
    _validate_review_contracts(retained_frames, output_dir, expected_report_date, errors)
    _validate_performance_rows(retained_frames, errors)
    from candle_data import CANDLE_FILES, validate_candle_exports
    if any((output_dir / name).exists() for name in CANDLE_FILES):
        try:
            master = pd.read_csv(output_dir / "master_metrics_data.csv.gz", index_col=0, parse_dates=True, low_memory=False)
            validate_candle_exports(output_dir, master, expected_report_date)
        except (ValueError, RuntimeError, KeyError, OSError, AssertionError) as exc:
            errors.append(f"Chart candle exports: {exc}")
    _validate_release_manifest(output_dir, expected_report_date, errors, require_release_manifest)
    return errors


def _validate_release_manifest(output_dir, expected_report_date, errors, required=False):
    path = Path(output_dir) / "release_manifest.json"
    if not path.is_file():
        if required:
            errors.append("release_manifest.json: required release contract is missing")
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"release_manifest.json: cannot parse manifest ({exc})")
        return
    if manifest.get("schema_version") != 1:
        errors.append("release_manifest.json: unsupported schema_version")
    if manifest.get("release_id") != str(expected_report_date.date()):
        errors.append("release_manifest.json: release_id does not match report date")
    if manifest.get("report_date") != str(expected_report_date.date()):
        errors.append("release_manifest.json: report_date does not match report date")
    files = manifest.get("files")
    expected_files = {name for name in OUTPUT_RULES}
    # Older frozen releases are line-only; a new bundle must be complete and hashed.
    from candle_data import CANDLE_FILES
    if any((Path(output_dir) / name).exists() for name in CANDLE_FILES) or (
        isinstance(files, dict) and any(name in files for name in CANDLE_FILES)
    ):
        expected_files.update(CANDLE_FILES)
    if not isinstance(files, dict) or set(files) != expected_files:
        errors.append("release_manifest.json: file inventory does not match generated outputs")
        return
    for name in expected_files:
        record = files[name]
        target = Path(output_dir) / name
        expected_hash = record.get("sha256") if isinstance(record, dict) else None
        expected_size = record.get("size_bytes") if isinstance(record, dict) else None
        if not isinstance(expected_hash, str) or not target.is_file():
            errors.append(f"release_manifest.json: invalid file record for {name}")
            continue
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
            errors.append(f"release_manifest.json: hash mismatch for {name}")
        if expected_size != target.stat().st_size:
            errors.append(f"release_manifest.json: size mismatch for {name}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="csv", help="Generated CSV directory")
    parser.add_argument(
        "--report-date",
        help="Expected YYYY-MM-DD cutoff (defaults to the release manifest's report_date)",
    )
    return parser.parse_args(argv)


# A release may be validated at most this many days after the clock's report date, so a
# run that crosses UTC midnight still validates while a leftover manifest does not.
MANIFEST_MAX_LAG_DAYS = 1


def _manifest_report_date(output_dir: str | Path, clock_report_date) -> tuple[str | None, str | None]:
    """Return (report_date, error) from the release manifest main.py wrote.

    The pipeline's report date is fixed when main.py starts; recomputing it from the wall
    clock here would expect the next day whenever generation crosses UTC midnight.
    """
    path = Path(output_dir) / "release_manifest.json"
    try:
        report_date = json.loads(path.read_text(encoding="utf-8"))["report_date"]
        manifest_date = pd.to_datetime(report_date).normalize()
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return None, f"release_manifest.json: cannot read report_date ({exc})"
    lag_days = (pd.to_datetime(clock_report_date).normalize() - manifest_date).days
    if not 0 <= lag_days <= MANIFEST_MAX_LAG_DAYS:
        return None, (
            f"release_manifest.json: report_date {manifest_date.date()} is not a current "
            f"release (clock report date {pd.to_datetime(clock_report_date).date()})"
        )
    return str(manifest_date.date()), None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.report_date:
        expected_report_date = args.report_date
    else:
        expected_report_date, error = _manifest_report_date(
            args.output_dir, CLOCK_REPORT_DATE
        )
        if error:
            print("Output validation failed:", file=sys.stderr)
            print(f"- {error}", file=sys.stderr)
            return 1

    errors = validate_outputs(args.output_dir, expected_report_date, require_release_manifest=True)
    if errors:
        print("Output validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"Validated {len(OUTPUT_RULES)} outputs for "
        f"{pd.to_datetime(expected_report_date).date()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
