"""Validate generated report artifacts without fetching or mutating data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RowBounds:
    minimum: int
    maximum: int | None = None


# Bounds are intentionally broad for long-form history, and tight for fixed-shape
# report tables. They catch truncation, header-only files, accidental duplication,
# and runaway exports without coupling validation to today's exact history length.
OUTPUT_RULES = {
    "1k_bucket_table.csv": RowBounds(1, 10_000),
    "5k_bucket_table.csv": RowBounds(1, 10_000),
    "bitcoin_dominance_history.csv": RowBounds(1, 100_000),
    "brk_onchain_raw.csv": RowBounds(365, 100_000),
    "cagr_data.csv": RowBounds(365, 100_000),
    "cycle_low_data.csv": RowBounds(1, 100_000),
    "drawdown_data.csv": RowBounds(1, 100_000),
    "electricity_cost_scenarios.csv": RowBounds(365, 100_000),
    "eoy_model_data.csv": RowBounds(365, 100_000),
    "fundamentals_table.csv": RowBounds(1, 1_000),
    "halving_data.csv": RowBounds(1, 100_000),
    "master_metrics_data.csv.gz": RowBounds(365, 100_000),
    "model_coefficients.csv": RowBounds(6, 6),
    "monthly_heatmap_data.csv": RowBounds(4, 1_000),
    "mtd_return_comparison.csv": RowBounds(2, 10),
    "mtd_returns_history.csv": RowBounds(29, 32),
    "network_model_metrics.csv": RowBounds(365, 100_000),
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
    "1k_bucket_table.csv": {"Price Range ($)", "Count", "Current Price"},
    "model_coefficients.csv": {
        "coefficient", "value", "report_date", "fit_end_date",
    },
    "5k_bucket_table.csv": {"Price Range ($)", "Count", "Current Price"},
    "bitcoin_dominance_history.csv": {
        "date", "bitcoin_dominance", "source_updated_at",
    },
    "brk_onchain_raw.csv": {"timestamp", "price_close"},
    "cagr_data.csv": {"time", "price_close_2_Year_CAGR", "price_close_4_Year_CAGR"},
    "cycle_low_data.csv": {"days_since_cycle_low", "index_value", "Cycle"},
    "drawdown_data.csv": {"days_since_ath", "drawdown_pct", "Cycle"},
    "electricity_cost_scenarios.csv": {
        "date",
        "BTC Price",
        "Fleet Efficiency (J/GH)",
        "Network Power Draw (W)",
        "Daily Electricity Consumption (kWh)",
        "Subsidy (BTC)",
        "Fees (BTC)",
        "Miner Revenue (BTC)",
        "Power Expense ($0.03/kWh)",
        "Power Expense ($0.04/kWh)",
        "Power Expense ($0.05/kWh)",
        "Power Expense ($0.06/kWh)",
        "Power Expense ($0.07/kWh)",
        "Power-Only Break-Even Tariff ($/kWh)",
        "Legacy PUE/Subsidy-Only Cost",
        "Bitcoin Production Cost",
        "Hayes Network Price",
        "Energy Value",
    },
    "eoy_model_data.csv": {"time", "price_close", "price_close_4_Year_CAGR"},
    "fundamentals_table.csv": {"Section", "Metric", "Current Value"},
    "halving_data.csv": {"days_since_halving", "index_value", "Era"},
    "master_metrics_data.csv.gz": {
        "time", "price_close", "market_cap", "metcalfe_value",
        "power_law_price", "60_day_ma_hash_rate", "hash_ribbon_capitulation",
    },
    "monthly_heatmap_data.csv": {
        "time", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
        "Aug", "Sep", "Oct", "Nov", "Dec", "Yearly",
    },
    "mtd_return_comparison.csv": {
        "Year", "End Price ($)", "Return (%)", "Report Date Return (%)",
    },
    "mtd_returns_history.csv": {"day", "Median", "Average"},
    "network_model_metrics.csv": {
        "date",
        "BTC Price",
        "Bitcoin Market Cap",
        "Bitcoin Supply",
        "Non-Zero Address Count",
        "Addresses Holding 0.001+ BTC",
        "Addresses Holding 0.01+ BTC",
        "Addresses Holding 0.1+ BTC",
        "Metcalfe Value (Any Balance)",
        "Metcalfe Value (0.001+ BTC)",
        "Metcalfe Value (0.01+ BTC)",
        "Metcalfe Value (0.1+ BTC)",
        "Metcalfe Scale (Any Balance)",
        "Metcalfe Scale (0.001+ BTC)",
        "Metcalfe Scale (0.01+ BTC)",
        "Metcalfe Scale (0.1+ BTC)",
        "BTC Price / Metcalfe Value",
        "Power Law Price",
        "BTC Price / Power Law Price",
        "Days Since Genesis",
        "Power Law Exponent",
        "Power Law Scale",
        "Hash Rate (H/s)",
        "Hash Rate 30-Day MA (H/s)",
        "Hash Rate 60-Day MA (H/s)",
        "Hash Ribbon 30D / 60D",
        "Hash Ribbon Capitulation",
    },
    "ohlc_data.csv": {"Time", "Open", "High", "Low", "Close"},
    "onchain_price_models.csv": {
        "date", "BTC Price", "Electricity Cost", "Metcalfe Value", "Power Law Price",
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
    "Bitcoin Fear & Greed Index",
}


RETAINED_OUTPUTS = {
    "1k_bucket_table.csv",
    "5k_bucket_table.csv",
    "bitcoin_dominance_history.csv",
    "cycle_low_data.csv",
    "eoy_model_data.csv",
    "electricity_cost_scenarios.csv",
    "halving_data.csv",
    "model_coefficients.csv",
    "monthly_heatmap_data.csv",
    "mtd_return_comparison.csv",
    "mtd_returns_history.csv",
    "network_model_metrics.csv",
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
# only. The master is ~99MB raw; holding it the way RETAINED_OUTPUTS does would be
# wasteful when the only thing left to assert is the cutoff.
INDEX_CUTOFF_OUTPUTS = {
    "master_metrics_data.csv.gz": "time",
    "cagr_data.csv": "time",
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

    dates = pd.to_datetime(index, errors="coerce").dt.normalize()
    if dates.isna().any():
        errors.append(f"{filename}: {column!r} contains invalid or missing dates")
        return
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
        "eoy_model_data.csv": ("time", "price_close"),
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

    for filename in ("1k_bucket_table.csv", "5k_bucket_table.csv"):
        frame = frames.get(filename)
        if frame is None or "Current Price" not in frame.columns:
            continue
        values = _unique_numeric_values(frame, "Current Price")
        if len(values) == 1:
            prices[filename] = values[0]
        else:
            errors.append(f"{filename}: expected one shared Current Price value")

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


def _validate_electricity_scenarios(
    frames: dict[str, pd.DataFrame],
    expected_report_date: pd.Timestamp,
    errors: list[str],
) -> None:
    filename = "electricity_cost_scenarios.csv"
    frame = frames.get(filename)
    if frame is None or frame.empty:
        return

    _validate_dated_output(
        frames, filename, "date", expected_report_date, errors
    )

    numeric_columns = [
        "BTC Price",
        "Daily Electricity Consumption (kWh)",
        "Subsidy (BTC)",
        "Fees (BTC)",
        "Miner Revenue (BTC)",
        "Power-Only Break-Even Tariff ($/kWh)",
    ]
    tariffs = (0.03, 0.04, 0.05, 0.06, 0.07)
    tariff_columns = [f"Power Expense (${tariff:.2f}/kWh)" for tariff in tariffs]
    if not set(numeric_columns + tariff_columns).issubset(frame.columns):
        return

    numeric = frame[numeric_columns + tariff_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = (
        numeric["Daily Electricity Consumption (kWh)"].gt(0)
        & numeric["Miner Revenue (BTC)"].gt(0)
    )
    if not valid.any():
        errors.append(f"{filename}: contains no valid positive energy/revenue rows")
        return

    revenue_expected = numeric["Subsidy (BTC)"] + numeric["Fees (BTC)"]
    if not np.allclose(
        numeric.loc[valid, "Miner Revenue (BTC)"],
        revenue_expected.loc[valid],
        rtol=1e-10,
        atol=1e-8,
    ):
        errors.append(f"{filename}: miner revenue does not equal subsidy plus fees")

    for tariff, column in zip(tariffs, tariff_columns):
        expected = (
            numeric["Daily Electricity Consumption (kWh)"]
            * tariff
            / numeric["Miner Revenue (BTC)"]
        )
        if not np.allclose(
            numeric.loc[valid, column],
            expected.loc[valid],
            rtol=1e-10,
            atol=1e-6,
        ):
            errors.append(
                f"{filename}: {column!r} does not match energy × tariff ÷ miner revenue"
            )

    breakeven_expected = (
        numeric["BTC Price"]
        * numeric["Miner Revenue (BTC)"]
        / numeric["Daily Electricity Consumption (kWh)"]
    )
    if not np.allclose(
        numeric.loc[valid, "Power-Only Break-Even Tariff ($/kWh)"],
        breakeven_expected.loc[valid],
        rtol=1e-10,
        atol=1e-10,
    ):
        errors.append(f"{filename}: break-even tariff does not match its inputs")

    onchain = frames.get("onchain_price_models.csv")
    if onchain is not None and {"date", "Electricity Cost"}.issubset(onchain.columns):
        scenario = frame[["date", "Power Expense ($0.05/kWh)"]].copy()
        comparison = scenario.merge(
            onchain[["date", "Electricity Cost"]], on="date", how="inner"
        ).dropna()
        if comparison.empty or not np.allclose(
            comparison["Power Expense ($0.05/kWh)"],
            comparison["Electricity Cost"],
            rtol=1e-10,
            atol=1e-6,
        ):
            errors.append(
                "onchain_price_models.csv: Electricity Cost disagrees with the "
                "canonical $0.05/kWh power-expense scenario"
            )


def _validate_network_models(
    frames: dict[str, pd.DataFrame],
    expected_report_date: pd.Timestamp,
    errors: list[str],
) -> None:
    filename = "network_model_metrics.csv"
    frame = frames.get(filename)
    if frame is None or frame.empty:
        return
    _validate_dated_output(frames, filename, "date", expected_report_date, errors)

    numeric_columns = [
        column for column in REQUIRED_COLUMNS[filename]
        if column not in {"date", "Hash Ribbon Capitulation"}
    ]
    if not set(numeric_columns).issubset(frame.columns):
        return
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

    metcalfe_models = (
        (
            "Non-Zero Address Count",
            "Metcalfe Value (Any Balance)",
            "Metcalfe Scale (Any Balance)",
        ),
        (
            "Addresses Holding 0.001+ BTC",
            "Metcalfe Value (0.001+ BTC)",
            "Metcalfe Scale (0.001+ BTC)",
        ),
        (
            "Addresses Holding 0.01+ BTC",
            "Metcalfe Value (0.01+ BTC)",
            "Metcalfe Scale (0.01+ BTC)",
        ),
        (
            "Addresses Holding 0.1+ BTC",
            "Metcalfe Value (0.1+ BTC)",
            "Metcalfe Scale (0.1+ BTC)",
        ),
    )
    for address_column, value_column, scale_column in metcalfe_models:
        valid = (
            numeric["BTC Price"].gt(0)
            & numeric["Bitcoin Supply"].gt(0)
            & numeric[address_column].gt(0)
        )
        if not valid.any():
            errors.append(f"{filename}: no valid rows for {value_column}")
            continue
        expected_scale = np.exp(
            (
                np.log(
                    numeric.loc[valid, "BTC Price"]
                    * numeric.loc[valid, "Bitcoin Supply"]
                )
                - 2 * np.log(numeric.loc[valid, address_column])
            ).mean()
        )
        if not np.allclose(
            numeric.loc[valid, scale_column], expected_scale, rtol=1e-10, atol=0
        ):
            errors.append(f"{filename}: {scale_column} does not match the fitted scale")
        expected_value = (
            expected_scale
            * numeric.loc[valid, address_column].pow(2)
            / numeric.loc[valid, "Bitcoin Supply"]
        )
        if not np.allclose(
            numeric.loc[valid, value_column], expected_value, rtol=1e-10, atol=1e-6
        ):
            errors.append(f"{filename}: {value_column} does not match scale × n² ÷ supply")

    power_valid = numeric["BTC Price"].gt(0) & numeric["Days Since Genesis"].gt(0)
    if power_valid.sum() >= 2:
        exponent, log_scale = np.polyfit(
            np.log(numeric.loc[power_valid, "Days Since Genesis"]),
            np.log(numeric.loc[power_valid, "BTC Price"]),
            1,
        )
        scale = np.exp(log_scale)
        expected_power = (
            scale * numeric.loc[power_valid, "Days Since Genesis"].pow(exponent)
        )
        if not np.allclose(
            numeric.loc[power_valid, "Power Law Exponent"], exponent,
            rtol=1e-10, atol=0,
        ):
            errors.append(f"{filename}: Power Law Exponent does not match the fit")
        if not np.allclose(
            numeric.loc[power_valid, "Power Law Scale"], scale,
            rtol=1e-10, atol=0,
        ):
            errors.append(f"{filename}: Power Law Scale does not match the fit")
        if not np.allclose(
            numeric.loc[power_valid, "Power Law Price"], expected_power,
            rtol=1e-10, atol=1e-6,
        ):
            errors.append(f"{filename}: Power Law Price does not match scale × age^exponent")
    else:
        errors.append(f"{filename}: fewer than two positive rows for the power-law fit")

    hash_rate = numeric["Hash Rate (H/s)"]
    fast = hash_rate.rolling(30).mean()
    slow = hash_rate.rolling(60).mean()
    check = slow.notna() & slow.ne(0)
    if check.any():
        if not np.allclose(
            numeric.loc[check, "Hash Rate 30-Day MA (H/s)"], fast.loc[check],
            rtol=1e-10, atol=1,
        ) or not np.allclose(
            numeric.loc[check, "Hash Rate 60-Day MA (H/s)"], slow.loc[check],
            rtol=1e-10, atol=1,
        ):
            errors.append(f"{filename}: hash-rate moving averages do not match 30/60-day means")
        expected_ratio = fast.loc[check] / slow.loc[check]
        if not np.allclose(
            numeric.loc[check, "Hash Ribbon 30D / 60D"], expected_ratio,
            rtol=1e-10, atol=1e-12,
        ):
            errors.append(f"{filename}: hash-ribbon ratio does not equal 30D ÷ 60D")
        actual_state = (
            frame.loc[check, "Hash Ribbon Capitulation"].astype(str).str.lower()
        )
        expected_state = (fast.loc[check] < slow.loc[check]).astype(str).str.lower()
        if not actual_state.equals(expected_state):
            errors.append(f"{filename}: hash-ribbon state disagrees with its averages")

    onchain = frames.get("onchain_price_models.csv")
    if onchain is not None and {
        "date", "Metcalfe Value", "Power Law Price"
    }.issubset(onchain.columns):
        detailed = frame[
            ["date", "Metcalfe Value (Any Balance)", "Power Law Price"]
        ].rename(columns={"Metcalfe Value (Any Balance)": "Metcalfe Value"})
        comparison = detailed.merge(
            onchain[["date", "Metcalfe Value", "Power Law Price"]],
            on="date",
            how="inner",
            suffixes=("_detail", "_onchain"),
        ).dropna()
        for column in ("Metcalfe Value", "Power Law Price"):
            if comparison.empty or not np.allclose(
                comparison[f"{column}_detail"], comparison[f"{column}_onchain"],
                rtol=1e-10, atol=1e-6,
            ):
                errors.append(
                    f"onchain_price_models.csv: {column} disagrees with {filename}"
                )


def _validate_report_agreement(
    frames: dict[str, pd.DataFrame],
    expected_report_date: pd.Timestamp,
    errors: list[str],
) -> None:
    for filename, column in (
        ("onchain_price_models.csv", "date"),
        ("eoy_model_data.csv", "time"),
    ):
        _validate_dated_output(
            frames, filename, column, expected_report_date, errors
        )
    _validate_dated_output(
        frames,
        "report_ohlc_summary.csv",
        "Report Date",
        expected_report_date,
        errors,
        require_every_row=True,
    )
    # Every coefficient row must name the release it was fitted for; a row carrying an
    # older report date means the file was not regenerated with the rest of the release.
    for column in ("report_date", "fit_end_date"):
        _validate_dated_output(
            frames,
            "model_coefficients.csv",
            column,
            expected_report_date,
            errors,
            require_every_row=True,
        )

    dominance = frames.get("bitcoin_dominance_history.csv")
    if dominance is not None and not dominance.empty:
        _validate_dated_output(
            frames,
            "bitcoin_dominance_history.csv",
            "date",
            expected_report_date,
            errors,
        )
        dates = _normalized_dates(
            dominance,
            "date",
            "bitcoin_dominance_history.csv",
            errors,
        )
        if dates is not None and dates.duplicated().any():
            errors.append("bitcoin_dominance_history.csv: contains duplicate dates")
        values = pd.to_numeric(dominance["bitcoin_dominance"], errors="coerce")
        if values.isna().any() or not values.between(0, 100, inclusive="neither").all():
            errors.append(
                "bitcoin_dominance_history.csv: dominance must be numeric and between 0 and 100"
            )
        source_updated_at = pd.to_datetime(
            dominance["source_updated_at"], errors="coerce", utc=True
        )
        if source_updated_at.isna().any():
            errors.append(
                "bitcoin_dominance_history.csv: contains invalid source_updated_at values"
            )

        summary = frames.get("summary_table.csv")
        if summary is not None and {"Metric", "Value"}.issubset(summary.columns):
            report_value = values.loc[dates.eq(expected_report_date)] if dates is not None else []
            summary_value = pd.to_numeric(
                summary.loc[
                    summary["Metric"].eq("Bitcoin Dominance"), "Value"
                ],
                errors="coerce",
            ).dropna()
            if (
                len(report_value) != 1
                or len(summary_value) != 1
                or not np.isclose(report_value.iloc[0], summary_value.iloc[0])
            ):
                errors.append(
                    "summary_table.csv: Bitcoin Dominance disagrees with report-date history"
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
    _validate_electricity_scenarios(frames, expected_report_date, errors)
    _validate_network_models(frames, expected_report_date, errors)


def validate_outputs(
    output_dir: str | Path,
    expected_report_date,
    rules: dict[str, RowBounds] | None = None,
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
            output_dir, filename, column, expected_report_date, errors
        )

    _validate_report_agreement(retained_frames, expected_report_date, errors)
    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="csv", help="Generated CSV directory")
    parser.add_argument(
        "--report-date",
        help="Expected YYYY-MM-DD cutoff (defaults to data_definitions.report_date)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.report_date:
        expected_report_date = args.report_date
    else:
        # Local configuration only; importing data_definitions performs no I/O.
        from data_definitions import report_date

        expected_report_date = report_date

    errors = validate_outputs(args.output_dir, expected_report_date)
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
