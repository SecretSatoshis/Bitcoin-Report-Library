"""Chart-ready cycle series: ATH drawdowns, returns from cycle lows, and halving eras."""

import pandas as pd

from metrics import bitcoin_halving_dates


def _ordinal(n: int) -> str:
    """Return 1 -> '1st', 2 -> '2nd', 11 -> '11th'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _era_label(position: int) -> str:
    """Halving era name by position (0 = genesis era)."""
    return "Genesis Era" if position == 0 else f"{_ordinal(position + 1)} Era"


def _build_period_bounds(boundaries, data_end, label_func):
    """Turn ascending boundary dates into (label, start, end) windows, oldest first.

    Windows are half-open, [start, end), so a boundary date belongs to one period only.
    The last window ends the day after `data_end`.
    """
    data_end = pd.Timestamp(data_end)
    horizon = data_end + pd.Timedelta(days=1)
    periods = []

    for i, start in enumerate(boundaries):
        start = pd.Timestamp(start)
        if start > data_end:
            break
        end = min(pd.Timestamp(boundaries[i + 1]), horizon) if i + 1 < len(boundaries) else horizon
        periods.append((label_func(i), start, end))

    return periods


# Approximate cycle-low dates. Each bounds a search window; the export starts at the
# lowest price actually inside it.
BITCOIN_CYCLE_LOW_DATES = [
    "2010-07-25",
    "2011-11-18",
    "2015-01-15",
    "2018-12-16",
    "2022-11-20",
    "2026-02-06",
]


# (label, ATH date, date the ATH was reclaimed). Cycles are not contiguous: the gaps are
# time spent at new highs. The open cycle has no end date.
BITCOIN_DRAWDOWN_CYCLES = [
    ("Drawdown Cycle 1", "2011-06-08", "2013-02-28"),
    ("Drawdown Cycle 2", "2013-11-29", "2017-03-03"),
    ("Drawdown Cycle 3", "2017-12-17", "2020-12-16"),
    ("Drawdown Cycle 4", "2021-11-10", "2024-03-04"),
    ("Drawdown Cycle 5", "2025-10-06", None),
]


def compute_drawdowns(data: pd.DataFrame) -> pd.DataFrame:
    """Long-form drawdown series per cycle: days_since_ath, drawdown_pct (0 at the ATH,
    negative below it) and Cycle. Day 0 is the cycle's starting ATH."""
    df = data[["price_close"]].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if df.empty:
        return pd.DataFrame(columns=["days_since_ath", "drawdown_pct", "Cycle"])

    data_end = df.index.max()
    out = []

    for cycle_name, start_date, end_date in BITCOIN_DRAWDOWN_CYCLES:
        start_dt = pd.to_datetime(start_date)
        # The open cycle ends at the latest observation, not today's date.
        end_dt = data_end if end_date is None else pd.to_datetime(end_date)

        period = df.loc[(df.index >= start_dt) & (df.index <= end_dt)].copy()
        if period.empty:
            continue

        period["ath"] = period["price_close"].cummax()
        period["drawdown_pct"] = (period["price_close"] / period["ath"] - 1.0) * 100.0
        period["days_since_ath"] = (period.index - start_dt).days

        period["Cycle"] = cycle_name

        out.append(period[["days_since_ath", "drawdown_pct", "Cycle"]])

    if not out:
        return pd.DataFrame(columns=["days_since_ath", "drawdown_pct", "Cycle"])

    return pd.concat(out, ignore_index=True)


def compute_cycle_lows(data: pd.DataFrame) -> pd.DataFrame:
    """Long-form price series per market cycle, indexed to its low: days_since_cycle_low,
    index_value (1.0 at the low, 2.0 = 2x) and Cycle."""
    df = data[["price_close"]].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if df.empty:
        return pd.DataFrame(columns=["days_since_cycle_low", "index_value", "Cycle"])

    cycle_periods = _build_period_bounds(
        BITCOIN_CYCLE_LOW_DATES,
        df.index.max(),
        lambda i: f"Market Cycle {i + 1}",
    )

    out = []
    for cycle_name, start_dt, end_dt in cycle_periods:
        period = df.loc[(df.index >= start_dt) & (df.index < end_dt)].copy()
        if period.empty:
            continue

        valid_prices = period["price_close"].dropna()
        valid_prices = valid_prices[valid_prices > 0]
        if valid_prices.empty:
            continue

        # Start at the real low so the series never drops below 1.0.
        low_date = valid_prices.idxmin()
        low_px = float(valid_prices.loc[low_date])
        period = period.loc[period.index >= low_date].copy()

        period["days_since_cycle_low"] = (period.index - low_date).days
        period["index_value"] = period["price_close"] / low_px
        period["Cycle"] = cycle_name

        out.append(period[["days_since_cycle_low", "index_value", "Cycle"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["days_since_cycle_low", "index_value", "Cycle"]
    )


def compute_halving_days(data: pd.DataFrame) -> pd.DataFrame:
    """Long-form price series per halving era, indexed to the halving-day price:
    days_since_halving, index_value (1.0 at the halving, 2.0 = 2x) and Era."""
    data = data[["price_close"]].copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    if data.empty:
        return pd.DataFrame(columns=["days_since_halving", "index_value", "Era"])

    # Eras come from the halving schedule, so the next halving starts a new era by itself.
    data_end = data.index.max()
    eras = _build_period_bounds(bitcoin_halving_dates(through=data_end), data_end, _era_label)

    out = []

    for era_name, start_dt, end_dt in eras:
        period = data.loc[(data.index >= start_dt) & (data.index < end_dt)].copy()
        if period.empty:
            continue

        # An era needs a real price on its halving day. Genesis has none, so it is omitted.
        valid_prices = period["price_close"].dropna()
        valid_prices = valid_prices[valid_prices > 0]
        if valid_prices.empty or start_dt not in valid_prices.index:
            continue

        start_px = float(valid_prices.loc[start_dt])

        period["days_since_halving"] = (period.index - start_dt).days
        period["index_value"] = period["price_close"] / start_px
        period["Era"] = era_name

        out.append(period[["days_since_halving", "index_value", "Era"]])

    if not out:
        return pd.DataFrame(columns=["days_since_halving", "index_value", "Era"])

    return pd.concat(out, ignore_index=True)
