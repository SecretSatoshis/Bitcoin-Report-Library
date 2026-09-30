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
    """
    Turn ascending boundary dates into (label, start, end) windows.

    Windows are half-open — [start, end) — so a date that both closes one period and
    opens the next is attributed to exactly one of them. The final window extends one
    day past `data_end` so the last observation is retained.

    Returns:
    list[tuple]: (label, start Timestamp, end Timestamp) per period, oldest first.
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


# Market cycle boundaries define the search window for each cycle. The exported
# series is anchored to the lowest positive price actually observed inside that
# window, since a hand-entered boundary can lead the final low by a few days (and
# an open cycle can make a later low before it is complete).
BITCOIN_CYCLE_LOW_DATES = [
    "2010-07-25",
    "2011-11-18",
    "2015-01-15",
    "2018-12-16",
    "2022-11-20",
    "2026-02-06",
]


# Drawdown cycles run from an all-time high until that high is reclaimed, so unlike
# market cycles they are NOT contiguous — the gaps between them are the periods spent
# at new highs. Each entry is (label, ATH date, recovery date); the open cycle's end is
# supplied from the data rather than hardcoded.
BITCOIN_DRAWDOWN_CYCLES = [
    ("Drawdown Cycle 1", "2011-06-08", "2013-02-28"),
    ("Drawdown Cycle 2", "2013-11-29", "2017-03-03"),
    ("Drawdown Cycle 3", "2017-12-17", "2020-12-16"),
    ("Drawdown Cycle 4", "2021-11-10", "2024-03-04"),
    ("Drawdown Cycle 5", "2025-10-06", None),  # open — ends at latest data
]


def compute_drawdowns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Long-form drawdown series:
      - days_since_ath
      - drawdown_pct (0 at ATH, negative when below ATH)
      - Cycle (label)

    Aligns each cycle to its start ATH date.
    """
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
        # An open cycle ends at the latest observation. Deriving it from the data rather
        # than today's clock keeps the export deterministic and stops the window running
        # past the data whenever the pipeline is re-run or replayed.
        end_dt = data_end if end_date is None else pd.to_datetime(end_date)

        period = df.loc[(df.index >= start_dt) & (df.index <= end_dt)].copy()
        if period.empty:
            continue

        # ATH path within the period
        period["ath"] = period["price_close"].cummax()

        # Drawdown percent
        period["drawdown_pct"] = (period["price_close"] / period["ath"] - 1.0) * 100.0

        # Days since the cycle's ATH start anchor (your start_dt)
        period["days_since_ath"] = (period.index - start_dt).days

        period["Cycle"] = cycle_name

        out.append(period[["days_since_ath", "drawdown_pct", "Cycle"]])

    if not out:
        return pd.DataFrame(columns=["days_since_ath", "drawdown_pct", "Cycle"])

    return pd.concat(out, ignore_index=True)


def compute_cycle_lows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market cycle performance indexed from cycle lows.

    Returns DataFrame with:
      - days_since_cycle_low
      - index_value (1.0 at cycle low, 2.0 = 2x gain, etc.)
      - Cycle (label)
    """
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
        # Half-open window: each cycle low both ends one cycle and begins the next, so an
        # inclusive end would emit that date twice under two different cycle labels.
        period = df.loc[(df.index >= start_dt) & (df.index < end_dt)].copy()
        if period.empty:
            continue

        valid_prices = period["price_close"].dropna()
        valid_prices = valid_prices[valid_prices > 0]
        if valid_prices.empty:
            continue

        # Use the actual lowest positive observation in the window. Starting the
        # export at that row guarantees the documented 1.0 floor and avoids
        # presenting a provisional/configured boundary as a confirmed cycle low.
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
    """
    Build a single long-form dataframe with:
      - days_since_halving
      - index_value (cycle index; 1.0 at halving, 2.0 = 2x, etc.)
      - Era (string name)
    """
    # Ensure datetime index
    data = data[["price_close"]].copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Ensure sorted
    data = data.sort_index()

    if data.empty:
        return pd.DataFrame(columns=["days_since_halving", "index_value", "Era"])

    # Eras are derived from the halving schedule rather than hardcoded, so the next
    # halving splits a new era automatically instead of silently stretching the current
    # one across two halvings.
    data_end = data.index.max()
    eras = _build_period_bounds(bitcoin_halving_dates(through=data_end), data_end, _era_label)

    out = []

    for era_name, start_dt, end_dt in eras:
        # Half-open window: the halving date itself belongs to the era it starts, so
        # closing the previous era inclusively would duplicate every boundary date.
        period = data.loc[(data.index >= start_dt) & (data.index < end_dt)].copy()
        if period.empty:
            continue

        # A halving comparison must have a real price on the halving boundary.
        # Genesis predates the first positive source price, so silently anchoring it
        # hundreds of days later mislabels both the x-axis and the 1.0 baseline.
        # Omit such an era instead; normal halving eras retain day 0 at index 1.0.
        valid_prices = period["price_close"].dropna()
        valid_prices = valid_prices[valid_prices > 0]
        if valid_prices.empty or start_dt not in valid_prices.index:
            continue

        start_px = float(valid_prices.loc[start_dt])

        period["days_since_halving"] = (period.index - start_dt).days
        period["index_value"] = period["price_close"] / start_px  # 1.0 at halving
        period["Era"] = era_name

        out.append(period[["days_since_halving", "index_value", "Era"]])

    if not out:
        return pd.DataFrame(columns=["days_since_halving", "index_value", "Era"])

    return pd.concat(out, ignore_index=True)
