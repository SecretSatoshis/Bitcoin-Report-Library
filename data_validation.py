"""Small shared contracts for source time series and published candles."""
import numpy as np
import pandas as pd

OHLC_COLUMNS = ["Open", "High", "Low", "Close"]


def validate_calendar(index, label, step=1):
    """Require dates that are unique, ordered, at midnight and complete at `step`-day spacing."""
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if (dates.empty or dates.hasnans or dates.has_duplicates
            or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize())
            or not dates.equals(pd.date_range(dates[0], dates[-1], freq=f"{step}D"))):
        raise RuntimeError(f"{label}: dates must be unique, ordered, and complete at {step}-day intervals")


def validate_candles(frame, label):
    """Require finite, positive candles whose High and Low bound Open and Close, one per date."""
    columns = OHLC_COLUMNS
    if frame.empty or not set(columns).issubset(frame.columns):
        raise ValueError(f"{label}: missing OHLC candles/columns")
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
             & (values.High >= values[["Open", "Close", "Low"]].max(axis=1))
             & (values.Low <= values[["Open", "Close", "High"]].min(axis=1)))
    if not valid.all():
        raise ValueError(f"{label}: invalid OHLC candle values or high/low ordering")
    if frame.index.has_duplicates:
        raise ValueError(f"{label}: duplicate candle dates")


def assert_ohlc_usable(ohlc_data: pd.DataFrame, label: str = "OHLC") -> None:
    """Raise before publication when an OHLC frame has no complete numeric candle."""
    if ohlc_data is None or ohlc_data.empty:
        raise RuntimeError(f"{label} data is empty; refusing to overwrite OHLC outputs")

    missing = [column for column in OHLC_COLUMNS if column not in ohlc_data.columns]
    if missing:
        raise RuntimeError(
            f"{label} data is missing required columns {missing}; refusing to overwrite OHLC outputs"
        )

    validate_candles(ohlc_data, label)
