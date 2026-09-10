"""Small shared contracts for source time series and published candles."""
import numpy as np
import pandas as pd


def validate_calendar(index, label, step=1):
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if (dates.empty or dates.hasnans or dates.has_duplicates
            or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize())
            or not dates.equals(pd.date_range(dates[0], dates[-1], freq=f"{step}D"))):
        raise RuntimeError(f"{label}: dates must be unique, ordered, and complete at {step}-day intervals")


def validate_candles(frame, label):
    columns = ["Open", "High", "Low", "Close"]
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
