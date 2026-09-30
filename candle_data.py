"""Daily, weekly and monthly Bitcoin candles cut off at the report date, and the master
data rows at each week and month close."""
from pathlib import Path

import numpy as np
import pandas as pd

from data_validation import validate_calendar, validate_candles

CANDLE_FILES = ('bitcoin_candles.csv.gz', 'weekly_metrics_data.csv.gz', 'monthly_metrics_data.csv.gz')


def _cutoff_daily_candles(daily, report_date, label):
    """Daily candles through the report date, from the first priced day, validated."""
    cutoff = pd.Timestamp(report_date).normalize()
    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.loc[daily.index <= cutoff]
    # Drop the all-zero pre-market candles; a gap after them still fails validation.
    nonzero = daily[['Open', 'High', 'Low', 'Close']].ne(0).any(axis=1)
    daily = daily.loc[nonzero.idxmax():] if nonzero.any() else daily.iloc[:0]
    validate_candles(daily, label)
    validate_calendar(daily.index, label)
    if daily.index[-1] != cutoff:
        raise ValueError(f'{label} must reach the report date')
    return daily


def period_candles(daily, interval, frequency):
    """Aggregate validated daily candles into one record per calendar period."""
    records = []
    for period, rows in daily.groupby(daily.index.to_period(frequency)):
        start, end = period.start_time.normalize(), period.end_time.normalize()
        # Skip the first period when the history starts partway through it.
        if rows.index[0] != start:
            continue
        observed = rows.index[-1]
        records.append({'interval': interval, 'period_start': start, 'period_end': end,
                        'observation_date': observed, 'complete': observed == end,
                        'Open': rows.Open.iloc[0], 'High': rows.High.max(),
                        'Low': rows.Low.min(), 'Close': rows.Close.iloc[-1]})
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError(f'No {interval} candle history')
    return result


def build_candle_tables(daily, master, report_date):
    """Daily/weekly/monthly candles and the master rows at each week and month close.

    Snapshot rows are indexed by `period_start`; `observation_date` is the row they copy.

    Daily closes must agree with the master's price_close. Returns {filename: frame}.
    """
    cutoff = pd.Timestamp(report_date).normalize()
    daily = _cutoff_daily_candles(daily, report_date, 'Chart daily candles')
    if cutoff not in master.index:
        raise ValueError('Chart candles and master must reach the report date')
    closes = pd.to_numeric(master['price_close'].reindex(daily.index), errors='coerce')
    if not np.allclose(daily.Close, closes, rtol=1e-9, atol=1e-8):
        raise ValueError('Daily candle closes disagree with master prices')
    tables, candles = {}, []
    for interval, frequency, filename in [('daily', 'D', None), ('weekly', 'W-SUN', CANDLE_FILES[1]), ('monthly', 'M', CANDLE_FILES[2])]:
        result = period_candles(daily, interval, frequency)
        candles.append(result)
        if filename:
            # reindex, not groupby().last(), which would skip nulls to an earlier row.
            snapshot = master.reindex(pd.DatetimeIndex(result.observation_date)).copy()
            snapshot.index = pd.DatetimeIndex(result.period_start, name='period_start')
            snapshot.insert(0, 'observation_date', pd.DatetimeIndex(result.observation_date))
            tables[filename] = snapshot
    tables[CANDLE_FILES[0]] = pd.concat(candles, ignore_index=True)
    return tables


def write_candle_tables(daily, master, report_date, output_dir='csv'):
    """Build the candle tables and write them, gzipped, to output_dir."""
    tables = build_candle_tables(daily, master, report_date)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=filename != CANDLE_FILES[0], compression='gzip', date_format='%Y-%m-%d')
    return tables


def validate_candle_exports(output_dir, master, report_date):
    """Rebuild the candle tables from the published files and require an exact match."""
    output = Path(output_dir)
    candles = pd.read_csv(output / CANDLE_FILES[0], parse_dates=['period_start', 'period_end', 'observation_date'])
    daily = candles.loc[candles.interval.eq('daily')].set_index('period_start')[['Open', 'High', 'Low', 'Close']]
    expected = build_candle_tables(daily, master, report_date)
    pd.testing.assert_frame_equal(candles, expected[CANDLE_FILES[0]], check_dtype=False, rtol=1e-9, atol=1e-8)
    for filename in CANDLE_FILES[1:]:
        actual = pd.read_csv(output / filename, index_col=0, parse_dates=[0, 1], low_memory=False)
        pd.testing.assert_frame_equal(actual, expected[filename], check_dtype=False, check_freq=False, rtol=1e-9, atol=1e-8)
