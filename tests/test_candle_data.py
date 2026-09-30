import unittest
import numpy as np
import pandas as pd
from candle_data import build_candle_tables, CANDLE_FILES


class CandleTests(unittest.TestCase):
    def fixture(self):
        dates = pd.date_range('2024-01-01','2024-03-04')
        close = np.arange(len(dates),dtype=float)+100
        daily = pd.DataFrame({'Open':close-1,'High':close+3,'Low':close-2,'Close':close},index=dates)
        master = pd.DataFrame({'price_close':close,'moving_average':close/2},index=dates)
        return daily, master

    def test_periods_freeze_leap_day_and_exact_null_observation(self):
        daily, master = self.fixture()
        master.loc['2024-02-29','moving_average'] = np.nan
        tables = build_candle_tables(daily,master,'2024-03-02')
        candles = tables[CANDLE_FILES[0]]
        feb = candles.loc[(candles.interval=='monthly') & (candles.period_start==pd.Timestamp('2024-02-01'))].iloc[0]
        self.assertEqual(feb.period_end,pd.Timestamp('2024-02-29'))
        self.assertTrue(feb.complete)
        self.assertEqual(feb.Open,daily.loc['2024-02-01','Open'])
        self.assertEqual(feb.High,daily.loc['2024-02-29','High'])
        self.assertTrue(pd.isna(tables[CANDLE_FILES[2]].loc['2024-02-01','moving_average']))
        week = candles.loc[candles.interval=='weekly'].iloc[-1]
        self.assertEqual(week.period_start,pd.Timestamp('2024-02-26'))
        self.assertFalse(week.complete)
        self.assertEqual(week.Close,daily.loc['2024-03-02','Close'])
        self.assertLessEqual(candles.observation_date.max(),pd.Timestamp('2024-03-02'))

    def test_missing_day_and_mismatched_price_rejected(self):
        daily, master = self.fixture()
        with self.assertRaises(RuntimeError):
            build_candle_tables(daily.drop(pd.Timestamp('2024-02-12')),master,'2024-03-02')
        master.loc['2024-02-12','price_close'] += 1
        with self.assertRaises(ValueError):
            build_candle_tables(daily,master,'2024-03-02')

    def test_release_manifest_accepts_complete_candle_bundle_only(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from release_manifest import write_release_manifest
        from validate_outputs import _validate_release_manifest
        with TemporaryDirectory() as directory, patch('validate_outputs.OUTPUT_RULES', {}):
            output = Path(directory)
            for filename in CANDLE_FILES:
                (output / filename).write_bytes(b'fixture')
            write_release_manifest(output, '2024-03-02')
            errors = []
            _validate_release_manifest(output, pd.Timestamp('2024-03-02'), errors, True)
            self.assertEqual(errors, [])
            (output / CANDLE_FILES[-1]).unlink()
            write_release_manifest(output, '2024-03-02')
            _validate_release_manifest(output, pd.Timestamp('2024-03-02'), errors, True)
            self.assertTrue(any('inventory' in error for error in errors))

    def test_leading_zero_era_and_initial_incomplete_period(self):
        daily, master = self.fixture()
        daily.loc[:'2024-01-02'] = 0
        result = build_candle_tables(daily,master,'2024-03-02')[CANDLE_FILES[0]]
        self.assertEqual(result.loc[result.interval=='daily'].period_start.min(),pd.Timestamp('2024-01-03'))
        self.assertEqual(result.loc[result.interval=='weekly'].period_start.min(),pd.Timestamp('2024-01-08'))
        self.assertEqual(result.loc[result.interval=='monthly'].period_start.min(),pd.Timestamp('2024-02-01'))

if __name__ == '__main__': unittest.main()
