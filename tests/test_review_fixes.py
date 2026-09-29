"""Regressions for the September repository review."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
import pandas as pd
import json
import data_format as ingest
from candle_data import weekly_ohlc
import report_tables as tables
import validate_outputs as release
from data_validation import validate_calendar
from tests.test_ingestion_reliability import FakeResponse


class ReviewFixTests(unittest.TestCase):
    def test_missing_duplicate_and_unordered_calendar_rejected(self):
        dates = pd.date_range('2024-01-01', periods=10)
        for broken in (dates.delete(4), dates.insert(4, dates[4]), dates[::-1]):
            with self.subTest(index=broken), self.assertRaises(RuntimeError):
                ingest.assert_no_internal_onchain_gaps(pd.DataFrame(
                    {'coinbase_sum_24h_usd': 1.0}, index=broken), dates[-1])
        validate_calendar(dates, 'complete')

    def test_ohlc_metadata_and_candle_contract(self):
        base = dict(index='day1', start=10, end=12)
        dates = FakeResponse(json_data=dict(base, data=['2024-01-01', '2024-01-02']))
        good = FakeResponse(json_data=dict(base, data=[[100,110,90,105]]*2))
        with patch.object(ingest.requests, 'get', side_effect=[dates, good]):
            self.assertEqual(len(ingest.get_brk_ohlc('day1')), 2)
        for payload in (dict(base, index='hour1', data=[[100,110,90,105]]*2),
                        dict(base, start=20, end=22, data=[[100,110,90,105]]*2),
                        dict(base, data=[[100,90,110,100]]*2)):
            with patch.object(ingest.requests, 'get', side_effect=[dates, FakeResponse(json_data=payload)]):
                with self.assertRaises(RuntimeError):
                    ingest.get_brk_ohlc('day1')

    def test_missing_week_candle_cannot_replace_output(self):
        candles = pd.DataFrame([[100,110,90,105]]*3, columns=ingest.OHLC_COLUMNS,
                               index=pd.to_datetime(['2024-01-01','2024-01-03','2024-01-04']))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'summary.csv'
            target.write_text('previous')
            with self.assertRaisesRegex(ValueError, 'missing a day'):
                tables.create_report_ohlc_summary(candles, '2024-01-04', target)
            self.assertEqual(target.read_text(), 'previous')

    def test_fundamentals_exact_dates_and_calendar_range(self):
        dates = pd.date_range('2023-01-01', periods=400)
        frame = pd.DataFrame({'metric': 10.0}, index=dates)
        template = {'Section': {'Metric': ('metric','number')}}
        frame.iloc[0] = 9999
        frame.loc[dates[-1]-pd.Timedelta(days=7), 'metric'] = np.nan
        result = tables.create_fundamentals_table(frame, template, dates[-1]).iloc[0]
        self.assertTrue(pd.isna(result['7 Day Change (%)']))
        self.assertEqual(result['52W High'], '10')
        frame.iloc[-1] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing on'):
            tables.create_fundamentals_table(frame, template, dates[-1])

    def test_optional_crypto_rate_limits_have_bounded_wait(self):
        with patch.object(ingest.requests, 'get', return_value=FakeResponse(status_code=429)) as get, patch.object(ingest.time, 'sleep') as sleep:
            self.assertTrue(ingest.get_crypto_data(['ethereum']).empty)
        self.assertEqual(get.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5,10,1])

    def test_all_optional_price_sources_missing_preserves_declared_columns(self):
        base = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=10),
                             'price_close': 100.0})
        with ExitStack() as stack:
            stack.enter_context(patch.object(ingest, 'get_brk_onchain', return_value=base))
            for function in ('get_price', 'get_marketcap', 'get_fear_and_greed_index',
                             'get_miner_data', 'get_bitcoin_dominance',
                             'get_btc_trade_volume_14d', 'get_crypto_data'):
                stack.enter_context(patch.object(ingest, function, return_value=pd.DataFrame()))
            data = ingest.get_data({'stocks':['MISSING'], 'crypto':['ethereum']}, '2024-01-01')
        self.assertTrue(data[['MISSING_close', 'MISSING_MarketCap', 'ethereum_close',
                              'ethereum_market_cap', 'ethereum_volume']].isna().all().all())
        filled = ingest.forward_fill_market_data(data)
        self.assertTrue(filled['MISSING_close'].isna().all())

    def test_absent_asset_keeps_correlation_schema(self):
        frame = pd.DataFrame({'price_close': np.arange(1,41)**2}, index=pd.date_range('2024-01-01', periods=40))
        result = ingest.create_btc_correlation_data(frame.index[-1], {'stocks':['MISSING']}, frame)
        for item in result.values():
            self.assertTrue(pd.isna(item.loc['price_close','MISSING_close']))

    def test_release_rejects_an_inconsistent_candle(self):
        summary = {f'{prefix} {column}':[value] for prefix in ('Daily','Week-to-Date')
                   for column,value in zip(('Open','High','Low','Close'), (100,110,90,105))}
        summary.update({'Week Start':['2026-09-07'], 'Week-to-Date Days':[2]})
        frames = {'report_ohlc_summary.csv':pd.DataFrame(summary)}
        frames['report_ohlc_summary.csv']['Daily High'] = 1
        errors=[]
        release._validate_review_contracts(frames, Path('/nonexistent'), pd.Timestamp('2026-09-08'), errors)
        self.assertTrue(any('report_ohlc_summary.csv' in error for error in errors))

    def test_completed_december_year_included_in_average(self):
        frame = pd.DataFrame({'price_close':[100,200,800]}, index=pd.to_datetime(['2022-12-31','2023-12-31','2024-12-31']))
        result = tables.monthly_heatmap(frame, export_csv=False)
        self.assertAlmostEqual(float(result.loc['Average','Yearly']), 200.0)

    def test_release_rejects_missing_day_and_mutated_fundamental(self):
        from data_definitions import metrics_template
        columns = {item[0] for group in metrics_template.values() for item in group.values()}
        dates = pd.date_range('2024-01-01', periods=400)
        master = pd.DataFrame(10.0, index=dates, columns=sorted(columns))
        master.index.name = 'time'
        fundamentals = tables.create_fundamentals_table(master, metrics_template, dates[-1])
        fundamentals.loc[0,'Current Value'] = '999999999'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            master.drop(index=dates[10]).to_csv(output/'master_metrics_data.csv.gz')
            errors=[]
            release._validate_index_cutoff(output, 'master_metrics_data.csv.gz', 'time', dates[-1], errors)
            release._validate_review_contracts({'fundamentals_table.csv': fundamentals}, output, dates[-1], errors)
        self.assertTrue(any('complete' in error for error in errors))
        self.assertTrue(any('fundamentals_table.csv' in error for error in errors))


class SecondReviewFixTests(unittest.TestCase):
    def test_cagr_uses_calendar_years_across_leap_days(self):
        dates = pd.date_range('2020-01-01', '2024-03-01')
        values = pd.DataFrame({'price_close': np.arange(1.0, len(dates) + 1)}, index=dates)
        cagr = ingest.calculate_rolling_cagr_for_all_columns(values, 4)
        start = values.loc['2020-03-01', 'price_close']
        end = values.loc['2024-03-01', 'price_close']
        expected = ((end / start) ** 0.25 - 1) * 100
        self.assertAlmostEqual(cagr.loc['2024-03-01', 'price_close_4_Year_CAGR'], expected)

    def test_weekly_ohlc_is_cut_off_at_the_report_date(self):
        dates = pd.date_range('2024-01-01', '2024-01-17')  # Monday start
        close = np.arange(100.0, 100.0 + len(dates))
        daily = pd.DataFrame({'Open': close, 'High': close + 5, 'Low': close - 5,
                              'Close': close}, index=dates)
        weekly = weekly_ohlc(daily, '2024-01-16', start='2024-01-03')
        self.assertEqual(weekly.index.name, 'Time')
        self.assertEqual(list(weekly.index), list(pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15'])))
        # The open week closes on the report date, not on the later partial day.
        self.assertEqual(weekly['Close'].iloc[-1], daily.loc['2024-01-16', 'Close'])
        with self.assertRaisesRegex(ValueError, 'report date'):
            weekly_ohlc(daily, '2024-01-20')

    def test_validator_reads_the_report_date_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'release_manifest.json'
            path.write_text(json.dumps({'report_date': '2026-09-26'}))
            # Generation crossed UTC midnight: the clock now says 09-27.
            self.assertEqual(release._manifest_report_date(directory, '2026-09-27'), ('2026-09-26', None))
            date, error = release._manifest_report_date(directory, '2026-09-29')
            self.assertIsNone(date)
            self.assertIn('not a current release', error)
            path.unlink()
            self.assertIsNone(release._manifest_report_date(directory, '2026-09-27')[0])

