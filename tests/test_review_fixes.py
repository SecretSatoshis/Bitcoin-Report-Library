"""Regressions for the September repository review."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
import pandas as pd
import data_format as ingest
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

    def test_release_rejects_mutated_coefficients_and_candle(self):
        mapping = {
            'power_law_exponent':'Power Law Exponent', 'power_law_scale':'Power Law Scale',
            'metcalfe_scale_any_balance':'Metcalfe Scale (Any Balance)',
            'metcalfe_scale_0p001_btc':'Metcalfe Scale (0.001+ BTC)',
            'metcalfe_scale_0p01_btc':'Metcalfe Scale (0.01+ BTC)',
            'metcalfe_scale_0p1_btc':'Metcalfe Scale (0.1+ BTC)'}
        summary = {f'{prefix} {column}':[value] for prefix in ('Daily','Week-to-Date')
                   for column,value in zip(('Open','High','Low','Close'), (100,110,90,105))}
        summary.update({'Week Start':['2026-09-07'], 'Week-to-Date Days':[2]})
        frames = {'model_coefficients.csv':pd.DataFrame({'coefficient':list(mapping),'value':1.0}),
                  'network_model_metrics.csv':pd.DataFrame({v:[1.0] for v in mapping.values()}),
                  'report_ohlc_summary.csv':pd.DataFrame(summary)}
        frames['model_coefficients.csv']['value'] = 999
        frames['report_ohlc_summary.csv']['Daily High'] = 1
        errors=[]
        release._validate_review_contracts(frames, Path('/nonexistent'), pd.Timestamp('2026-09-08'), errors)
        self.assertTrue(any('model_coefficients.csv' in error for error in errors))
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
