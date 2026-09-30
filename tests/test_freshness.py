"""Publish gates: bounded fills, staleness, gaps and review dates (freshness.py)."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd

import freshness
import sources
from data_validation import validate_calendar


class CumulativeOnchainGapTests(unittest.TestCase):
    """A hole inside a cumulative input aborts instead of zero-filling."""

    def frame(self) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {"coinbase_sum_24h_usd": 1.0, "supply": 19_000_000.0}, index=index
        )

    def test_complete_series_passes(self):
        freshness.assert_no_internal_onchain_gaps(self.frame(), "2024-01-30")

    def test_internal_gap_raises(self):
        frame = self.frame()
        frame.iloc[10, 0] = np.nan
        with self.assertRaises(RuntimeError) as ctx:
            freshness.assert_no_internal_onchain_gaps(frame, "2024-01-30")
        self.assertIn("internal gap", str(ctx.exception))
        self.assertIn("2024-01-11", str(ctx.exception))

    def test_internal_supply_gap_raises(self):
        frame = self.frame()
        frame.iloc[12, 1] = np.nan
        with self.assertRaisesRegex(RuntimeError, "supply has an internal gap"):
            freshness.assert_no_internal_onchain_gaps(frame, "2024-01-30")

    def test_leading_nulls_are_allowed(self):
        frame = self.frame()
        frame.iloc[:5, 0] = np.nan
        freshness.assert_no_internal_onchain_gaps(frame, "2024-01-30")

    def test_gap_after_the_report_date_is_ignored(self):
        frame = self.frame()
        frame.iloc[25, 0] = np.nan
        freshness.assert_no_internal_onchain_gaps(frame, "2024-01-20")

    def test_absent_column_raises(self):
        with self.assertRaises(RuntimeError):
            freshness.assert_no_internal_onchain_gaps(
                pd.DataFrame(index=pd.date_range("2024-01-01", periods=3)),
                "2024-01-03",
            )


class ReferenceDataVintageTests(unittest.TestCase):
    """Hand-maintained inputs must carry a plausible, current vintage."""

    def test_current_reference_vintage_passes(self):
        freshness.assert_reference_data_fresh(
            "2026-08-28", {"reference": "2026-08-22"}, max_age_days=365
        )

    def test_stale_future_and_invalid_vintages_fail(self):
        cases = (
            {"stale": "2025-01-01"},
            {"future": "2026-08-29"},
            {"invalid": "not-a-date"},
        )
        for vintages in cases:
            with self.subTest(vintages=vintages), self.assertRaises(RuntimeError):
                freshness.assert_reference_data_fresh(
                    "2026-08-28", vintages, max_age_days=365
                )

    def test_power_law_bands_expire_a_year_after_review(self):
        from data_definitions import POWER_LAW_BANDS_AS_OF, REFERENCE_DATA_VINTAGES

        self.assertIs(REFERENCE_DATA_VINTAGES["POWER_LAW_VALUATION_BANDS"], POWER_LAW_BANDS_AS_OF)
        bands = {"POWER_LAW_VALUATION_BANDS": POWER_LAW_BANDS_AS_OF}
        freshness.assert_reference_data_fresh(POWER_LAW_BANDS_AS_OF + pd.Timedelta(days=365), bands)
        with self.assertRaisesRegex(RuntimeError, "POWER_LAW_VALUATION_BANDS"):
            freshness.assert_reference_data_fresh(POWER_LAW_BANDS_AS_OF + pd.Timedelta(days=366), bands)


class FreshnessCheckTests(unittest.TestCase):
    def test_market_freshness_reports_true_source_age(self):
        index = pd.date_range("2024-01-01", periods=7, freq="D")
        marker = sources._source_observation_column("SPY_close")
        data = pd.DataFrame(
            {
                "price_close": np.arange(7, dtype=float),
                "SPY_close": [100.0] * 6 + [np.nan],
                marker: [pd.Timestamp("2024-01-01")] * 6 + [pd.NaT],
                "AAPL_MarketCap": [3_000.0] * 7,
            },
            index=index,
        )

        with self.assertWarnsRegex(
            RuntimeWarning, r"SPY_close \(source 2024-01-01, 6 days old\)"
        ):
            issues = freshness.warn_on_stale_market_data(
                data, "2024-01-07", max_age_days=5
            )
        self.assertTrue(any(issue.startswith("SPY_close") for issue in issues))

    def test_onchain_freshness_requires_the_report_date_row(self):
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {metric: 1.0 for metric in freshness.REQUIRED_ONCHAIN_METRICS},
            index=index,
        )
        freshness.assert_onchain_freshness(data, index[-1])
        with self.assertRaisesRegex(RuntimeError, "stale"):
            freshness.assert_onchain_freshness(data, index[-1] + pd.Timedelta(days=1))


class CalendarAndFillTests(unittest.TestCase):
    def test_missing_duplicate_and_unordered_calendar_rejected(self):
        dates = pd.date_range('2024-01-01', periods=10)
        for broken in (dates.delete(4), dates.insert(4, dates[4]), dates[::-1]):
            with self.subTest(index=broken), self.assertRaises(RuntimeError):
                freshness.assert_no_internal_onchain_gaps(pd.DataFrame(
                    {'coinbase_sum_24h_usd': 1.0}, index=broken), dates[-1])
        validate_calendar(dates, 'complete')

    def test_all_optional_price_sources_missing_preserves_declared_columns(self):
        base = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=10),
                             'price_close': 100.0})
        with ExitStack() as stack:
            stack.enter_context(patch.object(sources, 'get_brk_onchain', return_value=base))
            for function in ('get_price', 'get_marketcap', 'get_miner_data'):
                stack.enter_context(patch.object(sources, function, return_value=pd.DataFrame()))
            data = sources.get_data({'stocks':['MISSING']}, '2024-01-01')
        self.assertTrue(data[['MISSING_close', 'MISSING_MarketCap']].isna().all().all())
        filled = freshness.forward_fill_market_data(data)
        self.assertTrue(filled['MISSING_close'].isna().all())


if __name__ == "__main__":
    unittest.main()
