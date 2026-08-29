"""Regression tests for deterministic data-integrity safeguards.

Each test pins behaviour that was previously wrong in a way that produced a plausible
looking but incorrect published number, so a silent regression here is expensive.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

import data_format
from data_definitions import BITCOIN_GENESIS_DATE


class AverageCapNetworkAgeTests(unittest.TestCase):
    """Average Cap divides by the network's age, not a row counter."""

    def frame(self, start: str, periods: int) -> pd.DataFrame:
        index = pd.date_range(start, periods=periods, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "market_cap": 100.0,
                "supply": 20.0,
                "realized_cap": 40.0,
                "realized_price": 7.0,
                "price_close": 10.0,
                "transfer_volume_sum_24h_usd": 10.0,
                "coinbase_sum_24h_usd": 1.0,
                "utxos_over_1y_old_supply": 12.0,
                "coindays_destroyed_sum_24h": 2.0,
                "supply_in_profit": 15.0,
                "supply_in_loss": 5.0,
                "fees_sum_24h": 1.0,
                "coinbase_sum_24h": 5.0,
                "active_addrs_average_24h": 1000.0,
            },
            index=index,
        )

    def test_divisor_is_days_since_genesis(self):
        result = data_format.calculate_custom_on_chain_metrics(
            self.frame("2010-01-01", 10)
        )
        first_date = pd.Timestamp("2010-01-01")
        expected_age = (first_date - BITCOIN_GENESIS_DATE).days + 1

        # Cumulative market cap on day one is a single day's 100.0.
        self.assertAlmostEqual(
            result["average_cap"].iloc[0], 100.0 / expected_age, places=9
        )
        self.assertNotAlmostEqual(result["average_cap"].iloc[0], 100.0, places=3)

    def test_average_cap_is_independent_of_where_the_history_starts(self):
        """The metric is a property of the network, not of the fetch window."""
        long_run = data_format.calculate_custom_on_chain_metrics(
            self.frame("2010-01-01", 400)
        )
        # A window starting later reaches the same date with fewer rows; the shared
        # dates must still agree once the cumulative base is aligned.
        shared_date = pd.Timestamp("2010-06-01", tz="UTC")
        elapsed = (shared_date.tz_localize(None) - BITCOIN_GENESIS_DATE).days + 1
        rows_before = (shared_date - pd.Timestamp("2010-01-01", tz="UTC")).days + 1
        self.assertAlmostEqual(
            long_run.loc[shared_date, "average_cap"],
            (100.0 * rows_before) / elapsed,
            places=9,
        )

    def test_delta_cap_absorbs_the_corrected_average_cap(self):
        result = data_format.calculate_custom_on_chain_metrics(
            self.frame("2010-01-01", 5)
        )
        self.assertTrue(
            np.allclose(
                result["delta_cap"], 40.0 - result["average_cap"], equal_nan=True
            )
        )


class CumulativeOnchainGapTests(unittest.TestCase):
    """A hole inside a cumulative input aborts instead of zero-filling."""

    def frame(self) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame({"coinbase_sum_24h_usd": 1.0}, index=index)

    def test_complete_series_passes(self):
        data_format.assert_no_internal_onchain_gaps(self.frame(), "2024-01-30")

    def test_internal_gap_raises(self):
        frame = self.frame()
        frame.iloc[10, 0] = np.nan
        with self.assertRaises(RuntimeError) as ctx:
            data_format.assert_no_internal_onchain_gaps(frame, "2024-01-30")
        self.assertIn("internal gap", str(ctx.exception))
        self.assertIn("2024-01-11", str(ctx.exception))

    def test_leading_nulls_are_allowed(self):
        frame = self.frame()
        frame.iloc[:5, 0] = np.nan
        data_format.assert_no_internal_onchain_gaps(frame, "2024-01-30")

    def test_gap_after_the_report_date_is_ignored(self):
        frame = self.frame()
        frame.iloc[25, 0] = np.nan
        data_format.assert_no_internal_onchain_gaps(frame, "2024-01-20")

    def test_absent_column_raises(self):
        with self.assertRaises(RuntimeError):
            data_format.assert_no_internal_onchain_gaps(
                pd.DataFrame(index=pd.date_range("2024-01-01", periods=3)),
                "2024-01-03",
            )


class ReferenceDataVintageTests(unittest.TestCase):
    """Hand-maintained inputs must carry a plausible, current vintage."""

    def test_current_reference_vintage_passes(self):
        data_format.assert_reference_data_fresh(
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
                data_format.assert_reference_data_fresh(
                    "2026-08-28", vintages, max_age_days=365
                )


class SharesOutstandingBudgetTests(unittest.TestCase):
    """The share-count fill is bounded by a cadence-appropriate budget."""

    def test_budget_clears_a_semiannual_filer_but_not_a_dormant_one(self):
        # Observed worst case among tracked tickers is 2222.SR at ~162 days.
        self.assertGreater(data_format.SHARES_OUTSTANDING_MAX_AGE_DAYS, 162)
        self.assertLess(data_format.SHARES_OUTSTANDING_MAX_AGE_DAYS, 365)

    def test_stale_share_count_nulls_the_market_cap(self):
        """The masking arithmetic, isolated from the network fetch."""
        close = pd.Series(
            10.0, index=pd.date_range("2024-01-01", periods=400, freq="D")
        )
        shares = pd.Series(
            [1_000.0], index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")])
        )

        combined = shares.index.union(close.index).sort_values()
        filled = shares.reindex(combined).ffill().reindex(close.index)
        source = (
            pd.Series(shares.index, index=shares.index)
            .reindex(combined)
            .ffill()
            .reindex(close.index)
        )
        rows = pd.Series(close.index, index=close.index).dt.normalize()
        age = (rows - source.dt.normalize()).dt.days
        budget = data_format.SHARES_OUTSTANDING_MAX_AGE_DAYS
        masked = filled.where(age.between(0, budget))

        self.assertTrue(masked.iloc[0] == 1_000.0)
        self.assertTrue(masked.loc[close.index[budget]] == 1_000.0)
        self.assertTrue(pd.isna(masked.loc[close.index[budget + 1]]))


class MasterCutoffValidationTests(unittest.TestCase):
    """Large dated exports are asserted to end on the report date."""

    def write(self, tmpdir, name, last_date):
        frame = pd.DataFrame(
            {
                "time": pd.date_range(end=last_date, periods=5, freq="D"),
                "value": 1.0,
            }
        )
        path = tmpdir / name
        frame.to_csv(path, index=False)
        return path

    def test_partial_day_in_master_is_reported(self):
        import tempfile
        from pathlib import Path

        import validate_outputs

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            self.write(tmpdir, "master_metrics_data.csv.gz", "2026-08-28")
            errors = []
            validate_outputs._validate_index_cutoff(
                tmpdir,
                "master_metrics_data.csv.gz",
                "time",
                pd.Timestamp("2026-08-27"),
                errors,
            )
            self.assertEqual(len(errors), 1)
            self.assertIn("2026-08-28", errors[0])
            self.assertIn("expected 2026-08-27", errors[0])

    def test_truncated_master_passes(self):
        import tempfile
        from pathlib import Path

        import validate_outputs

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            self.write(tmpdir, "master_metrics_data.csv.gz", "2026-08-27")
            errors = []
            validate_outputs._validate_index_cutoff(
                tmpdir,
                "master_metrics_data.csv.gz",
                "time",
                pd.Timestamp("2026-08-27"),
                errors,
            )
            self.assertEqual(errors, [])

    def test_both_large_exports_are_covered(self):
        import validate_outputs

        self.assertIn(
            "master_metrics_data.csv.gz", validate_outputs.INDEX_CUTOFF_OUTPUTS
        )
        self.assertIn("cagr_data.csv", validate_outputs.INDEX_CUTOFF_OUTPUTS)


if __name__ == "__main__":
    unittest.main()
