"""Drawdown, cycle-low and halving series (cycles.py)."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import cycles


class CycleSeriesTests(unittest.TestCase):
    def test_cycle_series_starts_at_actual_low_and_never_falls_below_one(self):
        dates = pd.date_range("2010-07-25", "2011-11-17", freq="D")
        prices = pd.Series(10.0, index=dates)
        prices.loc["2010-07-25"] = 8.0
        prices.loc["2010-07-27"] = 5.0
        prices.loc["2010-07-28":] = 6.0
        data = pd.DataFrame({"price_close": prices})

        result = cycles.compute_cycle_lows(data)

        self.assertEqual(result["days_since_cycle_low"].iloc[0], 0)
        self.assertEqual(result["index_value"].iloc[0], 1.0)
        self.assertGreaterEqual(result["index_value"].min(), 1.0)
        self.assertEqual(len(result), len(dates) - 2)

    def test_halving_series_omits_genesis_without_a_day_zero_price(self):
        dates = pd.date_range("2010-01-01", "2013-01-15", freq="D")
        prices = pd.Series(0.0, index=dates)
        prices.loc["2010-08-01":] = 1.0
        prices.loc["2012-11-28":] = 10.0
        data = pd.DataFrame({"price_close": prices})

        result = cycles.compute_halving_days(data)

        self.assertNotIn("Genesis Era", set(result["Era"]))
        second_era = result[result["Era"] == "2nd Era"]
        self.assertFalse(second_era.empty)
        self.assertEqual(second_era["days_since_halving"].iloc[0], 0)
        self.assertEqual(second_era["index_value"].iloc[0], 1.0)



    def test_cycle_lows_anchor_within_peak_search_days(self):
        dates = pd.date_range("2015-01-01", "2016-01-01", freq="D")
        prices = pd.Series(250.0, index=dates)
        # Actual low is 1 day before listed 2015-01-15 (within ±PEAK_SEARCH_DAYS).
        prices.loc["2015-01-14"] = 175.0
        prices.loc["2015-01-15"] = 200.0
        # A dip months later is outside ±PEAK_SEARCH_DAYS and does not shift day 0.
        prices.loc["2015-08-01"] = 180.0
        with patch.object(cycles, "BITCOIN_CYCLE_LOW_DATES", ["2015-01-15"]):
            result = cycles.compute_cycle_lows(pd.DataFrame({"price_close": prices}))

        self.assertEqual(result["days_since_cycle_low"].iloc[0], 0)
        self.assertEqual(result["index_value"].iloc[0], 1.0)
        self.assertEqual(len(result), (pd.Timestamp("2016-01-01") - pd.Timestamp("2015-01-14")).days + 1)
        self.assertAlmostEqual(result["index_value"].iloc[1], 200.0 / 175.0)


class DrawdownTests(unittest.TestCase):
    def test_open_cycle_caps_when_starting_ath_is_reclaimed_and_otherwise_runs_to_end(self):
        dates = pd.date_range("2024-01-01", "2024-01-20")
        prices = pd.Series(100.0, index=dates)
        prices.loc["2024-01-03"] = 80.0
        prices.loc["2024-01-15":] = [150.0, 120.0, 90.0, 160.0, 150.0, 140.0]
        cycles_config = [("Cycle 1", "2024-01-01", "2024-01-05"), ("Cycle 2", "2024-01-15", None)]
        with patch.object(cycles, "BITCOIN_DRAWDOWN_CYCLES", cycles_config), patch.object(
            cycles, "PEAK_SEARCH_DAYS", 0
        ):
            result = cycles.compute_drawdowns(pd.DataFrame({"price_close": prices, "other": 1.0}))

        first = result[result["Cycle"] == "Cycle 1"]
        self.assertEqual(first["days_since_ath"].tolist(), [0, 1, 2, 3, 4])
        self.assertTrue(np.allclose(first["drawdown_pct"], [0.0, 0.0, -20.0, 0.0, 0.0]))
        second = result[result["Cycle"] == "Cycle 2"]
        # Cycle 2 reclaims 150 on day 3 (160) and stops there instead of starting a new drawdown.
        self.assertEqual(second["days_since_ath"].tolist(), [0, 1, 2, 3])
        self.assertAlmostEqual(second["drawdown_pct"].iloc[-1], 0.0)
        self.assertEqual(list(result.columns), ["days_since_ath", "drawdown_pct", "Cycle"])

        # An open cycle that has not reclaimed its starting ATH still runs to the latest day.
        prices.loc["2024-01-18":] = [130.0, 125.0, 120.0]
        with patch.object(cycles, "BITCOIN_DRAWDOWN_CYCLES", [("Cycle 2", "2024-01-15", None)]), patch.object(
            cycles, "PEAK_SEARCH_DAYS", 0
        ):
            open_result = cycles.compute_drawdowns(pd.DataFrame({"price_close": prices}))
        self.assertEqual(open_result["days_since_ath"].iloc[-1], 5)
        self.assertAlmostEqual(open_result["drawdown_pct"].iloc[-1], (120 / 150 - 1) * 100)

    def test_a_listed_ath_date_a_day_off_still_starts_at_the_highest_close(self):
        # The 2021 case: the highest close was Nov 9, listed as Nov 10.
        dates = pd.date_range("2021-11-01", "2021-11-30")
        prices = pd.Series(60_000.0, index=dates)
        prices.loc["2021-11-09"] = 67_330.0
        prices.loc["2021-11-10"] = 64_627.0
        prices.loc["2021-11-20"] = 50_496.0
        with patch.object(cycles, "BITCOIN_DRAWDOWN_CYCLES", [("Cycle 4", "2021-11-10", None)]):
            result = cycles.compute_drawdowns(pd.DataFrame({"price_close": prices}))

        self.assertEqual(result["days_since_ath"].iloc[0], 0)
        self.assertAlmostEqual(result["drawdown_pct"].iloc[0], 0.0)
        self.assertAlmostEqual(result["drawdown_pct"].min(), (50_496 / 67_330 - 1) * 100)
        self.assertEqual(len(result), 22)  # Nov 9 through Nov 30


if __name__ == "__main__":
    unittest.main()
