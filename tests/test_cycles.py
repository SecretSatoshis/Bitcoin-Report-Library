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



class DrawdownTests(unittest.TestCase):
    def test_each_cycle_runs_from_its_high_and_the_open_cycle_to_the_latest_day(self):
        dates = pd.date_range("2024-01-01", "2024-01-20")
        prices = pd.Series(100.0, index=dates)
        prices.loc["2024-01-03"] = 80.0
        prices.loc["2024-01-15":] = [150.0, 120.0, 90.0, 160.0, 150.0, 140.0]
        cycles_config = [("Cycle 1", "2024-01-01", "2024-01-05"), ("Cycle 2", "2024-01-15", None)]
        with patch.object(cycles, "BITCOIN_DRAWDOWN_CYCLES", cycles_config):
            result = cycles.compute_drawdowns(pd.DataFrame({"price_close": prices, "other": 1.0}))

        first = result[result["Cycle"] == "Cycle 1"]
        self.assertEqual(first["days_since_ath"].tolist(), [0, 1, 2, 3, 4])
        self.assertTrue(np.allclose(first["drawdown_pct"], [0.0, 0.0, -20.0, 0.0, 0.0]))
        second = result[result["Cycle"] == "Cycle 2"]
        self.assertEqual(second["days_since_ath"].iloc[-1], 5)
        self.assertAlmostEqual(second["drawdown_pct"].iloc[3], 0.0)  # new high on day 3
        self.assertAlmostEqual(second["drawdown_pct"].iloc[-1], (140 / 160 - 1) * 100)
        self.assertEqual(list(result.columns), ["days_since_ath", "drawdown_pct", "Cycle"])


if __name__ == "__main__":
    unittest.main()
