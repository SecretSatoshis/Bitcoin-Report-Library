"""Regression tests for published report calculations."""

import unittest

import numpy as np
import pandas as pd

import data_format
import report_tables


class ReportRegressionTests(unittest.TestCase):
    @staticmethod
    def _energy_input(subsidy=400.0, fees=4.0):
        return pd.DataFrame(
            {
                "hash_rate": [1.0e18],
                "cm_efficiency_j_gh": [0.03],
                "subsidy_sum_24h": [subsidy],
                "fees_sum_24h": [fees],
                "difficulty": [1.0e14],
                "inflation_rate": [1.0],
                "price_close": [100_000.0],
            },
            index=[pd.Timestamp("2024-01-01")],
        )

    def test_summary_table_requires_a_dominance_capture(self):
        report_data = pd.DataFrame(
            {
                "price_close": [50_000.0],
                "market_cap": [1.0e12],
                "supply": [20_000_000.0],
                "coinbase_sum_24h_usd": [25_000_000.0],
                "transfer_volume_sum_24h_usd": [10_000_000_000.0],
                "fear_greed_value": [55.0],
                "mvrv_ratio": [1.5],
            },
            index=[pd.Timestamp("2024-01-09")],
        )

        with self.assertRaisesRegex(RuntimeError, "Bitcoin dominance is required"):
            report_tables.create_summary_table(report_data, "2024-01-09")

    def test_electricity_cost_uses_observed_subsidy_plus_fees_and_tariffs(self):
        result = data_format.electric_price_models(self._energy_input()).iloc[0]

        expected_power_watts = 1.0e18 / 1.0e9 * 0.03
        expected_kwh = expected_power_watts * 24 / 1000
        expected_revenue = 404.0
        self.assertEqual(result["network_power_watts"], expected_power_watts)
        self.assertEqual(
            result["daily_electricity_consumption_kwh"], expected_kwh
        )
        self.assertEqual(result["miner_revenue_btc"], expected_revenue)

        for cents in range(3, 8):
            expected = expected_kwh * (cents / 100) / expected_revenue
            self.assertAlmostEqual(result[f"Electricity_Cost_{cents}c"], expected)
        self.assertEqual(result["Electricity_Cost"], result["Electricity_Cost_5c"])

        legacy = expected_kwh * 0.05 * 1.1 / 400.0
        self.assertAlmostEqual(
            result["Electricity_Cost_PUE_Subsidy_Only"], legacy
        )
        self.assertAlmostEqual(result["Bitcoin_Production_Cost"], legacy / 0.6)
        self.assertAlmostEqual(
            result["power_only_breakeven_tariff_usd_per_kwh"],
            100_000.0 * expected_revenue / expected_kwh,
        )

    def test_electricity_cost_returns_nan_for_zero_miner_revenue(self):
        result = data_format.electric_price_models(
            self._energy_input(subsidy=0.0, fees=0.0)
        ).iloc[0]

        self.assertTrue(pd.isna(result["Electricity_Cost"]))
        self.assertTrue(pd.isna(result["Electricity_Cost_PUE_Subsidy_Only"]))
        numeric = pd.to_numeric(result, errors="coerce").dropna()
        self.assertFalse(np.isinf(numeric).any())

    def test_electricity_scenario_table_preserves_all_model_definitions(self):
        modeled = data_format.electric_price_models(self._energy_input())
        result = report_tables.create_electricity_cost_scenarios(
            modeled, report_date="2024-01-01"
        )

        self.assertEqual(result.index.name, "date")
        self.assertIn("Power Expense ($0.03/kWh)", result.columns)
        self.assertIn("Power Expense ($0.07/kWh)", result.columns)
        self.assertIn("Legacy PUE/Subsidy-Only Cost", result.columns)
        self.assertIn("Bitcoin Production Cost", result.columns)
        self.assertIn("Hayes Network Price", result.columns)
        self.assertIn("Energy Value", result.columns)
        self.assertAlmostEqual(
            result.iloc[0]["Power Expense ($0.05/kWh)"],
            modeled.iloc[0]["Electricity_Cost"],
        )

    def test_price_buckets_exclude_placeholders_and_post_cutoff_rows(self):
        prices = pd.DataFrame(
            {"price_close": [0.0, 999.0, 1_000.0, 9_999.0]},
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )

        result = report_tables.calculate_price_buckets(
            prices, 1_000, report_date="2024-01-03"
        )

        self.assertEqual(int(result["Count"].sum()), 2)
        self.assertEqual(float(result["Current Price"].iloc[0]), 1_000.0)
        self.assertEqual(int(result.loc[0, "Count"]), 1)
        self.assertEqual(int(result.loc[1, "Count"]), 1)

    def test_roi_year_periods_use_calendar_offsets(self):
        dates = pd.date_range("2015-01-01", "2026-08-15", freq="D")
        prices = pd.DataFrame(
            {"price_close": np.arange(1, len(dates) + 1, dtype=float)}, index=dates
        )

        result = report_tables.calculate_roi_table(
            prices, report_date="2026-08-14"
        ).set_index("Time Frame")

        expected = {
            "1 Year": "2025-08-14",
            "2 Year": "2024-08-14",
            "4 Year": "2022-08-14",
            "5 Year": "2021-08-14",
            "10 Year": "2016-08-14",
        }
        for period, start_date in expected.items():
            self.assertEqual(result.loc[period, "Start Date"], pd.Timestamp(start_date))

    def test_indexed_ytd_uses_common_calendar_ordinal_and_asof_cap(self):
        dates = pd.date_range("2019-01-01", "2021-12-31", freq="D")

        def common_ordinal(timestamp):
            if timestamp.month == 2 and timestamp.day == 29:
                return 999
            ordinal = timestamp.dayofyear
            if timestamp.is_leap_year and timestamp.month > 2:
                ordinal -= 1
            return ordinal

        prices = pd.Series(
            [100.0 + common_ordinal(timestamp) for timestamp in dates],
            index=dates,
        )

        result = report_tables.create_indexed_returns_history(
            prices, report_date="2021-03-01", period="ytd", min_year=2019
        )

        self.assertEqual(result.index.name, "day_of_year")
        self.assertEqual(result["2021"].last_valid_index(), 60)
        self.assertAlmostEqual(result.loc[60, "2020"], result.loc[60, "2021"])
        # Common-calendar days plus the explicit prior-year-close anchor at row 0.
        self.assertEqual(result["2020"].notna().sum(), 366)

    def test_summary_history_has_both_30_day_endpoints(self):
        dates = pd.date_range("2024-02-20", "2024-04-01", freq="D")
        data = pd.DataFrame({"price_close": np.arange(len(dates))}, index=dates)

        result = report_tables.create_summary_history(
            data,
            report_date="2024-03-31",
            metrics={"Bitcoin Price USD": "price_close"},
        )

        self.assertEqual(len(result), 31)
        self.assertEqual(result["date"].iloc[0], "2024-03-01")
        self.assertEqual(result["date"].iloc[-1], "2024-03-31")

    def test_summary_snapshot_uses_raw_daily_onchain_values(self):
        data = pd.DataFrame(
            {
                "price_close": [50_000.0],
                "market_cap": [1_000_000.0],
                "supply": [20_000_000.0],
                "coinbase_sum_24h_usd": [10.0],
                "30_day_ma_coinbase_sum_24h_usd": [99.0],
                "transfer_volume_sum_24h_usd": [20.0],
                "30_day_ma_transfer_volume_sum_24h_usd": [88.0],
                "bitcoin_dominance": [55.0],
                "fear_greed_value": [50.0],
                "mvrv_ratio": [1.5],
            },
            index=[pd.Timestamp("2024-01-01")],
        )

        result = report_tables.create_summary_table(data, "2024-01-01").set_index(
            "Metric"
        )

        self.assertEqual(result.loc["Bitcoin Miner Revenue", "Value"], 10.0)
        self.assertEqual(result.loc["Bitcoin Transaction Volume", "Value"], 20.0)

    def test_yoy_change_matches_the_same_calendar_date(self):
        dates = pd.to_datetime(
            [
                "2019-02-28",
                "2020-02-28",
                "2020-02-29",
                "2020-03-01",
                "2021-02-28",
                "2021-03-01",
            ]
        )
        data = pd.DataFrame(
            {"metric": [100.0, 200.0, 300.0, 400.0, 600.0, 800.0]},
            index=dates,
        )

        result = data_format.calculate_yoy_change(data)

        self.assertEqual(result.loc["2020-02-28", "metric_YOY_change"], 100.0)
        self.assertEqual(result.loc["2020-02-29", "metric_YOY_change"], 200.0)
        self.assertEqual(result.loc["2021-02-28", "metric_YOY_change"], 200.0)
        self.assertEqual(result.loc["2021-03-01", "metric_YOY_change"], 100.0)

    def test_cycle_series_starts_at_actual_low_and_never_falls_below_one(self):
        dates = pd.date_range("2010-07-25", "2011-11-17", freq="D")
        prices = pd.Series(10.0, index=dates)
        prices.loc["2010-07-25"] = 8.0
        prices.loc["2010-07-27"] = 5.0
        prices.loc["2010-07-28":] = 6.0
        data = pd.DataFrame({"price_close": prices})

        result = data_format.compute_cycle_lows(data)

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

        result = data_format.compute_halving_days(data)

        self.assertNotIn("Genesis Era", set(result["Era"]))
        second_era = result[result["Era"] == "2nd Era"]
        self.assertFalse(second_era.empty)
        self.assertEqual(second_era["days_since_halving"].iloc[0], 0)
        self.assertEqual(second_era["index_value"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
