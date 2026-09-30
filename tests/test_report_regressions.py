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

    def test_price_moving_averages_use_complete_calendar_windows(self):
        dates = pd.date_range("2024-01-01", periods=1_500, freq="D")
        prices = pd.Series(np.arange(1.0, 1_501.0), index=dates)
        frame = pd.DataFrame({"BTC Price": prices})
        # A missing close leaves every window that spans it empty, as on the chart.
        gap = dates[1_450]
        frame = frame.drop(gap)

        result = report_tables.add_price_moving_averages(frame)

        self.assertTrue(np.isnan(result.loc[dates[48], "50-day MA"]))
        self.assertEqual(result.loc[dates[49], "50-day MA"], prices.iloc[:50].mean())
        self.assertTrue(np.isnan(result.loc[dates[198], "200-day MA"]))
        self.assertEqual(result.loc[dates[199], "200-day MA"], prices.iloc[:200].mean())
        self.assertTrue(np.isnan(result.loc[dates[88], "3-month MA"]))
        self.assertEqual(result.loc[dates[89], "3-month MA"], prices.iloc[:90].mean())
        self.assertTrue(np.isnan(result.loc[dates[362], "1-year MA"]))
        self.assertEqual(result.loc[dates[363], "1-year MA"], prices.iloc[:364].mean())
        self.assertTrue(np.isnan(result.loc[dates[1_398], "200-week MA"]))
        self.assertEqual(
            result.loc[dates[1_399], "200-week MA"], prices.iloc[:1_400].mean()
        )
        self.assertTrue(np.isnan(result.loc[dates[1_451], "3-month MA"]))
        self.assertTrue(np.isnan(result.loc[dates[-1], "1-year MA"]))
        self.assertNotIn("3-month MA", frame.columns)

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

    @staticmethod
    def _summary_input(nupl, multiple=0.9, supply_in_profit=14_000_000.0):
        dates = pd.date_range(end="2024-01-07", periods=len(nupl), freq="D")
        return pd.DataFrame(
            {
                "price_close": 50_000.0,
                "market_cap": 1_000_000.0,
                "supply": 20_000_000.0,
                "coinbase_sum_24h_usd": 10.0,
                "30_day_ma_coinbase_sum_24h_usd": 99.0,
                "transfer_volume_sum_24h_usd": 20.0,
                "30_day_ma_transfer_volume_sum_24h_usd": 88.0,
                "supply_in_profit": supply_in_profit,
                "nupl": nupl,
                "power_law_price_multiple": multiple,
            },
            index=dates,
        )

    def test_summary_snapshot_uses_raw_daily_onchain_values(self):
        data = self._summary_input([0.3] * 7)
        result = report_tables.create_summary_table(data, "2024-01-07").set_index("Metric")

        self.assertEqual(result.loc["Bitcoin Miner Revenue", "Value"], 10.0)
        self.assertEqual(result.loc["Bitcoin Transaction Volume", "Value"], 20.0)
        self.assertNotIn("Bitcoin Dominance", result.index)

    def test_investor_sentiment_comes_from_onchain_data(self):
        # Six days in Belief and one in Hope still average into Belief / Denial,
        # so a single day's move does not flip the label.
        data = self._summary_input([0.6] * 6 + [0.2], multiple=0.594)
        result = report_tables.create_summary_table(data, "2024-01-07").set_index("Metric")
        sentiment = result[result["Category"] == "Investor Sentiment"]["Value"]

        self.assertEqual(list(sentiment.index), [
            "Bitcoin Supply in Profit", "Bitcoin Market Sentiment", "Bitcoin Valuation",
        ])
        self.assertEqual(sentiment["Bitcoin Supply in Profit"], 70.0)
        self.assertEqual(sentiment["Bitcoin Market Sentiment"], "Belief / Denial")
        self.assertEqual(sentiment["Bitcoin Valuation"], "Below Fair Value")

    def test_sentiment_and_valuation_band_edges(self):
        cases = [(-0.1, "Capitulation"), (0.0, "Hope / Fear"), (0.36, "Optimism / Anxiety"),
                 (0.5, "Belief / Denial"), (0.8, "Euphoria / Greed")]
        for nupl, label in cases:
            with self.subTest(nupl=nupl):
                data = self._summary_input([nupl] * 7)
                self.assertEqual(report_tables._nupl_sentiment(data, "2024-01-07"), label)
        for multiple, label in [(0.45, "Undervalued"), (0.58, "Below Fair Value"),
                                (1.0, "Above Fair Value"), (1.73, "Overvalued"),
                                (3.0, "Extremely Overvalued")]:
            with self.subTest(multiple=multiple):
                self.assertEqual(report_tables._power_law_valuation(multiple), label)

    def test_sentiment_requires_a_full_nupl_week(self):
        with self.assertRaisesRegex(RuntimeError, "NUPL needs 7"):
            report_tables.create_summary_table(self._summary_input([0.3] * 6), "2024-01-07")

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

    def test_only_requested_columns_get_yoy_changes(self):
        dates = pd.date_range("2023-01-01", "2024-03-01", freq="D")
        data = pd.DataFrame(
            {"price_close": np.arange(1.0, len(dates) + 1), "SPY_close": 50.0},
            index=dates,
        )

        result = data_format.calculate_all_changes(data, ["price_close"])

        self.assertEqual(sorted(result.columns), sorted([
            "price_close_7_change", "price_close_90_change", "price_close_MTD_change",
            "price_close_YTD_change", "price_close_YOY_change", "SPY_close_7_change",
            "SPY_close_90_change", "SPY_close_MTD_change", "SPY_close_YTD_change",
        ]))

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
