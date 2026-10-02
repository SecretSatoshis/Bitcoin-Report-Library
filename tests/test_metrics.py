"""Calculated metrics: on-chain models, changes, CAGRs and correlations (metrics.py)."""

import unittest

import numpy as np
import pandas as pd

import metrics
import sources
from data_definitions import BITCOIN_GENESIS_DATE


class CustomOnchainContractTests(unittest.TestCase):
    def source_frame(self) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=731, freq="D", tz="UTC")
        frame = pd.DataFrame(
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
        frame.loc[index[0], "coinbase_sum_24h_usd"] = np.nan
        frame.loc[index[2], "realized_price"] = np.nan
        return frame

    def test_current_report_metric_names_and_formulas_remain_stable(self):
        result = metrics.calculate_custom_on_chain_metrics(self.source_frame())

        # All-time miner revenue is blank before its first value, then accumulates.
        self.assertTrue(pd.isna(result["thermocap_price"].iloc[0]))
        self.assertEqual(result["thermocap_price"].iloc[1], 0.05)
        self.assertEqual(result["thermocap_price"].iloc[2], 0.10)
        self.assertTrue(pd.isna(result["nvt_price"].iloc[728]))
        self.assertEqual(result["nvt_price"].iloc[729], 5.0)

        self.assertEqual(result["pct_supply_issued"].iloc[-1], 20.0 / 21_000_000)
        self.assertEqual(result["illiquid_supply"].iloc[-1], 12.0)
        self.assertEqual(result["liquid_supply"].iloc[-1], 8.0)
        # Intermediates stay local; only columns a consumer reads are published.
        for unpublished in ("RevAllTimeUSD", "NVTAdj90", "miner_revenue_1_Year", "mvrv_ratio", "hodl_bank_calc"):
            self.assertNotIn(unpublished, result)
        self.assertEqual(result["mvrv"].iloc[-1], 2.5)

        self.assertEqual(result["realized_price"].iloc[0], 7.0)
        self.assertEqual(result["realized_price"].iloc[2], 2.0)


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
        result = metrics.calculate_custom_on_chain_metrics(
            self.frame("2010-01-01", 10)
        )
        first_date = pd.Timestamp("2010-01-01")
        expected_age = (first_date - BITCOIN_GENESIS_DATE).days + 1

        # Cumulative market cap on day one is a single day's 100.0, over a supply of 20.
        self.assertAlmostEqual(
            result["average_cap_price"].iloc[0], 100.0 / expected_age / 20.0, places=9
        )
        self.assertNotAlmostEqual(result["average_cap_price"].iloc[0], 5.0, places=3)

    def test_average_cap_is_independent_of_where_the_history_starts(self):
        """The metric is a property of the network, not of the fetch window."""
        long_run = metrics.calculate_custom_on_chain_metrics(
            self.frame("2010-01-01", 400)
        )
        # A window starting later reaches the same date with fewer rows; the shared
        # dates must still agree once the cumulative base is aligned.
        shared_date = pd.Timestamp("2010-06-01", tz="UTC")
        elapsed = (shared_date.tz_localize(None) - BITCOIN_GENESIS_DATE).days + 1
        rows_before = (shared_date - pd.Timestamp("2010-01-01", tz="UTC")).days + 1
        self.assertAlmostEqual(
            long_run.loc[shared_date, "average_cap_price"],
            (100.0 * rows_before) / elapsed / 20.0,
            places=9,
        )

    def test_delta_cap_absorbs_the_corrected_average_cap(self):
        result = metrics.calculate_custom_on_chain_metrics(
            self.frame("2010-01-01", 5)
        )
        self.assertTrue(
            np.allclose(
                result["delta_cap_price"], 2.0 - result["average_cap_price"], equal_nan=True
            )
        )


class CorrelationTests(unittest.TestCase):
    def test_correlations_pair_returns_over_the_assets_trading_days(self):
        index = pd.date_range("2024-01-01", periods=40, freq="D")  # Monday start
        rng = np.random.default_rng(7)
        btc = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.03, len(index))), index=index)
        # SPY tracks BTC's move since its previous trading day exactly, but only
        # trades Monday-Friday.
        trading = index[index.dayofweek < 5]
        spy = btc.loc[trading] / 10
        marker = sources._source_observation_column("SPY_close")
        raw = pd.DataFrame({"price_close": btc, "SPY_close": spy}, index=index)
        raw[marker] = pd.Series(trading, index=trading).reindex(index)
        # Mimic the bounded ingestion fill that carries Friday into the weekend.
        raw[["SPY_close", marker]] = raw[["SPY_close", marker]].ffill(limit=5)

        observed = metrics.observed_market_values(raw, ["price_close", "SPY_close"])
        self.assertTrue(observed.loc[index.dayofweek >= 5, "SPY_close"].isna().all())

        columns = ["price_close", "SPY_close"]
        result = metrics.create_correlation_matrix_data(index[-1], columns, observed, period=30)
        self.assertAlmostEqual(result.loc["price_close", "SPY_close"], 1.0)
        self.assertAlmostEqual(result.loc["price_close", "price_close"], 1.0)

        # The forward-filled frame would have scored well below 1.
        padded = metrics.create_correlation_matrix_data(index[-1], columns, raw[columns], period=30)
        self.assertLess(padded.loc["price_close", "SPY_close"], 0.95)

    def test_correlations_are_nan_for_stale_or_short_histories(self):
        index = pd.date_range("2024-01-01", periods=40, freq="D")
        prices = pd.DataFrame(
            {
                "price_close": np.linspace(100.0, 140.0, len(index)),
                "OLD_close": np.linspace(10.0, 14.0, len(index)),
                "NEW_close": np.linspace(10.0, 14.0, len(index)),
            },
            index=index,
        )
        prices.loc[index[-10:], "OLD_close"] = np.nan  # stopped trading 10 days ago
        prices.loc[index[:25], "NEW_close"] = np.nan  # only 15 days of history

        result = metrics.create_correlation_matrix_data(
            index[-1], ["price_close", "OLD_close", "NEW_close"], prices, period=30
        )
        self.assertTrue(pd.isna(result.loc["price_close", "OLD_close"]))
        self.assertTrue(pd.isna(result.loc["price_close", "NEW_close"]))


class PeriodReturnBoundaryTests(unittest.TestCase):
    def test_analysis_changes_use_last_positive_close_before_boundary(self):
        dates = pd.to_datetime(
            [
                "2022-12-29",
                "2022-12-30",
                "2022-12-31",
                "2023-01-01",
                "2023-01-30",
                "2023-01-31",
                "2023-02-01",
            ]
        )
        data = pd.DataFrame(
            {
                "a": [80.0, np.nan, 0.0, 100.0, 140.0, 150.0, 180.0],
                "b": [40.0, 50.0, 0.0, 60.0, 90.0, 100.0, 125.0],
            },
            index=dates,
        )

        ytd = metrics.calculate_ytd_change(data)
        mtd = metrics.calculate_mtd_change(data)

        self.assertAlmostEqual(ytd.loc["2023-01-01", "a_ytd_change"], 25.0)
        self.assertAlmostEqual(ytd.loc["2023-01-01", "b_ytd_change"], 20.0)
        self.assertAlmostEqual(mtd.loc["2023-02-01", "a_mtd_change"], 20.0)
        self.assertAlmostEqual(mtd.loc["2023-02-01", "b_mtd_change"], 25.0)


class ElectricityAndChangeTests(unittest.TestCase):
    @staticmethod
    def _energy_input(subsidy=400.0, fees=4.0):
        return pd.DataFrame(
            {
                "hash_rate": [1.0e18],
                "cm_efficiency_j_gh": [0.03],
                "subsidy_sum_24h": [subsidy],
                "fees_sum_24h": [fees],
                "difficulty": [1.0e14],
                "price_close": [100_000.0],
            },
            index=[pd.Timestamp("2024-01-01")],
        )

    def test_electricity_cost_uses_observed_subsidy_plus_fees_and_tariffs(self):
        result = metrics.electric_price_models(self._energy_input()).iloc[0]

        expected_kwh = 1.0e18 / 1.0e9 * 0.03 * 24 / 1000
        expected_revenue = 404.0

        for cents in range(3, 8):
            expected = expected_kwh * (cents / 100) / expected_revenue
            self.assertAlmostEqual(result[f"electricity_cost_{cents}c"], expected)

    def test_electricity_cost_returns_nan_for_zero_miner_revenue(self):
        result = metrics.electric_price_models(
            self._energy_input(subsidy=0.0, fees=0.0)
        ).iloc[0]

        self.assertTrue(pd.isna(result["electricity_cost_5c"]))
        numeric = pd.to_numeric(result, errors="coerce").dropna()
        self.assertFalse(np.isinf(numeric).any())

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

        result = metrics.calculate_yoy_change(data)

        self.assertEqual(result.loc["2020-02-28", "metric_yoy_change"], 100.0)
        self.assertEqual(result.loc["2020-02-29", "metric_yoy_change"], 200.0)
        self.assertEqual(result.loc["2021-02-28", "metric_yoy_change"], 200.0)
        self.assertEqual(result.loc["2021-03-01", "metric_yoy_change"], 100.0)

    def test_only_requested_columns_get_yoy_changes(self):
        dates = pd.date_range("2023-01-01", "2024-03-01", freq="D")
        data = pd.DataFrame(
            {"price_close": np.arange(1.0, len(dates) + 1), "SPY_close": 50.0},
            index=dates,
        )

        result = metrics.calculate_all_changes(data, ["price_close"])

        self.assertEqual(sorted(result.columns), sorted([
            "price_close_7d_change", "price_close_90d_change", "price_close_mtd_change",
            "price_close_ytd_change", "price_close_yoy_change", "SPY_close_7d_change",
            "SPY_close_90d_change", "SPY_close_mtd_change", "SPY_close_ytd_change",
        ]))


class CorrelationSchemaTests(unittest.TestCase):
    def test_absent_asset_keeps_correlation_schema(self):
        frame = pd.DataFrame({'price_close': np.arange(1,41)**2}, index=pd.date_range('2024-01-01', periods=40))
        result = metrics.create_correlation_matrix_data(frame.index[-1], ['price_close', 'MISSING_close'], frame)
        self.assertTrue(pd.isna(result.loc['price_close', 'MISSING_close']))


class CagrTests(unittest.TestCase):
    def test_cagr_uses_calendar_years_across_leap_days(self):
        dates = pd.date_range('2020-01-01', '2024-03-01')
        values = pd.DataFrame({'price_close': np.arange(1.0, len(dates) + 1)}, index=dates)
        cagr = metrics.calculate_rolling_cagr_for_all_columns(values, 4)
        start = values.loc['2020-03-01', 'price_close']
        end = values.loc['2024-03-01', 'price_close']
        expected = ((end / start) ** 0.25 - 1) * 100
        self.assertAlmostEqual(cagr.loc['2024-03-01', 'price_close_4y_cagr'], expected)



class NetworkModelTests(unittest.TestCase):
    """The power-law and Metcalfe fits recover known parameters from the fit window only."""

    def frame(self):
        dates = pd.date_range("2015-01-01", "2024-12-31", freq="D")
        age = (dates - BITCOIN_GENESIS_DATE).days.astype(float)
        price = 1e-17 * age**5.6
        supply = 19_000_000.0
        # Addresses chosen so market cap is exactly 3e-4 x addresses squared.
        addresses = np.sqrt(price * supply / 3e-4)
        return pd.DataFrame(
            {"price_close": price, "market_cap": price * supply, "supply": supply,
             "addr_count": addresses, "hash_rate": np.linspace(1.0, 2.0, len(dates))},
            index=dates,
        )

    def test_fits_recover_the_generating_parameters(self):
        result, parameters = metrics.calculate_network_model_metrics(self.frame(), "2024-12-31")
        self.assertAlmostEqual(parameters["power_law_exponent"], 5.6, places=6)
        self.assertAlmostEqual(parameters["power_law_scale"] / 1e-17, 1.0, places=6)
        self.assertTrue(np.allclose(result["power_law_price_multiple"], 1.0))
        self.assertAlmostEqual(parameters["metcalfe_scale"] / 3e-4, 1.0, places=9)
        self.assertTrue(np.allclose(result["metcalfe_price_multiple"], 1.0))
        # Fitted constants go to the manifest, not into repeated columns.
        for constant in parameters:
            self.assertNotIn(constant, result)

    def test_rows_after_the_report_date_do_not_move_the_fit(self):
        frame = self.frame()
        frame.loc["2024-07-01":, "price_close"] *= 100
        _, parameters = metrics.calculate_network_model_metrics(frame, "2024-06-30")
        self.assertAlmostEqual(parameters["power_law_exponent"], 5.6, places=6)

    def test_hash_ribbon_flags_the_fast_average_below_the_slow(self):
        frame = self.frame()
        frame["hash_rate"] = np.r_[np.full(100, 10.0), np.full(len(frame) - 100, 5.0)]
        result, _ = metrics.calculate_network_model_metrics(frame, "2024-12-31")
        self.assertTrue(result["hash_ribbon_capitulation"].iloc[110])
        self.assertFalse(result["hash_ribbon_capitulation"].iloc[-1])


class RelativePriceTests(unittest.TestCase):
    """Bitcoin prices implied by fiat, metal and stock market caps."""

    def frame(self):
        dates = pd.date_range("2024-01-01", periods=3)
        return pd.DataFrame(
            {"supply": 20.0, "GC=F_close": [2_000.0, 2_100.0, np.nan], "SI=F_close": 25.0,
             "AAA_market_cap": 400.0},
            index=dates,
        )

    def test_fiat_price_divides_money_supply_by_bitcoin_supply(self):
        fiat = pd.DataFrame({"Country": ["United States"], "US Dollar Trillion": [5.0]})
        result = metrics.calculate_btc_price_to_surpass_fiat(self.frame(), fiat)
        self.assertEqual(result["united_states_m0_btc_price"].iloc[0], 5e12 / 20)
        self.assertNotIn("United_States_cap", result)

    def test_metal_caps_use_each_days_price_and_split_by_category(self):
        supply = pd.DataFrame({"Metal": ["Gold", "Silver"], "Supply Troy Ounces": [10.0, 100.0]})
        breakdown = pd.DataFrame({"Gold Supply Breakdown": ["Jewellery", "Other"], "Percentage Of Market": [60.0, 40.0]})
        caps = metrics.calculate_metal_market_caps(self.frame(), supply)
        # Each row uses that day's close; a missing close stays missing.
        self.assertEqual(caps["gold_market_cap_usd"].iloc[0], 10.0 * 2_000.0)
        self.assertEqual(caps["gold_market_cap_usd"].iloc[1], 10.0 * 2_100.0)
        self.assertTrue(pd.isna(caps["gold_market_cap_usd"].iloc[2]))
        self.assertTrue((caps["silver_market_cap_usd"] == 100.0 * 25.0).all())
        prices = metrics.calculate_btc_price_to_surpass_metal_categories(caps, breakdown)
        self.assertEqual(prices["gold_market_cap_btc_price"].iloc[0], 20_000.0 / 20)
        self.assertEqual(prices["gold_jewellery_market_cap_btc_price"].iloc[1], 21_000.0 * 0.6 / 20)
        self.assertEqual(prices["silver_market_cap_btc_price"].iloc[0], 2_500.0 / 20)

    def test_stock_price_divides_market_cap_by_bitcoin_supply(self):
        result = metrics.calculate_btc_price_to_surpass_stocks(self.frame(), ["AAA"])
        self.assertEqual(result["AAA_market_cap_btc_price"].iloc[0], 20.0)

    def test_zero_supply_publishes_nan_not_infinity(self):
        frame = self.frame()
        frame.iloc[0, frame.columns.get_loc("supply")] = 0.0
        fiat = pd.DataFrame({"Country": ["United States"], "US Dollar Trillion": [5.0]})
        fiat_prices = metrics.calculate_btc_price_to_surpass_fiat(frame, fiat)
        stock_prices = metrics.calculate_btc_price_to_surpass_stocks(frame, ["AAA"])
        self.assertTrue(pd.isna(fiat_prices["united_states_m0_btc_price"].iloc[0]))
        self.assertTrue(pd.isna(stock_prices["AAA_market_cap_btc_price"].iloc[0]))
        self.assertEqual(stock_prices["AAA_market_cap_btc_price"].iloc[1], 20.0)


class MovingAverageTests(unittest.TestCase):
    def test_only_30_and_365_day_averages_are_added(self):
        frame = pd.DataFrame({"x": np.arange(400.0)}, index=pd.date_range("2024-01-01", periods=400))
        result = metrics.calculate_moving_averages(frame, ["x"])
        self.assertEqual(sorted(set(result) - {"x"}), ["30_day_ma_x", "365_day_ma_x"])
        self.assertTrue(pd.isna(result["30_day_ma_x"].iloc[28]))
        self.assertEqual(result["30_day_ma_x"].iloc[29], np.arange(30.0).mean())
        self.assertEqual(result["365_day_ma_x"].iloc[-1], np.arange(35.0, 400.0).mean())


class NvtPriceModelTests(unittest.TestCase):
    """Input-smoothed NVT Price models and their multiples."""

    def frame(self, rows=1_200):
        dates = pd.date_range("2020-01-01", periods=rows)
        volume = pd.Series(100.0, index=dates)
        volume.iloc[-1] = 1_000.0  # a one-day volume spike
        return pd.DataFrame(
            {"transfer_volume_sum_24h_usd": volume, "supply": 10.0,
             "market_cap": 1_000.0, "price_close": 100.0},
            index=dates,
        )

    def test_models_smooth_volume_before_valuing_it(self):
        models = metrics.calculate_nvt_price_models(self.frame())
        # The 730-day median NVT is 1,000 / 100 = 10. The unsmoothed model follows the
        # spike; the smoothed models use median volume and ignore it.
        self.assertAlmostEqual(models["nvt_price"].iloc[-1], 10 * 1_000.0 / 10)
        for window in (30, 90, 365):
            self.assertAlmostEqual(models[f"nvt_price_{window}d"].iloc[-1], 10 * 100.0 / 10)
            self.assertAlmostEqual(models[f"nvt_price_multiple_{window}d"].iloc[-1], 1.0)
            # The 730-day NVT median must be complete before any model is published.
            self.assertTrue(pd.isna(models[f"nvt_price_{window}d"].iloc[728]))
            self.assertFalse(pd.isna(models[f"nvt_price_{window}d"].iloc[729]))

    def test_nonpositive_prices_leave_the_multiple_blank(self):
        data = self.frame()
        data.loc[data.index[-1], "price_close"] = 0.0
        models = metrics.calculate_nvt_price_models(data)
        self.assertTrue(pd.isna(models["nvt_price_multiple_90d"].iloc[-1]))


class PowerLawBandTests(unittest.TestCase):
    def test_bands_scale_the_fitted_model_by_the_reviewed_thresholds(self):
        model = pd.Series([100.0, 200.0])
        bands = metrics.calculate_power_law_price_bands(model)
        self.assertEqual(sorted(bands), [
            "power_law_price_band_058", "power_law_price_band_173", "power_law_price_band_300",
        ])
        self.assertTrue(np.allclose(bands["power_law_price_band_058"], [58.0, 116.0]))
        self.assertTrue(np.allclose(bands["power_law_price_band_173"], [173.0, 346.0]))
        self.assertTrue(np.allclose(bands["power_law_price_band_300"], [300.0, 600.0]))


if __name__ == "__main__":
    unittest.main()
