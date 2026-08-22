"""Regression tests for Yahoo historical market-cap reconstruction."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import data_format


class FakeYahooTicker:
    def __init__(
        self,
        history=None,
        shares=None,
        market_cap=None,
        info_market_cap=None,
    ):
        self._history = pd.DataFrame() if history is None else history
        self._shares = shares
        self._fast_info = (
            {} if market_cap is None else {"market_cap": market_cap}
        )
        self._info = (
            {} if info_market_cap is None else {"marketCap": info_market_cap}
        )
        self._tz = None
        self.history_calls = []
        self.share_calls = []

    def history(self, **kwargs):
        self.history_calls.append(kwargs)
        return self._history.copy()

    def get_shares_full(self, **kwargs):
        self.share_calls.append(kwargs)
        return None if self._shares is None else self._shares.copy()

    @property
    def fast_info(self):
        return self._fast_info

    @property
    def info(self):
        return self._info


class YahooMarketCapTests(unittest.TestCase):
    def test_history_is_clamped_to_2015_without_backward_fill(self):
        history = pd.DataFrame(
            {"Close": [10.0], "Stock Splits": [0.0]},
            index=pd.to_datetime(["2015-01-02"]),
        )
        shares = pd.Series([100.0], index=pd.to_datetime(["2015-01-02"]))
        ticker = FakeYahooTicker(history=history, shares=shares)

        with patch.object(data_format.yf, "Ticker", return_value=ticker):
            result = data_format.get_marketcap(
                {"stocks": ["TEST"]},
                "2014-12-30",
                end_date="2015-01-02",
            ).set_index("time")

        self.assertEqual(ticker.history_calls[0]["start"], "2015-01-01")
        self.assertTrue(result.loc[:"2015-01-01", "TEST_MarketCap"].isna().all())
        self.assertEqual(result.loc["2015-01-02", "TEST_MarketCap"], 1_000.0)

    def test_historical_marketcap_uses_close_and_handles_split_duplicates(self):
        dates = pd.to_datetime(
            ["2020-08-27", "2020-08-28", "2020-08-31", "2020-09-01"]
        )
        history = pd.DataFrame(
            {
                "Close": [10.0, 10.0, 10.0, 10.0],
                "Adj Close": [9.0, 9.0, 9.0, 9.0],
                "Stock Splits": [0.0, 0.0, 4.0, 0.0],
            },
            index=dates,
        )
        shares = pd.Series(
            [25.0, 100.0, 25.0, 100.0],
            index=pd.to_datetime(
                ["2020-08-28", "2020-08-31", "2020-08-31", "2020-09-01"]
            ),
        )
        ticker = FakeYahooTicker(history=history, shares=shares)

        with patch.object(data_format.yf, "Ticker", return_value=ticker):
            result = data_format.get_marketcap(
                {"stocks": ["TEST"]},
                "2020-08-27",
                end_date="2020-09-01",
            ).set_index("time")

        self.assertTrue(pd.isna(result.loc["2020-08-27", "TEST_MarketCap"]))
        self.assertEqual(result.loc["2020-08-28", "TEST_MarketCap"], 1_000.0)
        self.assertEqual(result.loc["2020-08-31", "TEST_MarketCap"], 1_000.0)
        self.assertEqual(result.loc["2020-09-01", "TEST_MarketCap"], 1_000.0)
        self.assertFalse(ticker.history_calls[0]["auto_adjust"])
        self.assertTrue(ticker.history_calls[0]["actions"])

        downstream = result.copy()
        downstream["supply"] = 20.0
        downstream = data_format.calculate_btc_price_for_stock_mkt_caps(
            downstream, ["TEST"]
        )
        self.assertEqual(
            downstream.loc["2020-09-01", "TEST_mc_btc_price"], 50.0
        )

    def test_split_adjustment_finds_leading_and_lagging_share_transitions(self):
        split = pd.Series(
            [4.0], index=pd.to_datetime(["2020-08-31"]), dtype="float64"
        )
        leading = pd.Series(
            [25.0, 100.0],
            index=pd.to_datetime(["2020-08-20", "2020-08-28"]),
        )
        lagging = pd.Series(
            [25.0, 25.0, 100.0],
            index=pd.to_datetime(["2020-08-20", "2020-08-31", "2020-09-04"]),
        )

        leading_result = data_format._split_adjust_yahoo_shares(leading, split)
        lagging_result = data_format._split_adjust_yahoo_shares(lagging, split)

        self.assertTrue(leading_result.eq(100.0).all())
        self.assertTrue(lagging_result.eq(100.0).all())

    def test_isolated_yahoo_share_outlier_is_not_forward_filled(self):
        shares = pd.Series(
            [100.0, 50.0, 101.0],
            index=pd.to_datetime(["2023-06-05", "2023-06-09", "2023-06-13"]),
        )

        result = data_format._split_adjust_yahoo_shares(
            shares, pd.Series(dtype="float64")
        )

        self.assertNotIn(pd.Timestamp("2023-06-09"), result.index)
        self.assertEqual(result.tolist(), [100.0, 101.0])

    def test_alias_share_histories_keep_current_marketcap_names(self):
        meta_dates = pd.to_datetime(["2022-05-27", "2022-06-09", "2022-06-10"])
        meta_history = pd.DataFrame(
            {"Close": [10.0, 10.0, 10.0], "Stock Splits": [0.0, 0.0, 0.0]},
            index=meta_dates.tz_localize("America/New_York"),
        )
        meta = FakeYahooTicker(
            history=meta_history,
            shares=pd.Series([80.0], index=pd.to_datetime(["2022-06-09"])),
        )
        fb = FakeYahooTicker(
            shares=pd.Series(
                [100.0, 90.0],
                index=pd.to_datetime(["2022-05-27", "2022-06-09"]),
            )
        )
        ticker_objects = {"META": meta, "FB": fb}

        with patch.object(
            data_format.yf, "Ticker", side_effect=lambda symbol: ticker_objects[symbol]
        ):
            result = data_format.get_marketcap(
                {"stocks": ["META"]},
                "2022-05-27",
                end_date="2022-06-10",
            ).set_index("time")

        self.assertEqual(result.loc["2022-05-27", "META_MarketCap"], 1_000.0)
        # Current-ticker META observation wins the overlapping date over FB's 90 shares.
        self.assertEqual(result.loc["2022-06-09", "META_MarketCap"], 800.0)
        self.assertNotIn("FB_MarketCap", result.columns)
        self.assertEqual(fb._tz, "America/New_York")

    def test_missing_history_uses_current_cap_only_on_final_date(self):
        failed = FakeYahooTicker(market_cap=1_234.0)

        with patch.object(data_format.yf, "Ticker", return_value=failed), \
            self.assertWarnsRegex(RuntimeWarning, "no closing-price history"):
            result = data_format.get_marketcap(
                {"stocks": ["FAIL"]},
                "2020-01-01",
                end_date="2020-01-03",
            ).set_index("time")

        self.assertTrue(pd.isna(result.loc["2020-01-01", "FAIL_MarketCap"]))
        self.assertTrue(pd.isna(result.loc["2020-01-02", "FAIL_MarketCap"]))
        self.assertEqual(result.loc["2020-01-03", "FAIL_MarketCap"], 1_234.0)

    def test_failed_ticker_retains_schema_without_harming_valid_ticker(self):
        dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
        valid = FakeYahooTicker(
            history=pd.DataFrame(
                {"Close": [10.0, 11.0], "Stock Splits": [0.0, 0.0]},
                index=dates,
            ),
            shares=pd.Series([100.0], index=pd.to_datetime(["2020-01-02"])),
        )
        failed = FakeYahooTicker()
        ticker_objects = {"GOOD": valid, "FAIL": failed}

        with patch.object(
            data_format.yf, "Ticker", side_effect=lambda symbol: ticker_objects[symbol]
        ), self.assertWarnsRegex(RuntimeWarning, "FAIL"):
            result = data_format.get_marketcap(
                {"stocks": ["GOOD", "FAIL"], "etfs": ["SPY"]},
                "2020-01-01",
                end_date="2020-01-03",
            )

        self.assertEqual(
            list(result.columns), ["time", "GOOD_MarketCap", "FAIL_MarketCap"]
        )
        self.assertEqual(
            result["GOOD_MarketCap"].dropna().tolist(), [1_000.0, 1_100.0]
        )
        self.assertTrue(result["FAIL_MarketCap"].isna().all())
        self.assertTrue(result["time"].is_monotonic_increasing)
        self.assertFalse(result["time"].duplicated().any())

    def test_local_currency_marketcap_is_converted_to_usd(self):
        dates = pd.to_datetime(["2026-08-19", "2026-08-20"])
        local_stock = FakeYahooTicker(
            history=pd.DataFrame(
                {"Close": [25.0, 26.0], "Stock Splits": [0.0, 0.0]},
                index=dates,
            ),
            shares=pd.Series([100.0], index=pd.to_datetime(["2026-08-19"])),
        )
        sar_usd = FakeYahooTicker(
            history=pd.DataFrame({"Close": [0.266, 0.267]}, index=dates)
        )
        ticker_objects = {"2222.SR": local_stock, "SARUSD=X": sar_usd}

        with patch.object(
            data_format.yf, "Ticker", side_effect=lambda symbol: ticker_objects[symbol]
        ):
            result = data_format.get_marketcap(
                {"stocks": ["2222.SR"]},
                "2026-08-19",
                end_date="2026-08-20",
            ).set_index("time")

        self.assertAlmostEqual(
            result.loc["2026-08-19", "2222.SR_MarketCap"], 665.0
        )
        self.assertAlmostEqual(
            result.loc["2026-08-20", "2222.SR_MarketCap"], 694.2
        )


if __name__ == "__main__":
    unittest.main()
