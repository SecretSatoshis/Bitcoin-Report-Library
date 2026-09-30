"""Source fetching and merging (sources.py)."""

import unittest
import warnings
from unittest.mock import patch

import pandas as pd

import sources


class NoMarketCapTicker:
    fast_info = {}
    info = {}


class BrkOnchainTests(unittest.TestCase):
    def test_chunks_merge_into_one_sorted_numeric_daily_frame(self):
        responses = [
            (["timestamp", "a"], [["1704153600", "2"], ["1704067200", "1"]]),
            (["timestamp", "b"], [["1704067200", "4"], ["1704153600", ""]]),
        ]
        with patch.object(sources, "BRK_METRICS", ["timestamp", "a", "b"]), patch.object(
            sources, "_brk_fetch_csv_resilient", return_value=responses
        ):
            frame = sources.get_brk_onchain("2024-01-01")

        self.assertEqual(list(frame.columns), ["time", "a", "b"])
        self.assertEqual(list(frame["time"]), list(pd.to_datetime(["2024-01-01", "2024-01-02"])))
        self.assertEqual(list(frame["a"]), [1.0, 2.0])
        self.assertEqual(frame["b"].iloc[0], 4.0)
        self.assertTrue(pd.isna(frame["b"].iloc[1]))

    def test_an_unresolved_series_fails_the_fetch(self):
        def fetch(chunk, missing=None, **_):
            missing.append("b")
            return [(["timestamp", "a"], [["1704067200", "1"]])]

        with patch.object(sources, "BRK_METRICS", ["timestamp", "a", "b"]), patch.object(
            sources, "_brk_fetch_csv_resilient", side_effect=fetch
        ), self.assertRaisesRegex(RuntimeError, "required series: b"):
            sources.get_brk_onchain("2024-01-01")


class GetDataTests(unittest.TestCase):
    def test_two_sources_with_the_same_column_fail(self):
        dates = pd.date_range("2024-01-01", periods=3)
        onchain = pd.DataFrame({"time": dates, "price_close": 1.0})
        prices = pd.DataFrame({"time": dates, "price_close": 2.0})
        with patch.object(sources, "get_brk_onchain", return_value=onchain), patch.object(
            sources, "get_price", return_value=prices
        ), patch.object(sources, "get_marketcap", return_value=pd.DataFrame()), patch.object(
            sources, "get_miner_data", return_value=pd.DataFrame()
        ), self.assertRaisesRegex(RuntimeError, "duplicate columns: price_close"):
            sources.get_data({}, "2024-01-01")


class MarketCapErrorTests(unittest.TestCase):
    def test_unavailable_data_warns_and_leaves_the_column_blank(self):
        with patch.object(sources.yf, "Ticker", return_value=NoMarketCapTicker()), patch.object(
            sources, "_historical_market_cap", side_effect=ValueError("no shares")
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sources.get_marketcap({"stocks": ["AAA"]}, "2024-01-01", end_date="2024-01-05")

        self.assertTrue(result["AAA_MarketCap"].isna().all())
        self.assertTrue(any("no shares" in str(w.message) for w in caught))

    def test_a_coding_error_is_not_swallowed(self):
        with patch.object(sources.yf, "Ticker", return_value=NoMarketCapTicker()), patch.object(
            sources, "_historical_market_cap", side_effect=AttributeError("bug")
        ), self.assertRaises(AttributeError):
            sources.get_marketcap({"stocks": ["AAA"]}, "2024-01-01", end_date="2024-01-05")


if __name__ == "__main__":
    unittest.main()
