"""Source fetching and merging (sources.py)."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

import freshness
import metrics
import sources


class FakeResponse:
    def __init__(self, status_code=200, text="", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = {} if json_data is None else json_data
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(
                f"HTTP {self.status_code}", response=self
            )


class NoMarketCapTicker:
    fast_info = {}
    info = {}


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


class SharesOutstandingBudgetTests(unittest.TestCase):
    """The share-count fill is bounded by a cadence-appropriate budget."""

    def test_budget_clears_a_semiannual_filer_but_not_a_dormant_one(self):
        # Observed worst case among tracked tickers is 2222.SR at ~162 days.
        self.assertGreater(sources.SHARES_OUTSTANDING_MAX_AGE_DAYS, 162)
        self.assertLess(sources.SHARES_OUTSTANDING_MAX_AGE_DAYS, 365)

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
        budget = sources.SHARES_OUTSTANDING_MAX_AGE_DAYS
        masked = filled.where(age.between(0, budget))

        self.assertTrue(masked.iloc[0] == 1_000.0)
        self.assertTrue(masked.loc[close.index[budget]] == 1_000.0)
        self.assertTrue(pd.isna(masked.loc[close.index[budget + 1]]))


class BrkAndYahooFetchTests(unittest.TestCase):
    def test_brk_ohlc_fetch_errors_and_empty_payloads_fail_loudly(self):
        with patch.object(
            sources.requests,
            "get",
            side_effect=requests.Timeout("timed out"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to fetch usable BRK"):
                sources.get_brk_ohlc()

        empty = FakeResponse(json_data={"data": []})
        with patch.object(sources.requests, "get", side_effect=[empty, empty]):
            with self.assertRaisesRegex(RuntimeError, "returned no day1 OHLC"):
                sources.get_brk_ohlc()

    def test_brk_bulk_retries_request_errors_and_transient_statuses(self):
        transient = FakeResponse(status_code=503, text="unavailable")
        success = FakeResponse(
            text="timestamp,price_close\n1704067200,42000\n"
        )
        with patch.object(
            sources.requests,
            "get",
            side_effect=[
                requests.exceptions.ChunkedEncodingError("truncated"),
                transient,
                success,
            ],
        ) as get_mock, patch.object(sources.time, "sleep") as sleep_mock:
            header, rows = sources._brk_fetch_csv(
                ["timestamp", "price_close"]
            )

        self.assertEqual(header, ["timestamp", "price_close"])
        self.assertEqual(rows, [["1704067200", "42000"]])
        self.assertEqual(get_mock.call_count, sources.BRK_BULK_MAX_ATTEMPTS)
        self.assertTrue(
            all(
                call.kwargs["timeout"] == sources.API_TIMEOUT
                for call in get_mock.call_args_list
            )
        )
        self.assertEqual(
            [call.args[0] for call in sleep_mock.call_args_list], [1.0, 2.0]
        )

    def test_brk_bulk_retry_count_is_bounded(self):
        with patch.object(
            sources.requests,
            "get",
            side_effect=requests.ConnectionError("offline"),
        ) as get_mock, patch.object(sources.time, "sleep") as sleep_mock:
            with self.assertRaises(requests.ConnectionError):
                sources._brk_fetch_csv(
                    ["timestamp", "price_close"],
                    max_attempts=3,
                    initial_backoff_seconds=0.25,
                )

        self.assertEqual(get_mock.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep_mock.call_args_list], [0.25, 0.5]
        )

    def test_brk_semantic_error_splits_without_transient_retries(self):
        def fake_get(_url, params, timeout):
            self.assertEqual(timeout, sources.API_TIMEOUT)
            requested = params["series"].split(",")
            non_timestamp = [name for name in requested if name != "timestamp"]
            if len(non_timestamp) > 1:
                return FakeResponse(
                    status_code=503,
                    text="semantic failure",
                    json_data={"error": {"code": "weight_exceeded"}},
                )
            metric = non_timestamp[0]
            return FakeResponse(
                text=f"timestamp,{metric}\n1704067200,1\n"
            )

        with patch.object(
            sources.requests, "get", side_effect=fake_get
        ) as get_mock, patch.object(sources.time, "sleep") as sleep_mock:
            responses = sources._brk_fetch_csv_resilient(
                ["timestamp", "metric_a", "metric_b"]
            )

        self.assertEqual(len(responses), 2)
        self.assertEqual(get_mock.call_count, 3)
        sleep_mock.assert_not_called()

    def test_source_fetch_reindex_uses_bounded_fill_and_provenance(self):
        index = pd.to_datetime(["2024-01-01", "2024-01-10"])
        raw = pd.DataFrame(
            [[100.0], [110.0]],
            index=index,
            columns=pd.MultiIndex.from_tuples([("SPY", "Close")]),
        )
        marker = sources._source_observation_column("SPY_close")

        with patch.object(sources.yf, "download", return_value=raw):
            result = sources.get_price(
                {"stocks": ["SPY"]}, start_date="2024-01-01"
            ).set_index("time")

        self.assertEqual(result.loc["2024-01-06", "SPY_close"], 100.0)
        self.assertTrue(pd.isna(result.loc["2024-01-07", "SPY_close"]))
        self.assertEqual(result.loc["2024-01-06", marker], pd.Timestamp("2024-01-01"))
        self.assertTrue(pd.isna(result.loc["2024-01-07", marker]))

    def test_market_fill_honors_total_source_age_for_prices_and_market_caps(self):
        index = pd.date_range("2024-01-01", periods=10, freq="D")
        marker = sources._source_observation_column("SPY_close")
        data = pd.DataFrame(
            {
                "price_close": [40_000.0] + [np.nan] * 9,
                "SPY_close": [100.0] * 6 + [np.nan] * 4,
                marker: [pd.Timestamp("2024-01-01")] * 6 + [pd.NaT] * 4,
                "AAPL_market_cap": [3_000.0] + [np.nan] * 9,
                sources.MINER_EFFICIENCY_VALUE_COLUMN: [0.03]
                + [np.nan] * 9,
                sources.MINER_EFFICIENCY_SOURCE_DATE_COLUMN: [
                    pd.Timestamp("2024-01-01")
                ]
                + [pd.NaT] * 9,
                sources.MINER_EFFICIENCY_SOURCE_URL_COLUMN: ["sheet-url"]
                + [np.nan] * 9,
            },
            index=index,
        )

        result = freshness.forward_fill_market_data(
            data, market_max_age_days=5, miner_max_age_days=8
        )

        self.assertTrue(pd.isna(result.loc["2024-01-07", "SPY_close"]))
        self.assertNotIn(marker, result.columns)
        self.assertEqual(result.loc["2024-01-06", "AAPL_market_cap"], 3_000.0)
        self.assertTrue(pd.isna(result.loc["2024-01-07", "AAPL_market_cap"]))
        self.assertTrue(pd.isna(result.loc["2024-01-02", "price_close"]))
        self.assertEqual(
            result.loc["2024-01-09", sources.MINER_EFFICIENCY_VALUE_COLUMN],
            0.03,
        )
        self.assertTrue(
            pd.isna(
                result.loc[
                    "2024-01-10", sources.MINER_EFFICIENCY_VALUE_COLUMN
                ]
            )
        )

    def test_miner_fetch_uses_timeout_and_retains_source_provenance(self):
        response = FakeResponse(
            text=(
                "time,efficiency_j_th\n"
                "2024-01-01,30\n"
                "2024-02-01,28\n"
            )
        )
        sheet_url = "https://example.test/sheet/edit?usp=sharing"
        export_url = "https://example.test/sheet/export?format=csv"

        with patch.object(
            sources.requests, "get", return_value=response
        ) as get_mock:
            result = sources.get_miner_data(sheet_url).set_index("time")

        get_mock.assert_called_once_with(export_url, timeout=sources.API_TIMEOUT)
        self.assertEqual(
            result.loc["2024-01-31", sources.MINER_EFFICIENCY_VALUE_COLUMN],
            0.03,
        )
        self.assertEqual(
            result.loc[
                "2024-01-31", sources.MINER_EFFICIENCY_SOURCE_DATE_COLUMN
            ],
            pd.Timestamp("2024-01-01"),
        )
        self.assertEqual(
            result.loc[
                "2024-01-31", sources.MINER_EFFICIENCY_SOURCE_URL_COLUMN
            ],
            export_url,
        )

    def test_stale_miner_value_is_retained_with_its_true_source_date(self):
        index = pd.date_range("2024-01-01", "2024-03-05", freq="D")
        source_date = pd.Timestamp("2024-01-01")
        data = pd.DataFrame(
            {
                sources.MINER_EFFICIENCY_VALUE_COLUMN: [0.03] * len(index),
                sources.MINER_EFFICIENCY_SOURCE_DATE_COLUMN: [source_date]
                * len(index),
                sources.MINER_EFFICIENCY_SOURCE_URL_COLUMN: ["sheet-url"]
                * len(index),
            },
            index=index,
        )

        with self.assertWarnsRegex(
            RuntimeWarning, "source observation 2024-01-01, which is 64 days old"
        ):
            issues = freshness.warn_on_stale_miner_efficiency(
                data, "2024-03-05", max_age_days=62
            )
        self.assertEqual(len(issues), 1)
        self.assertEqual(
            data.loc["2024-03-05", sources.MINER_EFFICIENCY_VALUE_COLUMN],
            0.03,
        )

    def test_default_miner_fill_carries_last_available_estimate(self):
        index = pd.date_range("2024-01-01", "2024-05-01", freq="D")
        data = pd.DataFrame(
            {
                sources.MINER_EFFICIENCY_VALUE_COLUMN: [0.03]
                + [np.nan] * (len(index) - 1),
                sources.MINER_EFFICIENCY_SOURCE_DATE_COLUMN: [
                    pd.Timestamp("2024-01-01")
                ]
                + [pd.NaT] * (len(index) - 1),
                sources.MINER_EFFICIENCY_SOURCE_URL_COLUMN: ["sheet-url"]
                + [np.nan] * (len(index) - 1),
            },
            index=index,
        )

        result = freshness.forward_fill_market_data(data)

        self.assertEqual(
            result.loc["2024-05-01", sources.MINER_EFFICIENCY_VALUE_COLUMN],
            0.03,
        )
        self.assertEqual(
            result.loc[
                "2024-05-01", sources.MINER_EFFICIENCY_SOURCE_DATE_COLUMN
            ],
            pd.Timestamp("2024-01-01"),
        )


class BrkOhlcContractTests(unittest.TestCase):
    def test_ohlc_metadata_and_candle_contract(self):
        base = dict(index='day1', start=10, end=12)
        dates = FakeResponse(json_data=dict(base, data=['2024-01-01', '2024-01-02']))
        good = FakeResponse(json_data=dict(base, data=[[100,110,90,105]]*2))
        with patch.object(sources.requests, 'get', side_effect=[dates, good]):
            self.assertEqual(len(sources.get_brk_ohlc()), 2)
        for payload in (dict(base, index='hour1', data=[[100,110,90,105]]*2),
                        dict(base, start=20, end=22, data=[[100,110,90,105]]*2),
                        dict(base, data=[[100,90,110,100]]*2)):
            with patch.object(sources.requests, 'get', side_effect=[dates, FakeResponse(json_data=payload)]):
                with self.assertRaises(RuntimeError):
                    sources.get_brk_ohlc()


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


class PrePricePlaceholderTests(unittest.TestCase):
    def test_zero_placeholders_before_the_first_price_are_blanked(self):
        frame = pd.DataFrame(
            {"price_close": [0.0, 0.0, 5.0, 6.0], "market_cap": [0.0, 0.0, 50.0, 60.0],
             "lth_realized_price": [0.0, 0.0, 0.0, 4.0], "fees_sum_24h": [0.0, 0.0, 0.0, 1.0],
             "supply": [1.0, 2.0, 3.0, 4.0]},
            index=pd.date_range("2010-08-14", periods=4),
        )
        result = sources._blank_pre_price_placeholders(frame)
        self.assertTrue(result[["price_close", "market_cap"]].iloc[:2].isna().all().all())
        self.assertEqual(result["market_cap"].iloc[2], 50.0)
        # An empty cohort's realized price of 0 is blank even after the first price.
        self.assertTrue(result["lth_realized_price"].iloc[:3].isna().all())
        # Real zeros in series that do not need a price are kept.
        self.assertEqual(result["fees_sum_24h"].iloc[0], 0.0)
        self.assertEqual(list(result["supply"]), [1.0, 2.0, 3.0, 4.0])


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

        self.assertTrue(result["AAA_market_cap"].isna().all())
        self.assertTrue(any("no shares" in str(w.message) for w in caught))

    def test_a_coding_error_is_not_swallowed(self):
        with patch.object(sources.yf, "Ticker", return_value=NoMarketCapTicker()), patch.object(
            sources, "_historical_market_cap", side_effect=AttributeError("bug")
        ), self.assertRaises(AttributeError):
            sources.get_marketcap({"stocks": ["AAA"]}, "2024-01-01", end_date="2024-01-05")


class YahooMarketCapTests(unittest.TestCase):
    def test_history_is_clamped_to_2015_without_backward_fill(self):
        history = pd.DataFrame(
            {"Close": [10.0], "Stock Splits": [0.0]},
            index=pd.to_datetime(["2015-01-02"]),
        )
        shares = pd.Series([100.0], index=pd.to_datetime(["2015-01-02"]))
        ticker = FakeYahooTicker(history=history, shares=shares)

        with patch.object(sources.yf, "Ticker", return_value=ticker):
            result = sources.get_marketcap(
                {"stocks": ["TEST"]},
                "2014-12-30",
                end_date="2015-01-02",
            ).set_index("time")

        self.assertEqual(ticker.history_calls[0]["start"], "2015-01-01")
        self.assertTrue(result.loc[:"2015-01-01", "TEST_market_cap"].isna().all())
        self.assertEqual(result.loc["2015-01-02", "TEST_market_cap"], 1_000.0)

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

        with patch.object(sources.yf, "Ticker", return_value=ticker):
            result = sources.get_marketcap(
                {"stocks": ["TEST"]},
                "2020-08-27",
                end_date="2020-09-01",
            ).set_index("time")

        self.assertTrue(pd.isna(result.loc["2020-08-27", "TEST_market_cap"]))
        self.assertEqual(result.loc["2020-08-28", "TEST_market_cap"], 1_000.0)
        self.assertEqual(result.loc["2020-08-31", "TEST_market_cap"], 1_000.0)
        self.assertEqual(result.loc["2020-09-01", "TEST_market_cap"], 1_000.0)
        self.assertFalse(ticker.history_calls[0]["auto_adjust"])
        self.assertTrue(ticker.history_calls[0]["actions"])

        downstream = result.copy()
        downstream["supply"] = 20.0
        downstream = metrics.calculate_btc_price_for_stock_mkt_caps(
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

        leading_result = sources._split_adjust_yahoo_shares(leading, split)
        lagging_result = sources._split_adjust_yahoo_shares(lagging, split)

        self.assertTrue(leading_result.eq(100.0).all())
        self.assertTrue(lagging_result.eq(100.0).all())

    def test_isolated_yahoo_share_outlier_is_not_forward_filled(self):
        shares = pd.Series(
            [100.0, 50.0, 101.0],
            index=pd.to_datetime(["2023-06-05", "2023-06-09", "2023-06-13"]),
        )

        result = sources._split_adjust_yahoo_shares(
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
            sources.yf, "Ticker", side_effect=lambda symbol: ticker_objects[symbol]
        ):
            result = sources.get_marketcap(
                {"stocks": ["META"]},
                "2022-05-27",
                end_date="2022-06-10",
            ).set_index("time")

        self.assertEqual(result.loc["2022-05-27", "META_market_cap"], 1_000.0)
        # Current-ticker META observation wins the overlapping date over FB's 90 shares.
        self.assertEqual(result.loc["2022-06-09", "META_market_cap"], 800.0)
        self.assertNotIn("FB_market_cap", result.columns)
        self.assertEqual(fb._tz, "America/New_York")

    def test_missing_history_uses_current_cap_only_on_final_date(self):
        failed = FakeYahooTicker(market_cap=1_234.0)

        with patch.object(sources.yf, "Ticker", return_value=failed), \
            self.assertWarnsRegex(RuntimeWarning, "no closing-price history"):
            result = sources.get_marketcap(
                {"stocks": ["FAIL"]},
                "2020-01-01",
                end_date="2020-01-03",
            ).set_index("time")

        self.assertTrue(pd.isna(result.loc["2020-01-01", "FAIL_market_cap"]))
        self.assertTrue(pd.isna(result.loc["2020-01-02", "FAIL_market_cap"]))
        self.assertEqual(result.loc["2020-01-03", "FAIL_market_cap"], 1_234.0)

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
            sources.yf, "Ticker", side_effect=lambda symbol: ticker_objects[symbol]
        ), self.assertWarnsRegex(RuntimeWarning, "FAIL"):
            result = sources.get_marketcap(
                {"stocks": ["GOOD", "FAIL"], "etfs": ["SPY"]},
                "2020-01-01",
                end_date="2020-01-03",
            )

        self.assertEqual(
            list(result.columns), ["time", "GOOD_market_cap", "FAIL_market_cap"]
        )
        self.assertEqual(
            result["GOOD_market_cap"].dropna().tolist(), [1_000.0, 1_100.0]
        )
        self.assertTrue(result["FAIL_market_cap"].isna().all())
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
            sources.yf, "Ticker", side_effect=lambda symbol: ticker_objects[symbol]
        ):
            result = sources.get_marketcap(
                {"stocks": ["2222.SR"]},
                "2026-08-19",
                end_date="2026-08-20",
            ).set_index("time")

        self.assertAlmostEqual(
            result.loc["2026-08-19", "2222.SR_market_cap"], 665.0
        )
        self.assertAlmostEqual(
            result.loc["2026-08-20", "2222.SR_market_cap"], 694.2
        )


if __name__ == "__main__":
    unittest.main()
