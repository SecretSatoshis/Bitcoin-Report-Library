"""Regression tests for source ingestion and freshness controls."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

import data_validation
import freshness
import metrics
import sources
import report_tables


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


class IngestionReliabilityTests(unittest.TestCase):
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

    def test_ohlc_tables_refuse_empty_candles(self):
        empty = pd.DataFrame(columns=data_validation.OHLC_COLUMNS)
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            report_tables.weekly_ohlc_table(empty)
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            report_tables.create_report_ohlc_summary(empty, "2024-01-01")

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
                "AAPL_MarketCap": [3_000.0] + [np.nan] * 9,
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
        self.assertEqual(result.loc["2024-01-06", "AAPL_MarketCap"], 3_000.0)
        self.assertTrue(pd.isna(result.loc["2024-01-07", "AAPL_MarketCap"]))
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

        result = metrics.create_btc_correlation_data(
            index[-1], {"etfs": ["SPY"]}, observed, periods=[30]
        )["price_close_30_days"]
        self.assertAlmostEqual(result.loc["price_close", "SPY_close"], 1.0)
        self.assertEqual(result.loc["price_close", "price_close"], 1.0)

        # The forward-filled frame would have scored well below 1.
        padded = metrics.create_btc_correlation_data(
            index[-1], {"etfs": ["SPY"]}, raw[["price_close", "SPY_close"]], periods=[30]
        )["price_close_30_days"]
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

        result = metrics.create_btc_correlation_data(
            index[-1], {"stocks": ["OLD", "NEW"]}, prices, periods=[30]
        )["price_close_30_days"]
        self.assertTrue(pd.isna(result.loc["price_close", "OLD_close"]))
        self.assertTrue(pd.isna(result.loc["price_close", "NEW_close"]))

    def test_onchain_freshness_requires_the_report_date_row(self):
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {metric: 1.0 for metric in freshness.REQUIRED_ONCHAIN_METRICS},
            index=index,
        )
        freshness.assert_onchain_freshness(data, index[-1])
        with self.assertRaisesRegex(RuntimeError, "stale"):
            freshness.assert_onchain_freshness(data, index[-1] + pd.Timedelta(days=1))


if __name__ == "__main__":
    unittest.main()
