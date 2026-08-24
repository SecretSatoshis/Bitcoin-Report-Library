import tempfile
"""Regression tests for source ingestion and freshness controls."""

import unittest
from datetime import datetime as real_datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

import data_format
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
            data_format.requests,
            "get",
            side_effect=requests.Timeout("timed out"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to fetch usable BRK"):
                data_format.get_brk_ohlc()

        empty = FakeResponse(json_data={"data": []})
        with patch.object(data_format.requests, "get", side_effect=[empty, empty]):
            with self.assertRaisesRegex(RuntimeError, "returned no week1 OHLC"):
                data_format.get_brk_ohlc()

    def test_ohlc_writers_do_not_replace_existing_files_with_empty_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            weekly_path = Path(temp_dir) / "weekly.csv"
            daily_path = Path(temp_dir) / "daily.csv"
            weekly_path.write_text("existing-weekly\n", encoding="utf-8")
            daily_path.write_text("existing-daily\n", encoding="utf-8")

            empty = pd.DataFrame(columns=data_format.OHLC_COLUMNS)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                report_tables.calculate_ohlc(empty, output_file=weekly_path)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                report_tables.create_report_ohlc_summary(
                    empty, "2024-01-01", output_file=daily_path
                )

            self.assertEqual(
                weekly_path.read_text(encoding="utf-8"), "existing-weekly\n"
            )
            self.assertEqual(
                daily_path.read_text(encoding="utf-8"), "existing-daily\n"
            )

    def test_brk_bulk_retries_request_errors_and_transient_statuses(self):
        transient = FakeResponse(status_code=503, text="unavailable")
        success = FakeResponse(
            text="timestamp,price_close\n1704067200,42000\n"
        )
        with patch.object(
            data_format.requests,
            "get",
            side_effect=[
                requests.exceptions.ChunkedEncodingError("truncated"),
                transient,
                success,
            ],
        ) as get_mock, patch.object(data_format.time, "sleep") as sleep_mock:
            header, rows, _ = data_format._brk_fetch_csv(
                ["timestamp", "price_close"]
            )

        self.assertEqual(header, ["timestamp", "price_close"])
        self.assertEqual(rows, [["1704067200", "42000"]])
        self.assertEqual(get_mock.call_count, data_format.BRK_BULK_MAX_ATTEMPTS)
        self.assertTrue(
            all(
                call.kwargs["timeout"] == data_format.API_TIMEOUT
                for call in get_mock.call_args_list
            )
        )
        self.assertEqual(
            [call.args[0] for call in sleep_mock.call_args_list], [1.0, 2.0]
        )

    def test_brk_bulk_retry_count_is_bounded(self):
        with patch.object(
            data_format.requests,
            "get",
            side_effect=requests.ConnectionError("offline"),
        ) as get_mock, patch.object(data_format.time, "sleep") as sleep_mock:
            with self.assertRaises(requests.ConnectionError):
                data_format._brk_fetch_csv(
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
            self.assertEqual(timeout, data_format.API_TIMEOUT)
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
            data_format.requests, "get", side_effect=fake_get
        ) as get_mock, patch.object(data_format.time, "sleep") as sleep_mock:
            responses = data_format._brk_fetch_csv_resilient(
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
        marker = data_format._source_observation_column("SPY_close")

        with patch.object(data_format.yf, "download", return_value=raw), patch.object(
            data_format, "datetime"
        ) as datetime_mock:
            datetime_mock.today.return_value = real_datetime(2024, 1, 10)
            result = data_format.get_price(
                {"stocks": ["SPY"]}, start_date="2024-01-01"
            ).set_index("time")

        self.assertEqual(result.loc["2024-01-06", "SPY_close"], 100.0)
        self.assertTrue(pd.isna(result.loc["2024-01-07", "SPY_close"]))
        self.assertEqual(result.loc["2024-01-06", marker], pd.Timestamp("2024-01-01"))
        self.assertTrue(pd.isna(result.loc["2024-01-07", marker]))

    def test_crypto_source_resample_uses_the_same_bounded_fill(self):
        jan_1_ms = 1_704_067_200_000
        jan_10_ms = 1_704_844_800_000
        response = FakeResponse(
            json_data={
                "prices": [[jan_1_ms, 2_000.0], [jan_10_ms, 2_100.0]],
                "total_volumes": [[jan_1_ms, 10.0], [jan_10_ms, 11.0]],
                "market_caps": [[jan_1_ms, 20.0], [jan_10_ms, 21.0]],
            }
        )
        marker = data_format._source_observation_column("ethereum_close")

        with patch.object(
            data_format.requests, "get", return_value=response
        ), patch.object(data_format.time, "sleep"):
            result = data_format.get_crypto_data(["ethereum"]).set_index("time")

        self.assertEqual(result.loc["2024-01-06", "ethereum_close"], 2_000.0)
        self.assertTrue(pd.isna(result.loc["2024-01-07", "ethereum_close"]))
        self.assertEqual(result.loc["2024-01-06", marker], pd.Timestamp("2024-01-01"))
        self.assertTrue(pd.isna(result.loc["2024-01-07", marker]))

    def test_market_fill_honors_total_source_age_for_prices_and_market_caps(self):
        index = pd.date_range("2024-01-01", periods=10, freq="D")
        marker = data_format._source_observation_column("SPY_close")
        data = pd.DataFrame(
            {
                "price_close": [40_000.0] + [np.nan] * 9,
                "SPY_close": [100.0] * 6 + [np.nan] * 4,
                marker: [pd.Timestamp("2024-01-01")] * 6 + [pd.NaT] * 4,
                "AAPL_MarketCap": [3_000.0] + [np.nan] * 9,
                data_format.MINER_EFFICIENCY_VALUE_COLUMN: [0.03]
                + [np.nan] * 9,
                data_format.MINER_EFFICIENCY_SOURCE_DATE_COLUMN: [
                    pd.Timestamp("2024-01-01")
                ]
                + [pd.NaT] * 9,
                data_format.MINER_EFFICIENCY_SOURCE_URL_COLUMN: ["sheet-url"]
                + [np.nan] * 9,
            },
            index=index,
        )

        result = data_format.forward_fill_market_data(
            data, market_max_age_days=5, miner_max_age_days=8
        )

        self.assertTrue(pd.isna(result.loc["2024-01-07", "SPY_close"]))
        self.assertNotIn(marker, result.columns)
        self.assertEqual(result.loc["2024-01-06", "AAPL_MarketCap"], 3_000.0)
        self.assertTrue(pd.isna(result.loc["2024-01-07", "AAPL_MarketCap"]))
        self.assertTrue(pd.isna(result.loc["2024-01-02", "price_close"]))
        self.assertEqual(
            result.loc["2024-01-09", data_format.MINER_EFFICIENCY_VALUE_COLUMN],
            0.03,
        )
        self.assertTrue(
            pd.isna(
                result.loc[
                    "2024-01-10", data_format.MINER_EFFICIENCY_VALUE_COLUMN
                ]
            )
        )

    def test_market_freshness_reports_true_source_age(self):
        index = pd.date_range("2024-01-01", periods=7, freq="D")
        marker = data_format._source_observation_column("SPY_close")
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
            issues = data_format.warn_on_stale_market_data(
                data, "2024-01-07", max_age_days=5
            )
        self.assertTrue(any(issue.startswith("SPY_close") for issue in issues))

    def test_bitcoin_dominance_retains_api_observation_date(self):
        coindata = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-09", "2024-01-10"]),
                "price_close": [42_000.0, 43_000.0],
            }
        )
        dominance = pd.DataFrame(
            {
                "bitcoin_dominance": [51.0],
                "time": pd.to_datetime(["2024-01-01"]),
            }
        )
        empty = pd.DataFrame(columns=["time"])

        with patch.object(data_format, "get_brk_onchain", return_value=coindata), \
            patch.object(data_format, "get_price", return_value=empty), \
            patch.object(data_format, "get_marketcap", return_value=empty), \
            patch.object(data_format, "get_fear_and_greed_index", return_value=empty), \
            patch.object(data_format, "get_miner_data", return_value=empty), \
            patch.object(data_format, "get_bitcoin_dominance", return_value=dominance), \
            patch.object(data_format, "get_btc_trade_volume_14d", return_value=empty), \
            patch.object(data_format, "get_crypto_data", return_value=empty):
            result = data_format.get_data({"stocks": [], "crypto": []}, "2024-01-01")

        marker = data_format._source_observation_column("bitcoin_dominance")
        self.assertEqual(result.loc["2024-01-10", marker], pd.Timestamp("2024-01-01"))
        with self.assertWarnsRegex(
            RuntimeWarning,
            r"bitcoin_dominance \(source 2024-01-01, 9 days old\)",
        ):
            issues = data_format.warn_on_stale_market_data(
                result, "2024-01-10", max_age_days=5
            )
        self.assertTrue(
            any(issue.startswith("bitcoin_dominance") for issue in issues)
        )
        filled = data_format.forward_fill_market_data(
            result, market_max_age_days=5
        )
        self.assertTrue(pd.isna(filled.loc["2024-01-10", "bitcoin_dominance"]))

    def test_bitcoin_dominance_history_uses_completed_report_date_once(self):
        first_snapshot = pd.DataFrame(
            {
                "bitcoin_dominance": [51.25],
                "time": [pd.Timestamp("2024-01-10T00:30:00Z")],
            }
        )
        later_snapshot = pd.DataFrame(
            {
                "bitcoin_dominance": [59.0],
                "time": [pd.Timestamp("2024-01-10T03:00:00Z")],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bitcoin_dominance_history.csv"
            first = data_format.update_bitcoin_dominance_history(
                first_snapshot, "2024-01-09", path
            )
            rerun = data_format.update_bitcoin_dominance_history(
                later_snapshot, "2024-01-09", path
            )
            written = pd.read_csv(path)

        self.assertEqual(first.loc[0, "date"], pd.Timestamp("2024-01-09"))
        self.assertEqual(first.loc[0, "bitcoin_dominance"], 51.25)
        self.assertEqual(rerun.loc[0, "bitcoin_dominance"], 51.25)
        self.assertEqual(written.loc[0, "date"], "2024-01-09")
        self.assertEqual(written.loc[0, "bitcoin_dominance"], 51.25)

    def test_bitcoin_dominance_history_rejects_a_late_first_capture(self):
        late_snapshot = pd.DataFrame(
            {
                "bitcoin_dominance": [51.25],
                "time": [pd.Timestamp("2024-01-10T12:00:00Z")],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bitcoin_dominance_history.csv"
            with self.assertRaisesRegex(RuntimeError, "not captured near"):
                data_format.update_bitcoin_dominance_history(
                    late_snapshot, "2024-01-09", path
                )
            self.assertFalse(path.exists())

    def test_get_data_merges_report_date_dominance_history(self):
        coindata = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-09", "2024-01-10"]),
                "price_close": [42_000.0, 43_000.0],
            }
        )
        dominance = pd.DataFrame(
            {
                "bitcoin_dominance": [51.25],
                "time": [pd.Timestamp("2024-01-10T00:30:00Z")],
            }
        )
        empty = pd.DataFrame(columns=["time"])

        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "bitcoin_dominance_history.csv"
            with patch.object(data_format, "get_brk_onchain", return_value=coindata), \
                patch.object(data_format, "get_price", return_value=empty), \
                patch.object(data_format, "get_marketcap", return_value=empty), \
                patch.object(data_format, "get_fear_and_greed_index", return_value=empty), \
                patch.object(data_format, "get_miner_data", return_value=empty), \
                patch.object(data_format, "get_bitcoin_dominance", return_value=dominance), \
                patch.object(data_format, "get_btc_trade_volume_14d", return_value=empty), \
                patch.object(data_format, "get_crypto_data", return_value=empty):
                result = data_format.get_data(
                    {"stocks": [], "crypto": []},
                    "2024-01-01",
                    report_date="2024-01-09",
                    bitcoin_dominance_history_path=history_path,
                )

            written = pd.read_csv(history_path)

        marker = data_format._source_observation_column("bitcoin_dominance")
        self.assertEqual(result.loc["2024-01-09", "bitcoin_dominance"], 51.25)
        self.assertEqual(result.loc["2024-01-09", marker], pd.Timestamp("2024-01-09"))
        self.assertEqual(written.loc[0, "date"], "2024-01-09")
        issues = data_format.warn_on_stale_market_data(result, "2024-01-09")
        self.assertFalse(any(issue.startswith("bitcoin_dominance") for issue in issues))
        filled = data_format.forward_fill_market_data(result)
        self.assertEqual(filled.loc["2024-01-09", "bitcoin_dominance"], 51.25)

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
            data_format.requests, "get", return_value=response
        ) as get_mock:
            result = data_format.get_miner_data(sheet_url).set_index("time")

        get_mock.assert_called_once_with(export_url, timeout=data_format.API_TIMEOUT)
        self.assertEqual(
            result.loc["2024-01-31", data_format.MINER_EFFICIENCY_VALUE_COLUMN],
            0.03,
        )
        self.assertEqual(
            result.loc[
                "2024-01-31", data_format.MINER_EFFICIENCY_SOURCE_DATE_COLUMN
            ],
            pd.Timestamp("2024-01-01"),
        )
        self.assertEqual(
            result.loc[
                "2024-01-31", data_format.MINER_EFFICIENCY_SOURCE_URL_COLUMN
            ],
            export_url,
        )

    def test_stale_miner_value_is_retained_with_its_true_source_date(self):
        index = pd.date_range("2024-01-01", "2024-03-05", freq="D")
        source_date = pd.Timestamp("2024-01-01")
        data = pd.DataFrame(
            {
                data_format.MINER_EFFICIENCY_VALUE_COLUMN: [0.03] * len(index),
                data_format.MINER_EFFICIENCY_SOURCE_DATE_COLUMN: [source_date]
                * len(index),
                data_format.MINER_EFFICIENCY_SOURCE_URL_COLUMN: ["sheet-url"]
                * len(index),
            },
            index=index,
        )

        with self.assertWarnsRegex(
            RuntimeWarning, "source observation 2024-01-01, which is 64 days old"
        ):
            issues = data_format.warn_on_stale_miner_efficiency(
                data, "2024-03-05", max_age_days=62
            )
        self.assertEqual(len(issues), 1)
        self.assertEqual(
            data.loc["2024-03-05", data_format.MINER_EFFICIENCY_VALUE_COLUMN],
            0.03,
        )

    def test_default_miner_fill_carries_last_available_estimate(self):
        index = pd.date_range("2024-01-01", "2024-05-01", freq="D")
        data = pd.DataFrame(
            {
                data_format.MINER_EFFICIENCY_VALUE_COLUMN: [0.03]
                + [np.nan] * (len(index) - 1),
                data_format.MINER_EFFICIENCY_SOURCE_DATE_COLUMN: [
                    pd.Timestamp("2024-01-01")
                ]
                + [pd.NaT] * (len(index) - 1),
                data_format.MINER_EFFICIENCY_SOURCE_URL_COLUMN: ["sheet-url"]
                + [np.nan] * (len(index) - 1),
            },
            index=index,
        )

        result = data_format.forward_fill_market_data(data)

        self.assertEqual(
            result.loc["2024-05-01", data_format.MINER_EFFICIENCY_VALUE_COLUMN],
            0.03,
        )
        self.assertEqual(
            result.loc[
                "2024-05-01", data_format.MINER_EFFICIENCY_SOURCE_DATE_COLUMN
            ],
            pd.Timestamp("2024-01-01"),
        )

    def test_rolling_correlations_do_not_pad_stale_prices(self):
        index = pd.date_range("2024-01-01", periods=4, freq="D")
        prices = pd.DataFrame(
            {
                "price_close": [100.0, 101.0, 102.0, 103.0],
                "SPY_close": [100.0, np.nan, 120.0, 132.0],
            },
            index=index,
        )

        correlation = data_format.calculate_rolling_correlations(prices, [2])[2]

        self.assertTrue(
            pd.isna(correlation.loc[index[-1]].loc["price_close", "SPY_close"])
        )


if __name__ == "__main__":
    unittest.main()
