"""Pairwise return alignment and the published grouped matrix contract."""

import unittest
import numpy as np
import pandas as pd

import metrics
import report_tables
from validate_outputs import _validate_correlation_matrix


class CorrelationMatrixTests(unittest.TestCase):
    def test_pairs_use_shared_trading_dates_and_ignore_future_prices(self):
        dates = pd.date_range("2024-01-01", periods=111)
        rng = np.random.default_rng(21)
        btc = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.03, len(dates))), index=dates)
        trading = dates[dates.dayofweek < 5]
        prices = pd.DataFrame({"price_close": btc, "SPY_close": btc.loc[trading] / 10})
        qqq_returns = rng.normal(0, 0.02, len(trading))
        prices["QQQ_close"] = pd.Series(50 * np.cumprod(1 + qqq_returns), index=trading)
        cutoff = dates[-2]
        # An extreme later observation cannot affect the report-date snapshot.
        prices.loc[dates[-1], :] = [1e9, 1, 1e9]
        columns = list(prices.columns)
        matrix = metrics.create_correlation_matrix_data(cutoff, columns, prices)
        self.assertAlmostEqual(matrix.loc["price_close", "SPY_close"], 1)
        window = trading[(trading > cutoff - pd.Timedelta(days=90)) & (trading <= cutoff)]
        prior = trading[trading <= cutoff - pd.Timedelta(days=90)][-1]
        observed = prices.loc[pd.DatetimeIndex([prior]).append(window)]
        spy_returns = np.diff(observed.SPY_close) / observed.SPY_close.to_numpy()[:-1]
        qqq_returns = np.diff(observed.QQQ_close) / observed.QQQ_close.to_numpy()[:-1]
        expected = np.corrcoef(spy_returns, qqq_returns)[0, 1]
        self.assertAlmostEqual(matrix.loc["SPY_close", "QQQ_close"], expected)
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 1)

    def test_unavailable_histories_are_blank_including_the_diagonal(self):
        dates = pd.date_range("2024-01-01", periods=380)
        prices = pd.DataFrame({"OK": np.linspace(100, 180, len(dates)), "OLD": np.linspace(20, 60, len(dates)),
                               "NEW": np.linspace(30, 60, len(dates)), "CONSTANT": 10}, index=dates)
        prices.loc[dates[-8]:, "OLD"] = np.nan
        prices.loc[:dates[-10], "NEW"] = np.nan
        result = metrics.create_correlation_matrix_data(dates[-1], [*prices, "MISSING"], prices)
        for column in ["OLD", "NEW", "CONSTANT", "MISSING"]:
            self.assertTrue(result[column].isna().all(), column)
        self.assertAlmostEqual(result.loc["OK", "OK"], 1)

    def test_export_has_all_17_assets_in_both_axes_and_validator_rejects_bad_values(self):
        dates = pd.date_range("2024-01-01", periods=380)
        columns = [c for _, _, _, c in report_tables.CORRELATION_ASSETS]
        prices = pd.DataFrame({c: np.linspace(100 + i, 180 + i * 2, len(dates))
                               for i, c in enumerate(columns)}, index=dates)
        matrices = report_tables.create_correlation_matrices(prices, dates[-1])
        table = report_tables.create_correlation_matrix_table(matrices, dates[-1])
        self.assertEqual(table.shape, (51, 22))
        self.assertEqual(set(table["Window Days"]), {30, 90, 365})
        window = table.loc[table["Window Days"].eq(90)]
        self.assertEqual(list(window.Ticker), list(table.columns[5:]))
        self.assertEqual(window.groupby("Category", sort=False).size().tolist(), [1, 4, 4, 4, 4])
        errors = []
        _validate_correlation_matrix({"correlation_matrix.csv": table}, dates[-1], errors)
        self.assertEqual(errors, [])
        table.loc[0, "SPY"] = -0.5
        _validate_correlation_matrix({"correlation_matrix.csv": table}, dates[-1], errors)
        self.assertTrue(any("symmetric" in e for e in errors))
        table.loc[0, "BTC"] = 1.2
        _validate_correlation_matrix({"correlation_matrix.csv": table}, dates[-1], errors)
        self.assertTrue(any("outside" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
