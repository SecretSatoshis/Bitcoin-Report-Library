"""Published report tables (report_tables.py)."""

import unittest

import numpy as np
import pandas as pd

import data_validation
import report_tables


class PerformanceTableTests(unittest.TestCase):
    def test_performance_table_uses_one_resolved_asof_row(self):
        dates = pd.to_datetime(["2023-01-06", "2024-01-05", "2024-01-11"])
        report_data = pd.DataFrame(
            {
                "price_close": [50.0, 100.0, 999.0],
                "price_close_7d_change": [1.0, 2.0, 999.0],
                "price_close_mtd_change": [3.0, 4.0, 999.0],
                "price_close_ytd_change": [5.0, 6.0, 999.0],
                "price_close_90d_change": [7.0, 8.0, 999.0],
            },
            index=dates,
        )

        result = report_tables._build_performance_table(
            report_data=report_data,
            report_date="2024-01-10",
            correlation_results={},
            asset_groups={"Bitcoin": [("Bitcoin - [BTC]", "price_close")]},
        ).iloc[0]

        self.assertEqual(result["Price"], 100.0)
        self.assertEqual(result["7 Day Return (%)"], 2.0)
        self.assertEqual(result["MTD Return (%)"], 4.0)
        self.assertEqual(result["YTD Return (%)"], 6.0)
        self.assertEqual(result["90 Day Return (%)"], 8.0)
        self.assertEqual(result["52 Week High"], 100.0)
        self.assertEqual(result["52 Week Low"], 50.0)

    def test_performance_table_lists_bitcoin_once_in_its_own_category(self):
        tickers = ["price_close"] + [
            ticker for assets in report_tables.PERFORMANCE_GROUPS.values()
            for _, ticker in assets if ticker != "price_close"
        ]
        columns = {}
        for ticker in tickers:
            price = ticker if ticker == "price_close" else f"{ticker}_close"
            columns[price] = [10.0]
            for suffix in ("7d", "mtd", "ytd", "90d"):
                columns[f"{price}_{suffix}_change"] = [1.0]
        report_data = pd.DataFrame(columns, index=pd.to_datetime(["2024-01-05"]))
        correlations = {"price_close_90_days": pd.DataFrame(
            0.5, index=["price_close"], columns=[f"{t}_close" for t in tickers[1:]])}

        result = report_tables.create_full_performance_table(
            report_data, "2024-01-05", correlations)

        bitcoin = result[result["Asset"] == "Bitcoin - [BTC]"]
        self.assertEqual(len(bitcoin), 1)
        self.assertEqual(bitcoin["Category"].iloc[0], "Bitcoin")
        self.assertEqual(result["Asset"].iloc[0], "Bitcoin - [BTC]")
        self.assertEqual(len(result), 17)
        self.assertEqual(
            result.groupby("Category", sort=False).size().to_dict(),
            {"Bitcoin": 1, "Equity Market Indexes": 4, "Sectors": 4,
             "Macro Asset Classes": 4, "Bitcoin Industry Performance": 4},
        )


class OhlcTableTests(unittest.TestCase):
    def test_ohlc_tables_refuse_empty_candles(self):
        empty = pd.DataFrame(columns=data_validation.OHLC_COLUMNS)
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            report_tables.create_report_ohlc_summary(empty, "2024-01-01")


class PeriodReturnBoundaryTests(unittest.TestCase):
    @staticmethod
    def comparison_prices():
        return pd.DataFrame(
            {
                "price_close": [
                    80.0,
                    100.0,
                    110.0,
                    120.0,
                    140.0,
                    160.0,
                    200.0,
                    220.0,
                    240.0,
                    999.0,
                ]
            },
            index=pd.to_datetime(
                [
                    "2019-12-31",
                    "2020-01-31",
                    "2020-02-01",
                    "2020-02-02",
                    "2020-02-29",
                    "2020-12-31",
                    "2021-01-31",
                    "2021-02-01",
                    "2021-02-02",
                    "2021-02-03",
                ]
            ),
        )

    def test_heatmap_month_and_year_use_prior_calendar_closes(self):
        data = pd.DataFrame(
            {
                "price_close": [80.0, 0.0, 100.0, 120.0, 110.0, 132.0],
            },
            index=pd.to_datetime(
                [
                    "2022-12-30",
                    "2022-12-31",
                    "2023-01-01",
                    "2023-01-31",
                    "2023-02-01",
                    "2023-02-15",
                ]
            ),
        )

        result = report_tables.monthly_heatmap(
            data, report_date="2023-02-15"
        )

        self.assertEqual(result.index.name, "Year")
        self.assertAlmostEqual(result.loc[2023, "Jan"], 50.0)
        self.assertAlmostEqual(result.loc[2023, "Feb"], 10.0)
        self.assertAlmostEqual(result.loc[2023, "Yearly"], 65.0)

    def test_period_table_without_current_data_raises(self):
        prices = pd.DataFrame(
            {"price_close": [100.0, 110.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"])
        )
        with self.assertRaisesRegex(ValueError, "No MTD price history for 2021-02-02"):
            report_tables.create_period_returns_table(prices, "2021-02-02", "mtd")

    def test_monthly_and_yearly_comparisons_share_boundary_semantics(self):
        prices = self.comparison_prices()

        monthly = report_tables.create_period_returns_table(
            prices, "2021-02-02", "mtd"
        ).set_index("Year")
        yearly = report_tables.create_period_returns_table(
            prices, "2021-02-02", "ytd"
        ).set_index("Year")

        self.assertEqual(monthly.loc[2021, "Start Price ($)"], 200.0)
        self.assertEqual(monthly.loc[2021, "End Price ($)"], 240.0)
        self.assertAlmostEqual(monthly.loc[2021, "Return (%)"], 20.0)
        self.assertAlmostEqual(monthly.loc[2021, "Report Date Return (%)"], 20.0)
        self.assertAlmostEqual(
            monthly.loc["Median Projection", "Return (%)"], 40.0
        )

        self.assertEqual(yearly.loc[2021, "Start Price ($)"], 160.0)
        self.assertEqual(yearly.loc[2021, "End Price ($)"], 240.0)
        self.assertAlmostEqual(yearly.loc[2021, "Return (%)"], 50.0)
        self.assertAlmostEqual(yearly.loc[2021, "Report Date Return (%)"], 50.0)
        self.assertAlmostEqual(
            yearly.loc["Median Projection", "Return (%)"], 100.0
        )

    def test_indexed_histories_include_shared_prior_close_anchor(self):
        prices = self.comparison_prices()["price_close"]

        monthly = report_tables.create_price_paths(
            prices, report_date="2021-02-02", period="mtd", min_year=2020
        )
        yearly = report_tables.create_price_paths(
            prices, report_date="2021-02-02", period="ytd", min_year=2020
        )

        self.assertEqual(monthly.loc[0, "2020"], 200.0)
        self.assertEqual(monthly.loc[0, "2021"], 200.0)
        self.assertAlmostEqual(monthly.loc[1, "2020"], 220.0)
        self.assertAlmostEqual(monthly.loc[1, "2021"], 220.0)
        self.assertAlmostEqual(monthly.loc[2, "2021"], 240.0)
        self.assertEqual(monthly["2021"].last_valid_index(), 2)

        self.assertEqual(yearly.loc[0, "2020"], 160.0)
        self.assertEqual(yearly.loc[0, "2021"], 160.0)
        self.assertAlmostEqual(yearly.loc[31, "2020"], 200.0)
        self.assertAlmostEqual(yearly.loc[31, "2021"], 200.0)
        self.assertAlmostEqual(yearly.loc[33, "2021"], 240.0)


class ReturnAndSummaryTableTests(unittest.TestCase):
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

        result = report_tables.create_price_paths(
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
        sentiment = result[result["Category"] == "Investor Sentiment"]

        self.assertEqual(list(sentiment.index), [
            "Bitcoin Supply in Profit (%)", "Bitcoin Market Sentiment", "Bitcoin Valuation",
        ])
        self.assertEqual(sentiment.loc["Bitcoin Supply in Profit (%)", "Value"], 70.0)
        self.assertEqual(sentiment.loc["Bitcoin Market Sentiment", "Label"], "Belief / Denial")
        self.assertEqual(sentiment.loc["Bitcoin Valuation", "Label"], "Below Fair Value")
        # Value is numeric throughout; text lives only in Label.
        self.assertTrue(pd.api.types.is_float_dtype(result["Value"]))
        self.assertTrue(result.loc[["Bitcoin Market Sentiment", "Bitcoin Valuation"], "Value"].isna().all())

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


class CalendarBoundaryTableTests(unittest.TestCase):
    def test_missing_week_candle_cannot_replace_output(self):
        candles = pd.DataFrame([[100,110,90,105]]*3, columns=data_validation.OHLC_COLUMNS,
                               index=pd.to_datetime(['2024-01-01','2024-01-03','2024-01-04']))
        with self.assertRaisesRegex(ValueError, 'missing a day'):
            report_tables.create_report_ohlc_summary(candles, '2024-01-04')

    def test_fundamentals_exact_dates_and_calendar_range(self):
        dates = pd.date_range('2023-01-01', periods=400)
        frame = pd.DataFrame({'metric': 10.0}, index=dates)
        template = {'Section': {'Metric': ('metric','number')}}
        frame.iloc[0] = 9999
        frame.loc[dates[-1]-pd.Timedelta(days=7), 'metric'] = np.nan
        result = report_tables.create_fundamentals_table(frame, template, dates[-1]).iloc[0]
        self.assertTrue(pd.isna(result['7 Day Change (%)']))
        self.assertEqual(result['52W High'], '10')
        frame.iloc[-1] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing on'):
            report_tables.create_fundamentals_table(frame, template, dates[-1])

    def test_completed_december_year_included_in_average(self):
        frame = pd.DataFrame({'price_close':[100,200,800]}, index=pd.to_datetime(['2022-12-31','2023-12-31','2024-12-31']))
        result = report_tables.monthly_heatmap(frame)
        self.assertAlmostEqual(float(result.loc['Average','Yearly']), 200.0)



class RelativeValueTableTests(unittest.TestCase):
    def test_fiat_money_supply_comes_from_the_reference_data_and_rows_sort_by_size(self):
        from data_definitions import FIAT_MONEY_SUPPLY

        us_m0 = float(FIAT_MONEY_SUPPLY.set_index("Country").loc["United States", "US Dollar Trillion"]) * 1e12
        row = {"price_close": 100_000.0, "market_cap": 2e12, "united_states_m0_btc_price": 250_000.0,
               "AAPL_market_cap_btc_price": 200_000.0, "AAPL_market_cap": 4e12}
        data = pd.DataFrame([row], index=pd.to_datetime(["2024-01-05"]))
        table = report_tables.create_asset_valuation_table(data, "2024-01-05").set_index("Asset")

        self.assertEqual(table.loc["US M0", "Market Cap (USD)"], us_m0)
        self.assertEqual(table.loc["US M0", "Move Needed (%)"], 150.0)
        self.assertEqual(table.loc["Bitcoin", "Move Needed (%)"], 0.0)
        caps = table["Market Cap (USD)"].dropna()
        self.assertTrue(caps.is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
