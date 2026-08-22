"""Regression tests for calendar-period return boundaries."""

import unittest

import numpy as np
import pandas as pd

import data_format
import report_tables


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

        ytd = data_format.calculate_ytd_change(data)
        mtd = data_format.calculate_mtd_change(data)

        self.assertAlmostEqual(ytd.loc["2023-01-01", "a_YTD_change"], 25.0)
        self.assertAlmostEqual(ytd.loc["2023-01-01", "b_YTD_change"], 20.0)
        self.assertAlmostEqual(mtd.loc["2023-02-01", "a_MTD_change"], 20.0)
        self.assertAlmostEqual(mtd.loc["2023-02-01", "b_MTD_change"], 25.0)

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
            data, report_date="2023-02-15", export_csv=False
        )

        self.assertEqual(result.index.name, "time")
        self.assertAlmostEqual(result.loc[2023, "Jan"], 50.0)
        self.assertAlmostEqual(result.loc[2023, "Feb"], 10.0)
        self.assertAlmostEqual(result.loc[2023, "Yearly"], 65.0)

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

    def test_monthly_and_yearly_comparisons_share_boundary_semantics(self):
        prices = self.comparison_prices()

        monthly = report_tables.create_monthly_returns_table(
            prices, report_date="2021-02-02"
        ).set_index("Year")
        yearly = report_tables.create_yearly_returns_table(
            prices, report_date="2021-02-02"
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

        monthly = report_tables.create_indexed_returns_history(
            prices, report_date="2021-02-02", period="mtd", min_year=2020
        )
        yearly = report_tables.create_indexed_returns_history(
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


if __name__ == "__main__":
    unittest.main()
