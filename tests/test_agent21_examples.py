"""Landing-page example facts (agent21_examples.py)."""

import json
import tempfile
import unittest

import pandas as pd

import agent21_examples as ex

REPORT_DATE = "2026-10-07"


def cost_basis_frame(price, realized=50.0, sth=80.0, multiple_3=150.0, start="2021-01-01"):
    dates = pd.date_range(start, REPORT_DATE, freq="D")
    return pd.DataFrame(
        {
            "price_close": price(dates) if callable(price) else price,
            "realized_price": realized,
            "sth_realized_price": sth,
            "realizedcap_multiple_3": multiple_3,
        },
        index=dates,
    )


def levels():
    return pd.DataFrame(
        [
            {"label": "Bull Case", "price": 160, "type": "case"},
            {"label": "Base Case", "price": 120, "type": "case"},
            {"label": "Bear Case", "price": 70, "type": "case"},
            {"label": "Resistance $100 - Psychological Level", "price": 100, "type": "resistance"},
            {"label": "Resistance $95 - Old High", "price": 95, "type": "resistance"},
            {"label": "Resistance $80 - Below Price", "price": 80, "type": "resistance"},
            {"label": "Support $74 - 2024 Prior ATH", "price": 74, "type": "support"},
            {"label": "Support $60 - Year Low", "price": 60, "type": "support"},
        ]
    )


def weekly_candles(dates):
    weeks = pd.date_range(dates[0], dates[-1], freq="W-MON")
    return pd.DataFrame(
        {
            "interval": "weekly",
            "period_start": weeks,
            "period_end": weeks + pd.Timedelta(days=6),
            "Open": 1.0,
            "High": 2.0,
            "Low": 0.5,
            "Close": 1.5,
            "complete": True,
        }
    )


class CostBasisFactsTests(unittest.TestCase):
    def test_side_since_is_the_first_close_of_the_current_run(self):
        frame = cost_basis_frame(lambda d: [70.0 if day < pd.Timestamp("2026-08-19") else 90.0 for day in d])
        facts = ex.cost_basis_facts(frame, REPORT_DATE)
        self.assertEqual((facts["sth_side"], facts["sth_side_since"]), ("above", "2026-08-19"))
        self.assertEqual((facts["realized_side"], facts["realized_side_since"]), ("above", "2021-01-01"))

    def test_price_below_a_level_reports_below(self):
        frame = cost_basis_frame(lambda d: [90.0 if day < pd.Timestamp("2026-09-01") else 70.0 for day in d])
        facts = ex.cost_basis_facts(frame, REPORT_DATE)
        self.assertEqual((facts["sth_side"], facts["sth_side_since"]), ("below", "2026-09-01"))

    def test_no_close_above_3x_or_below_realized_gives_none(self):
        facts = ex.cost_basis_facts(cost_basis_frame(90.0), REPORT_DATE)
        self.assertIsNone(facts["last_close_above_3x"])
        self.assertIsNone(facts["last_stretch_below_realized"])

    def test_last_close_above_3x_is_the_latest_one(self):
        frame = cost_basis_frame(90.0)
        frame.loc["2021-04-15":"2021-04-17", "price_close"] = 200.0
        self.assertEqual(ex.cost_basis_facts(frame, REPORT_DATE)["last_close_above_3x"], "2021-04-17")

    def test_stretch_counts_closes_below_realized_in_the_year_to_the_last_one(self):
        frame = cost_basis_frame(90.0)
        frame.loc["2022-06-13":"2022-06-22", "price_close"] = 40.0  # 10 days
        frame.loc["2023-01-03":"2023-01-12", "price_close"] = 40.0  # 10 days
        frame.loc["2021-03-01":"2021-03-05", "price_close"] = 40.0  # older than a year before
        stretch = ex.cost_basis_facts(frame, REPORT_DATE)["last_stretch_below_realized"]
        self.assertEqual(stretch, {"start": "2022-06-13", "end": "2023-01-12", "days": 20})

    def test_chart_covers_five_years_weekly_ending_at_the_report_date(self):
        chart = ex.cost_basis_facts(cost_basis_frame(90.0, start="2015-01-01"), REPORT_DATE)["chart"]
        self.assertEqual(chart["dates"][-1], REPORT_DATE)
        self.assertGreaterEqual(chart["dates"][0], "2021-10-08")
        gaps = pd.to_datetime(pd.Series(chart["dates"])).diff().dropna().dt.days
        self.assertTrue((gaps == 7).all())
        self.assertEqual(len(chart["price"]), len(chart["dates"]))

    def test_series_must_reach_the_report_date(self):
        frame = cost_basis_frame(90.0).loc[:"2026-10-06"]
        with self.assertRaises(ValueError):
            ex.cost_basis_facts(frame, REPORT_DATE)


class OutlookFactsTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", REPORT_DATE, freq="D")
        self.data = pd.DataFrame({"price_close": 83.0}, index=dates)
        self.data.loc["2025-12-31", "price_close"] = 87.0
        self.data.loc["2025-10-08", "price_close"] = 123.0
        self.data.loc["2026-06-30", "price_close"] = 58.0
        self.data.loc["2025-01-15", "price_close"] = 200.0  # more than a year before
        self.candles = weekly_candles(dates)

    def facts(self):
        return ex.outlook_facts(self.data, self.candles, levels(), 2026, REPORT_DATE)

    def test_nearest_levels_skip_those_on_the_wrong_side_of_price(self):
        facts = self.facts()
        self.assertEqual(facts["nearest_resistance"], {"price": 95.0, "name": "Old High"})
        self.assertEqual(facts["nearest_support"], {"price": 74.0, "name": "2024 Prior ATH"})

    def test_year_figures(self):
        facts = self.facts()
        self.assertEqual(facts["previous_year_close"], 87.0)
        self.assertEqual(facts["days_left_in_year"], 85)
        self.assertEqual(facts["year_high"], {"date": "2025-10-08", "close": 123.0})
        self.assertEqual(facts["year_low"], {"date": "2026-06-30", "close": 58.0})
        self.assertEqual([c["name"] for c in facts["cases"]], ["Bull Case", "Base Case", "Bear Case"])

    def test_weekly_candles_cover_the_last_52_weeks(self):
        candles = self.facts()["weekly_candles"]
        self.assertIn(len(candles), (52, 53))
        self.assertGreater(candles[0]["start"], "2025-10-01")

    def test_no_level_on_one_side_gives_none(self):
        self.data.loc[REPORT_DATE, "price_close"] = 500.0
        self.assertIsNone(self.facts()["nearest_resistance"])


class WriteTests(unittest.TestCase):
    def test_written_file_is_compact_sorted_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = ex.write_agent21_examples({"report_date": REPORT_DATE, "schema_version": 1}, directory)
            text = path.read_text(encoding="utf-8")
        self.assertEqual(path.name, ex.AGENT21_EXAMPLES_FILE)
        self.assertEqual(text, '{"report_date":"2026-10-07","schema_version":1}\n')
        self.assertEqual(json.loads(text)["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
