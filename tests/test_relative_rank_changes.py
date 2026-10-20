"""Rolling and calendar ranking comparisons for the relative valuation dashboard."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from report_tables import (
    RELATIVE_RANK_FILE,
    asset_ranks_on_date,
    create_relative_value_rank_changes,
    relative_rank_baselines,
)
from validate_outputs import _validate_relative_rank_changes


class RelativeRankTests(unittest.TestCase):
    def frame(self, previous, current):
        return pd.DataFrame(
            [previous, current], index=pd.to_datetime(["2026-10-04", "2026-10-11"])
        )

    def weekly(self, frame):
        table = create_relative_value_rank_changes(frame, "2026-10-11")
        return table.loc[table.Timeframe.eq("wtd")].set_index("Asset")

    def test_up_down_unchanged_and_fixed_reference_values(self):
        frame = self.frame(
            {"market_cap": 30e12, "AAPL_market_cap": 20e12, "NVDA_market_cap": 10e12},
            {"market_cap": 15e12, "AAPL_market_cap": 25e12, "NVDA_market_cap": 10e12},
        )
        rows = self.weekly(frame)
        self.assertEqual(rows.loc["Apple", "Change"], 1)
        self.assertEqual(rows.loc["Bitcoin", "Change"], -1)
        self.assertEqual(rows.loc["NVIDIA", "Change"], 0)
        self.assertEqual(rows.loc["US M0", "Change"], 0)
        self.assertEqual(rows.loc["US M0", "Current Rank"], 4)
        self.assertEqual(rows.loc["Meta", "Status"], "missing_both")

    def test_new_entrant_shifts_existing_assets_and_has_no_baseline_rank(self):
        previous = {"market_cap": 30e12, "AAPL_market_cap": 20e12}
        rows = self.weekly(self.frame(previous, {**previous, "SPCX_market_cap": 35e12}))
        self.assertEqual(rows.loc["Bitcoin", "Change"], -1)
        self.assertEqual(rows.loc["Apple", "Change"], -1)
        self.assertEqual(rows.loc["SpaceX", "Current Rank"], 1)
        self.assertEqual(rows.loc["SpaceX", "Status"], "missing_baseline")
        self.assertTrue(pd.isna(rows.loc["SpaceX", "Change"]))

    def test_equal_caps_share_competition_rank(self):
        caps = {"market_cap": 20e12, "AAPL_market_cap": 20e12, "NVDA_market_cap": 10e12}
        frame = self.frame(caps, caps)
        ranks = asset_ranks_on_date(frame, "2026-10-11")
        self.assertEqual(ranks["Bitcoin"], 1)
        self.assertEqual(ranks["Apple"], 1)
        self.assertEqual(ranks["NVIDIA"], 3)
        rows = self.weekly(frame.iloc[::-1])
        self.assertTrue(rows.Change.dropna().eq(0).all())

    def test_invalid_caps_and_missing_current_data_are_unavailable(self):
        rows = self.weekly(self.frame(
            {"market_cap": 30e12, "AAPL_market_cap": 20e12},
            {"market_cap": np.inf, "AAPL_market_cap": -1, "NVDA_market_cap": 0},
        ))
        self.assertEqual(rows.loc["Bitcoin", "Status"], "missing_current")
        self.assertEqual(rows.loc["Apple", "Status"], "missing_current")
        self.assertEqual(rows.loc["NVIDIA", "Status"], "missing_both")
        self.assertTrue(pd.isna(rows.loc["Bitcoin", "Change"]))

    def test_missing_baseline_day_never_borrows_an_older_snapshot(self):
        frame = pd.DataFrame({"market_cap": [30e12, 30e12]},
                             index=pd.to_datetime(["2026-10-03", "2026-10-11"]))
        rows = self.weekly(frame)
        self.assertTrue(rows.Status.eq("missing_baseline").where(rows["Current Rank"].notna(), True).all())
        self.assertTrue(rows["Previous Rank"].isna().all())
        self.assertEqual(rows.loc["US M0", "Status"], "missing_baseline")
        self.assertTrue(asset_ranks_on_date(frame, "2026-10-12").empty)

    def test_calendar_boundaries_and_leap_year(self):
        cases = {
            "2026-10-12": ["2026-10-11", "2026-10-11", "2026-09-30", "2025-12-31", "2026-10-05"],
            "2026-10-11": ["2026-10-10", "2026-10-04", "2026-09-30", "2025-12-31", "2026-10-04"],
            "2026-10-01": ["2026-09-30", "2026-09-27", "2026-09-30", "2025-12-31", "2026-09-24"],
            "2024-01-01": ["2023-12-31"] * 4 + ["2023-12-25"],
            "2024-03-01": ["2024-02-29", "2024-02-25", "2023-12-31", "2023-12-31", "2024-02-23"],
            "2024-04-01": ["2024-03-31", "2024-03-31", "2024-03-31", "2023-12-31", "2024-03-25"],
        }
        for date, expected in cases.items():
            with self.subTest(date=date):
                self.assertEqual([str(d.date()) for d in relative_rank_baselines(date).values()][:5], expected)

    def test_rolling_windows_use_exact_days_including_leap_years(self):
        for date, expected in {
            "2024-03-01": ["2024-02-23", "2024-01-31", "2023-12-02", "2023-03-02"],
            "2026-10-12": ["2026-10-05", "2026-09-12", "2026-07-14", "2025-10-12"],
        }.items():
            with self.subTest(date=date):
                baselines = relative_rank_baselines(date)
                self.assertEqual([str(baselines[p].date()) for p in ("7d", "30d", "90d", "365d")], expected)
        frame = pd.DataFrame({
            "market_cap": [40e12, 20e12, 5e12, 30e12, 30e12],
            "AAPL_market_cap": [20e12, 30e12, 40e12, 20e12, 20e12],
        }, index=pd.to_datetime(["2025-10-12", "2026-07-14", "2026-09-12", "2026-10-05", "2026-10-12"]))
        result = create_relative_value_rank_changes(frame, "2026-10-12")
        bitcoin = result.loc[result.Asset.eq("Bitcoin")].set_index("Timeframe")
        self.assertEqual(bitcoin.loc[["7d", "30d", "90d", "365d"], "Change"].tolist(), [0, 2, 1, 0])

    def test_seven_day_change_does_not_reset_on_monday(self):
        frame = pd.DataFrame({
            "market_cap": [30e12, 10e12, 30e12],
            "AAPL_market_cap": [20e12, 40e12, 20e12],
        }, index=pd.to_datetime(["2026-10-05", "2026-10-11", "2026-10-12"]))
        table = create_relative_value_rank_changes(frame, "2026-10-12")
        bitcoin = table.loc[table.Asset.eq("Bitcoin")].set_index("Timeframe")
        self.assertEqual(bitcoin.loc["7d", "Baseline Date"], "2026-10-05")
        self.assertEqual(bitcoin.loc["7d", "Change"], 0)
        self.assertEqual(bitcoin.loc["wtd", "Change"], 1)
        missing = create_relative_value_rank_changes(frame.drop(pd.Timestamp("2026-10-05")), "2026-10-12")
        rolling = missing.loc[missing.Timeframe.eq("7d") & missing.Asset.eq("Bitcoin")].iloc[0]
        self.assertEqual(rolling.Status, "missing_baseline")
        self.assertTrue(pd.isna(rolling.Change))

    def test_release_validation_recomputes_dates_ranks_and_availability(self):
        frame = self.frame({"market_cap": 30e12}, {"market_cap": 20e12})
        table = create_relative_value_rank_changes(frame, "2026-10-11")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            frame.rename_axis("date").to_csv(directory / "master_metrics_data.csv.gz")
            table.to_csv(directory / RELATIVE_RANK_FILE, index=False)
            published = pd.read_csv(directory / RELATIVE_RANK_FILE)
            errors = []
            _validate_relative_rank_changes({RELATIVE_RANK_FILE: published}, directory,
                                            pd.Timestamp("2026-10-11"), errors)
            self.assertEqual(errors, [])
            for column, value in [("Change", 10), ("Current Rank", 2),
                                  ("Baseline Date", "2020-01-01"), ("Status", "available")]:
                with self.subTest(column=column):
                    bad = published.copy()
                    bad.loc[0, column] = value
                    errors = []
                    _validate_relative_rank_changes({RELATIVE_RANK_FILE: bad}, directory,
                                                    pd.Timestamp("2026-10-11"), errors)
                    self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
