"""Regression tests for report cutoffs and publication validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import report_tables
from validate_outputs import (
    REQUIRED_COLUMNS,
    RowBounds,
    SUMMARY_HISTORY_METRICS,
    validate_outputs,
)


class AsOfReportTests(unittest.TestCase):
    def test_performance_table_uses_one_resolved_asof_row(self):
        dates = pd.to_datetime(["2023-01-06", "2024-01-05", "2024-01-11"])
        report_data = pd.DataFrame(
            {
                "price_close": [50.0, 100.0, 999.0],
                "price_close_7_change": [1.0, 2.0, 999.0],
                "price_close_MTD_change": [3.0, 4.0, 999.0],
                "price_close_YTD_change": [5.0, 6.0, 999.0],
                "price_close_90_change": [7.0, 8.0, 999.0],
            },
            index=dates,
        )

        result = report_tables._build_performance_table(
            report_data=report_data,
            report_date="2024-01-10",
            correlation_results={},
            asset_configs=[
                {"name": "BTC", "label": "Bitcoin - [BTC]", "ticker": "price_close"}
            ],
            category="Test",
        ).iloc[0]

        self.assertEqual(result["Price"], 100.0)
        self.assertEqual(result["7 Day Return (%)"], 2.0)
        self.assertEqual(result["MTD Return (%)"], 4.0)
        self.assertEqual(result["YTD Return (%)"], 6.0)
        self.assertEqual(result["90 Day Return (%)"], 8.0)
        self.assertEqual(result["52 Week High"], 100.0)
        self.assertEqual(result["52 Week Low"], 50.0)



class OutputValidationTests(unittest.TestCase):
    def _write_summary_history(self, directory: Path, end_date: str) -> Path:
        dates = pd.date_range(end=pd.Timestamp(end_date), periods=31, freq="D")
        rows = [
            {"Metric": metric, "date": date.strftime("%Y-%m-%d"), "Value": 100.0}
            for metric in sorted(SUMMARY_HISTORY_METRICS)
            for date in dates
        ]
        path = directory / "summary_history.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_validator_accepts_complete_summary_window_and_rejects_infinity(self):
        rules = {"summary_history.csv": RowBounds(31, 1_000)}
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = self._write_summary_history(directory, "2024-02-15")

            self.assertEqual(
                validate_outputs(directory, "2024-02-15", rules=rules), []
            )

            frame = pd.read_csv(path)
            frame.loc[0, "Value"] = np.inf
            frame.to_csv(path, index=False)
            errors = validate_outputs(directory, "2024-02-15", rules=rules)

            self.assertTrue(any("infinity" in error for error in errors))

    def test_validator_rejects_wrong_report_date_and_header_only_output(self):
        summary_rules = {"summary_history.csv": RowBounds(31, 1_000)}
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            self._write_summary_history(directory, "2024-02-16")
            errors = validate_outputs(
                directory, "2024-02-15", rules=summary_rules
            )
            self.assertTrue(any("spans" in error for error in errors))

            empty = directory / "price_outlook.csv"
            pd.DataFrame(
                columns=sorted(REQUIRED_COLUMNS["price_outlook.csv"])
            ).to_csv(empty, index=False)
            errors = validate_outputs(
                directory,
                "2024-02-15",
                rules={"price_outlook.csv": RowBounds(1, 10)},
            )
            self.assertTrue(any("below minimum" in error for error in errors))

    def test_validator_rejects_a_missing_required_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            errors = validate_outputs(
                temp_dir,
                "2024-02-15",
                rules={"summary_table.csv": RowBounds(1, 100)},
            )
        self.assertEqual(
            errors, ["summary_table.csv: required output is missing"]
        )

    def test_validator_cross_checks_btc_mtd_and_ytd_returns(self):
        rules = {
            "performance_table.csv": RowBounds(1, 10),
            "mtd_return_comparison.csv": RowBounds(1, 10),
            "ytd_return_comparison.csv": RowBounds(1, 10),
            "monthly_heatmap_data.csv": RowBounds(1, 10),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            pd.DataFrame(
                [
                    {
                        "Category": "Bitcoin",
                        "Asset": "Bitcoin - [BTC]",
                        "Price": 100.0,
                        "MTD Return (%)": -5.0,
                        "YTD Return (%)": 10.0,
                    }
                ]
            ).to_csv(directory / "performance_table.csv", index=False)

            for period, value in (("mtd", -5.0), ("ytd", 10.0)):
                pd.DataFrame(
                    [
                        {
                            "Year": 2024,
                            "End Price ($)": 100.0,
                            "Return (%)": value,
                            "Report Date Return (%)": value,
                        }
                    ]
                ).to_csv(
                    directory / f"{period}_return_comparison.csv", index=False
                )

            heatmap_row = {month: np.nan for month in (
                "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                "Aug", "Sep", "Oct", "Nov", "Dec",
            )}
            heatmap_row.update({"time": 2024, "Feb": -5.0, "Yearly": 10.0})
            heatmap_path = directory / "monthly_heatmap_data.csv"
            pd.DataFrame([heatmap_row]).to_csv(heatmap_path, index=False)

            self.assertEqual(
                validate_outputs(directory, "2024-02-15", rules=rules), []
            )

            heatmap = pd.read_csv(heatmap_path)
            heatmap.loc[0, "Feb"] = -4.0
            heatmap.to_csv(heatmap_path, index=False)
            errors = validate_outputs(directory, "2024-02-15", rules=rules)

            self.assertTrue(any("BTC MTD return" in error for error in errors))

    def test_validator_enforces_cycle_and_halving_anchors(self):
        rules = {
            "cycle_low_data.csv": RowBounds(1, 10),
            "halving_data.csv": RowBounds(1, 10),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            pd.DataFrame(
                {
                    "days_since_cycle_low": [0, 1],
                    "index_value": [1.0, 1.2],
                    "Cycle": ["Market Cycle 1", "Market Cycle 1"],
                }
            ).to_csv(directory / "cycle_low_data.csv", index=False)
            pd.DataFrame(
                {
                    "days_since_halving": [0, 1],
                    "index_value": [1.0, 1.1],
                    "Era": ["2nd Era", "2nd Era"],
                }
            ).to_csv(directory / "halving_data.csv", index=False)

            self.assertEqual(
                validate_outputs(directory, "2024-02-15", rules=rules), []
            )

            cycle = pd.read_csv(directory / "cycle_low_data.csv")
            cycle.loc[1, "index_value"] = 0.9
            cycle.to_csv(directory / "cycle_low_data.csv", index=False)
            halving = pd.read_csv(directory / "halving_data.csv")
            halving["Era"] = "Genesis Era"
            halving.to_csv(directory / "halving_data.csv", index=False)

            errors = validate_outputs(directory, "2024-02-15", rules=rules)
            self.assertTrue(any("falls below" in error for error in errors))
            self.assertTrue(any("Genesis Era" in error for error in errors))

    def test_validator_recomputes_price_moving_averages(self):
        rules = {"onchain_price_models.csv": RowBounds(1, 5_000)}
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            dates = pd.date_range(end="2024-02-15", periods=1_500, freq="D")
            frame = pd.DataFrame(
                {
                    "BTC Price": np.linspace(100.0, 1_600.0, len(dates)),
                    "Electricity Cost": 1.0,
                    "Metcalfe Value": 1.0,
                    "Power Law Price": 1.0,
                },
                index=pd.Index(dates, name="date"),
            )
            path = directory / "onchain_price_models.csv"
            report_tables.add_price_moving_averages(frame).to_csv(path)

            self.assertEqual(
                validate_outputs(directory, "2024-02-15", rules=rules), []
            )

            published = pd.read_csv(path)
            published.loc[len(published) - 1, "200-day MA"] += 1.0
            published.to_csv(path, index=False)
            errors = validate_outputs(directory, "2024-02-15", rules=rules)
            self.assertTrue(any("200-day MA" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
