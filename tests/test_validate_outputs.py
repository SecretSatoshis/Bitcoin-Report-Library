"""Release validation (validate_outputs.py)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import report_tables
import validate_outputs as validator
from candle_data import CANDLE_FILES
from validate_outputs import (
    REQUIRED_COLUMNS,
    RowBounds,
    SUMMARY_HISTORY_METRICS,
    validate_outputs,
)


class OutputValidationTests(unittest.TestCase):
    def _performance_frame(self):
        rows = [
            {"Category": category, "Asset": label, "Price": 10.0, "7 Day Return (%)": 1.0,
             "MTD Return (%)": 1.0, "YTD Return (%)": 1.0, "90 Day Return (%)": 1.0}
            for category, assets in report_tables.PERFORMANCE_GROUPS.items()
            for label, _ in assets
        ]
        return pd.DataFrame(rows)

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

    def test_validator_rejects_performance_rows_without_prices(self):

        errors = []
        validator._validate_performance_rows({"performance_table.csv": self._performance_frame()}, errors)
        self.assertEqual(errors, [])

        outage = self._performance_frame()
        outage.loc[outage["Category"] == "Sectors", "Price"] = float("nan")
        validator._validate_performance_rows({"performance_table.csv": outage}, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing price or returns for Technology Sector ETF - [XLK]", errors[0])

        errors = []
        validator._validate_performance_rows(
            {"performance_table.csv": self._performance_frame().iloc[:-1]}, errors)
        self.assertIn("rows do not match", errors[0])

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
            "performance_table.csv": RowBounds(1, 20),
            "mtd_return_comparison.csv": RowBounds(1, 10),
            "ytd_return_comparison.csv": RowBounds(1, 10),
            "monthly_heatmap_data.csv": RowBounds(1, 10),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            performance = self._performance_frame()
            performance.loc[0, ["Price", "MTD Return (%)", "YTD Return (%)"]] = [100.0, -5.0, 10.0]
            performance.to_csv(directory / "performance_table.csv", index=False)

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


class CandleTests(unittest.TestCase):
    def test_release_manifest_accepts_complete_candle_bundle_only(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from release_manifest import write_release_manifest
        from validate_outputs import _validate_release_manifest
        with TemporaryDirectory() as directory, patch('validate_outputs.OUTPUT_RULES', {}):
            output = Path(directory)
            for filename in CANDLE_FILES:
                (output / filename).write_bytes(b'fixture')
            write_release_manifest(output, '2024-03-02')
            errors = []
            _validate_release_manifest(output, pd.Timestamp('2024-03-02'), errors, True)
            self.assertEqual(errors, [])
            (output / CANDLE_FILES[-1]).unlink()
            write_release_manifest(output, '2024-03-02')
            _validate_release_manifest(output, pd.Timestamp('2024-03-02'), errors, True)
            self.assertTrue(any('inventory' in error for error in errors))


class MasterCutoffValidationTests(unittest.TestCase):
    """Large dated exports are asserted to end on the report date."""

    def write(self, tmpdir, name, last_date):
        frame = pd.DataFrame(
            {
                "time": pd.date_range(end=last_date, periods=5, freq="D"),
                "value": 1.0,
            }
        )
        path = tmpdir / name
        frame.to_csv(path, index=False)
        return path

    def test_partial_day_in_master_is_reported(self):
        import tempfile
        from pathlib import Path


        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            self.write(tmpdir, "master_metrics_data.csv.gz", "2026-08-28")
            errors = []
            validator._validate_index_cutoff(
                tmpdir,
                "master_metrics_data.csv.gz",
                "time",
                pd.Timestamp("2026-08-27"),
                errors,
            )
            self.assertEqual(len(errors), 1)
            self.assertIn("2026-08-28", errors[0])
            self.assertIn("expected 2026-08-27", errors[0])

    def test_truncated_master_passes(self):
        import tempfile
        from pathlib import Path


        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            self.write(tmpdir, "master_metrics_data.csv.gz", "2026-08-27")
            errors = []
            validator._validate_index_cutoff(
                tmpdir,
                "master_metrics_data.csv.gz",
                "time",
                pd.Timestamp("2026-08-27"),
                errors,
            )
            self.assertEqual(errors, [])

    def test_master_export_cutoff_is_covered(self):

        self.assertIn(
            "master_metrics_data.csv.gz", validator.INDEX_CUTOFF_OUTPUTS
        )


class ReleaseSourceAgreementTests(unittest.TestCase):
    def test_release_rejects_an_inconsistent_candle(self):
        summary = {f'{prefix} {column}':[value] for prefix in ('Daily','Week-to-Date')
                   for column,value in zip(('Open','High','Low','Close'), (100,110,90,105))}
        summary.update({'Week Start':['2026-09-07'], 'Week-to-Date Days':[2]})
        frames = {'report_ohlc_summary.csv':pd.DataFrame(summary)}
        frames['report_ohlc_summary.csv']['Daily High'] = 1
        errors=[]
        validator._validate_review_contracts(frames, Path('/nonexistent'), pd.Timestamp('2026-09-08'), errors)
        self.assertTrue(any('report_ohlc_summary.csv' in error for error in errors))

    def test_release_rejects_missing_day_and_mutated_fundamental(self):
        from data_definitions import FUNDAMENTALS_TEMPLATE
        columns = {item[0] for group in FUNDAMENTALS_TEMPLATE.values() for item in group.values()}
        dates = pd.date_range('2024-01-01', periods=400)
        master = pd.DataFrame(10.0, index=dates, columns=sorted(columns))
        master.index.name = 'time'
        fundamentals = report_tables.create_fundamentals_table(master, FUNDAMENTALS_TEMPLATE, dates[-1])
        fundamentals.loc[0,'Current Value'] = '999999999'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            master.drop(index=dates[10]).to_csv(output/'master_metrics_data.csv.gz')
            errors=[]
            validator._validate_index_cutoff(output, 'master_metrics_data.csv.gz', 'time', dates[-1], errors)
            validator._validate_review_contracts({'fundamentals_table.csv': fundamentals}, output, dates[-1], errors)
        self.assertTrue(any('complete' in error for error in errors))
        self.assertTrue(any('fundamentals_table.csv' in error for error in errors))


class ManifestReportDateTests(unittest.TestCase):
    def test_validator_reads_the_report_date_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'release_manifest.json'
            path.write_text(json.dumps({'report_date': '2026-09-26'}))
            # Generation crossed UTC midnight: the clock now says 09-27.
            self.assertEqual(validator._manifest_report_date(directory, '2026-09-27'), ('2026-09-26', None))
            date, error = validator._manifest_report_date(directory, '2026-09-29')
            self.assertIsNone(date)
            self.assertIn('not a current release', error)
            path.unlink()
            self.assertIsNone(validator._manifest_report_date(directory, '2026-09-27')[0])


if __name__ == "__main__":
    unittest.main()
