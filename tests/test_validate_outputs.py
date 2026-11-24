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
    def test_price_level_dates_are_required_only_for_technical_lines(self):
        from data_definitions import PRICE_OUTLOOK_LEVELS, PRICE_OUTLOOK_YEAR

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            table = PRICE_OUTLOOK_LEVELS.assign(outlook_year=PRICE_OUTLOOK_YEAR)
            rules = {"price_outlook.csv": RowBounds(1, 100)}
            table.to_csv(directory / "price_outlook.csv", index=False)
            self.assertEqual(validate_outputs(directory, "2026-10-18", rules=rules), [])
            index = table.index[table.type.eq("support")][0]
            for value in ["", "2026-02-30", "2026-12-01", "2026-6-30"]:
                with self.subTest(value=value):
                    invalid = table.copy()
                    invalid.loc[index, "start_date"] = value
                    invalid.to_csv(directory / "price_outlook.csv", index=False)
                    self.assertTrue(any("start dates" in error for error in
                        validate_outputs(directory, "2026-10-18", rules=rules)))

    def _performance_frame(self):
        rows = [
            {"Category": category, "Asset": label, "Price": 10.0,
             **{column: 1.0 for column in report_tables.PERFORMANCE_RETURN_FIELDS}}
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

        # A 4-year return is unavailable before an asset has 4 years of history.
        young = self._performance_frame()
        young.loc[0, "4 Year Return (%)"] = np.nan
        validator._validate_performance_rows({"performance_table.csv": young}, errors)
        self.assertEqual(errors, [])
        young["4 Year Return (%)"] = young["4 Year Return (%)"].astype(object)
        young.loc[0, "4 Year Return (%)"] = "invalid"
        validator._validate_performance_rows({"performance_table.csv": young}, errors)
        self.assertIn("invalid long-horizon returns", errors[-1])
        errors = []

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
            heatmap_row.update({"Year": 2024, "Feb": -5.0, "Yearly": 10.0})
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
            # The averages as the master release computes them (metrics.py): the
            # validator's calendar windows must agree with them.
            for column, days in validator.PRICE_MOVING_AVERAGE_DAYS.items():
                frame[column] = frame["BTC Price"].rolling(window=days).mean()
            path = directory / "onchain_price_models.csv"
            frame.to_csv(path)

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
            parameters = {'power_law_exponent': 5.6, 'power_law_scale': 1e-17, 'metcalfe_scale': 3e-4}
            write_release_manifest(output, '2024-03-02', model_parameters=parameters)
            errors = []
            _validate_release_manifest(output, pd.Timestamp('2024-03-02'), errors, True)
            self.assertEqual(errors, [])
            (output / CANDLE_FILES[-1]).unlink()
            write_release_manifest(output, '2024-03-02', model_parameters=parameters)
            _validate_release_manifest(output, pd.Timestamp('2024-03-02'), errors, True)
            self.assertTrue(any('inventory' in error for error in errors))


class MasterCutoffValidationTests(unittest.TestCase):
    """Large dated exports are asserted to end on the report date."""

    def write(self, tmpdir, name, last_date):
        frame = pd.DataFrame(
            {
                "date": pd.date_range(end=last_date, periods=5, freq="D"),
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
                "date",
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
                "date",
                pd.Timestamp("2026-08-27"),
                errors,
            )
            self.assertEqual(errors, [])

    def test_master_export_cutoff_is_covered(self):

        self.assertIn(
            "master_metrics_data.csv.gz", validator.INDEX_CUTOFF_OUTPUTS
        )

    def test_wholly_empty_master_column_is_rejected(self):
        # 2026-10-18: a dropped GC=F download published every gold column empty.
        rules = {"master_metrics_data.csv.gz": RowBounds(1, None)}
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            path = self.write(tmpdir, "master_metrics_data.csv.gz", "2026-10-17")
            frame = pd.read_csv(path)
            frame["SI=F_close"] = 30.0
            frame["GC=F_close"] = np.nan
            frame.to_csv(path, index=False)

            errors = validate_outputs(tmpdir, "2026-10-17", rules=rules)

            self.assertIn(
                "master_metrics_data.csv.gz: columns with no values ['GC=F_close']", errors
            )

            frame.loc[0, "GC=F_close"] = 2_000.0
            frame.to_csv(path, index=False)
            errors = validate_outputs(tmpdir, "2026-10-17", rules=rules)

            self.assertFalse([e for e in errors if "columns with no values" in e])


class ReleaseSourceAgreementTests(unittest.TestCase):
    def test_release_rejects_an_inconsistent_candle(self):
        summary = {f'{prefix}_{column}':[value] for prefix in ('daily','week_to_date')
                   for column,value in zip(('open','high','low','close'), (100,110,90,105))}
        summary.update({'week_start':['2026-09-07'], 'week_to_date_days':[2]})
        frames = {'report_ohlc_summary.csv':pd.DataFrame(summary)}
        frames['report_ohlc_summary.csv']['daily_high'] = 1
        errors=[]
        validator._validate_review_contracts(frames, Path('/nonexistent'), pd.Timestamp('2026-09-08'), errors)
        self.assertTrue(any('report_ohlc_summary.csv' in error for error in errors))

    def test_release_rejects_missing_day_and_mutated_fundamental(self):
        from data_definitions import FUNDAMENTALS_TEMPLATE
        columns = {item[0] for group in FUNDAMENTALS_TEMPLATE.values() for item in group.values()}
        dates = pd.date_range('2024-01-01', periods=400)
        master = pd.DataFrame(10.0, index=dates, columns=sorted(columns))
        master.index.name = 'date'
        fundamentals = report_tables.create_fundamentals_table(master, FUNDAMENTALS_TEMPLATE, dates[-1])
        fundamentals.loc[0,'Current Value'] = '999999999'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            master.drop(index=dates[10]).to_csv(output/'master_metrics_data.csv.gz')
            errors=[]
            validator._validate_index_cutoff(output, 'master_metrics_data.csv.gz', 'date', dates[-1], errors)
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


class StrictInvariantTests(unittest.TestCase):
    def test_relative_value_comparison_rejects_missing_or_non_positive_values(self):
        good = pd.DataFrame(
            [{"Asset": "Bitcoin", "Market Cap (USD)": 1.5e12, "BTC Price at Market Cap": 75_000.0}]
        )
        errors = []
        validator._validate_relative_value_comparison({"relative_value_comparison.csv": good}, errors)
        self.assertEqual(errors, [])

        bad = pd.DataFrame(
            [
                {"Asset": "Bitcoin", "Market Cap (USD)": 1.5e12, "BTC Price at Market Cap": 75_000.0},
                {"Asset": "Saudi Aramco", "Market Cap (USD)": np.nan, "BTC Price at Market Cap": np.nan},
            ]
        )
        errors = []
        validator._validate_relative_value_comparison({"relative_value_comparison.csv": bad}, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("Saudi Aramco", errors[0])

    def test_stock_market_cap_guards_reject_15t_cap_and_200pct_single_day_jump(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        clean = pd.DataFrame({"AAPL_market_cap": [3.0e12, 3.1e12, 3.05e12]}, index=dates)
        errors = []
        validator._validate_stock_market_caps(clean, errors)
        self.assertEqual(errors, [])

        bad = pd.DataFrame(
            {
                "MSFT_market_cap": [0.3e12, 16.0e12, 16.0e12],
                "AAPL_market_cap": [1.0e12, 4.0e12, 4.0e12],
            },
            index=dates,
        )
        errors = []
        validator._validate_stock_market_caps(bad, errors)
        self.assertTrue(any("$15T cap guard" in e for e in errors))
        self.assertTrue(any(">200% single-day jump" in e for e in errors))

    def test_etf_invariants_reject_large_other_change_and_low_aum_coverage(self):
        daily = pd.DataFrame(
            [
                {"fund": "IBIT", "date": "2026-10-16", "btc_held": 800_000.0, "btc_source": "exact",
                 "flow_btc": -892.0, "flow_usd": -60e6, "other_change_btc": 9_020.0, "market_share_pct": 80.0},
                {"fund": "FBTC", "date": "2026-10-16", "btc_held": 200_000.0, "btc_source": "carried forward",
                 "flow_btc": 0.0, "flow_usd": 0.0, "other_change_btc": 0.0, "market_share_pct": 20.0},
            ]
        )
        totals = pd.DataFrame(
            [{"date": "2026-10-16", "total_btc": 1_000_000.0, "total_flow_btc": -892.0,
              "cumulative_flow_usd": 50e9, "flow_weighted_entry_price": 60_000.0}]
        )
        errors = []
        validator._validate_etf({"etf_daily.csv": daily, "etf_totals_daily.csv": totals}, errors)
        self.assertTrue(any("other_change_btc" in e for e in errors))

        low_cov_daily = daily.copy()
        low_cov_daily.loc[0, "other_change_btc"] = -5.0
        low_cov_daily.loc[0, "btc_source"] = "carried forward"
        errors = []
        validator._validate_etf({"etf_daily.csv": low_cov_daily, "etf_totals_daily.csv": totals}, errors)
        self.assertTrue(any("AUM coverage" in e for e in errors))

    def test_master_metrics_is_loaded_once_across_all_sub_validators(self):
        from data_definitions import FUNDAMENTALS_TEMPLATE

        dates = pd.date_range("2024-01-01", periods=370, freq="D")
        cols = {item[0] for group in FUNDAMENTALS_TEMPLATE.values() for item in group.values()}
        cols |= {"price_close", "market_cap", "metcalfe_value", "power_law_price",
                 "60_day_ma_hash_rate", "hash_ribbon_capitulation", "price_close_4y_cagr",
                 "nupl", "supply_in_profit", "supply", "power_law_price_multiple"}
        master = pd.DataFrame(10.0, index=dates, columns=sorted(cols))
        master["nupl"] = 0.3
        master["power_law_price_multiple"] = 0.9
        master["supply_in_profit"] = 7.0
        master["hash_ribbon_capitulation"] = False
        master.index.name = "date"
        fundamentals = report_tables.create_fundamentals_table(master, FUNDAMENTALS_TEMPLATE, dates[-1])
        ranks = report_tables.create_relative_value_rank_changes(master, dates[-1])

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            master.to_csv(directory / "master_metrics_data.csv.gz", compression="gzip")
            fundamentals.to_csv(directory / "fundamentals_table.csv", index=False)
            ranks.to_csv(directory / report_tables.RELATIVE_RANK_FILE, index=False)
            rules = {
                "master_metrics_data.csv.gz": RowBounds(1, 10_000),
                "fundamentals_table.csv": RowBounds(1, 1_000),
                report_tables.RELATIVE_RANK_FILE: RowBounds(1, 10_000),
            }
            real_read_csv = pd.read_csv
            master_reads = []

            def counting_read_csv(path, *args, **kwargs):
                if str(path).endswith("master_metrics_data.csv.gz"):
                    master_reads.append(str(path))
                return real_read_csv(path, *args, **kwargs)

            with patch.object(validator.pd, "read_csv", side_effect=counting_read_csv):
                errors = validate_outputs(directory, dates[-1], rules=rules)

        self.assertEqual(errors, [])
        self.assertEqual(len(master_reads), 1)


if __name__ == "__main__":
    unittest.main()
