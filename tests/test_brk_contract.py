"""Pinned upstream definitions, request parameters and numerical migration checks."""

import csv
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import brk_contract as contract
import sources
from data_definitions import BRK_METRICS
from tests.test_sources import FakeResponse


FIXTURES = Path(__file__).parent / "fixtures"


class BrkContractTests(unittest.TestCase):
    def test_default_bulk_request_uses_canonical_daily_index(self):
        with patch.object(
            sources.requests,
            "get",
            return_value=FakeResponse(text="timestamp,hashrate\n1793404800,1e21\n"),
        ) as get:
            header, rows = sources._brk_fetch_csv(["timestamp", "hashrate"])
        self.assertEqual(
            get.call_args.kwargs["params"],
            {
                "series": "timestamp,hashrate",
                "index": "day1",
                "start": 0,
                "format": "csv",
            },
        )
        self.assertEqual(header, ["timestamp", "hashrate"])
        self.assertEqual(len(rows), 1)

    def test_every_configured_wire_input_exists_and_supports_day1_in_official_snapshot(
        self,
    ):
        fixture = json.loads((FIXTURES / "brk-v0.13-contract.json").read_text())
        requested = contract.request_series(BRK_METRICS)
        self.assertEqual(len(requested), len(set(requested)))
        self.assertEqual(set(requested), set(fixture["series"]))
        for name in requested:
            self.assertIn("day1", fixture["series"][name]["indexes"], name)
        self.assertEqual(fixture["series"]["hashrate"]["kind"], "Hashrate")
        self.assertEqual(fixture["series"]["inflation_rate"]["kind"], "Percent")

    def test_age_aggregates_partition_exact_boundaries_and_do_not_skip_split_bands(
        self,
    ):
        frame = pd.DataFrame(
            {
                f"age_{band}_supply": [i + 1.0]
                for i, band in enumerate(contract.AGE_BANDS)
            }
        )
        selected = [
            "utxos_under_1m_old_supply",
            "utxos_under_3m_old_supply",
            "utxos_under_1y_old_supply",
            "utxos_over_1y_old_supply",
            "utxos_under_10y_old_supply",
        ]
        result = contract.restore_columns(frame, selected).iloc[0]
        self.assertEqual(result["utxos_under_1m_old_supply"], 1 + 2 + 3 + 4)
        self.assertEqual(result["utxos_under_3m_old_supply"], sum(range(1, 7)))
        self.assertEqual(result["utxos_under_10y_old_supply"], sum(range(1, 21)))
        self.assertEqual(
            result["utxos_under_1y_old_supply"] + result["utxos_over_1y_old_supply"],
            sum(range(1, 24)),
        )
        frame.loc[0, "age_1y_to_18m_supply"] = np.nan
        result = contract.restore_columns(frame, selected).iloc[0]
        self.assertTrue(pd.isna(result["utxos_over_1y_old_supply"]))
        self.assertEqual(result["utxos_under_1y_old_supply"], sum(range(1, 12)))

    def test_balance_thresholds_include_the_lower_edge_and_exclude_zero_balance(self):
        values = {
            f"balance_{band}_address_count": [i + 1]
            for i, band in enumerate(contract.BALANCE_BANDS)
        }
        values["balance_0_sats_address_count"] = [1_000_000]
        frame = pd.DataFrame(values)
        result = contract.restore_columns(
            frame,
            [
                "addrs_over_1sat_addr_count",
                "addrs_over_1btc_addr_count",
                "addrs_over_10k_btc_addr_count",
            ],
        ).iloc[0]
        self.assertEqual(result["addrs_over_1sat_addr_count"], sum(range(1, 15)))
        self.assertEqual(result["addrs_over_1btc_addr_count"], sum(range(9, 15)))
        self.assertEqual(result["addrs_over_10k_btc_addr_count"], 13 + 14)

    def test_old_coin_transfer_aggregates_remain_cumulative(self):
        frame = pd.DataFrame(
            {
                f"age_{band}_transfer_volume_cumulative": [i * 10, i * 10 + 2]
                for i, band in enumerate(contract.AGE_BANDS)
            }
        )
        result = contract.restore_columns(
            frame, ["utxos_over_10y_old_transfer_volume_cumulative"]
        )
        self.assertEqual(result.iloc[0, 0], 200 + 210 + 220)
        self.assertEqual(result.iloc[1, 0] - result.iloc[0, 0], 6)

    def test_puell_matches_current_brk_studio_formula_without_fees_or_zero_denominator(
        self,
    ):
        frame = pd.DataFrame(
            {
                "subsidy_sum_24h_usd": [2.0, 0.0, 2.0, np.nan],
                "subsidy_sum_1y_usd": [365.0, 365.0, 0.0, 365.0],
                "coinbase_sum_24h_usd": [1000.0] * 4,
            }
        )
        values = contract.restore_columns(frame, ["puell_multiple"])["puell_multiple"]
        self.assertEqual(list(values[:2]), [2.0, 0.0])
        self.assertTrue(values[2:].isna().all())

    def test_direct_aliases_preserve_units_and_activity_definition(self):
        frame = pd.DataFrame(
            {
                "hashrate": [1e21],
                "circulating_supply": [20e6],
                "funded_address_count": [57e6],
                "active_address_count_avg_24h": [7600.0],
                "sending_address_count_avg_24h": [3000.0],
                "inflation_rate": [0.82],
            }
        )
        out = contract.restore_columns(
            frame, [*contract.ALIASES, "inflation_rate"]
        ).iloc[0]
        self.assertEqual(out["hash_rate"], 1e21)
        self.assertEqual(out["supply"], 20e6)
        self.assertEqual(out["active_addrs_average_24h"], 7600.0)
        self.assertEqual(out["inflation_rate"], 0.82)
        with self.assertRaisesRegex(RuntimeError, "inputs missing"):
            contract.restore_columns(frame, ["utxos_under_3m_old_supply"])

    def test_real_upstream_csv_flows_through_fetch_and_keeps_the_published_schema(self):
        fixture = pd.read_csv(FIXTURES / "brk-v0.13-sample.csv", dtype=str).fillna("")
        requests = []

        def fetch(url, params, timeout):
            self.assertEqual(params["index"], "day1")
            columns = params["series"].split(",")
            requests.extend(columns)
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(fixture[columns].itertuples(index=False, name=None))
            return FakeResponse(text=output.getvalue())

        with patch.object(sources.requests, "get", side_effect=fetch):
            frame = sources.get_brk_onchain("2026-10-30")
        self.assertTrue(set(BRK_METRICS) - {"timestamp"} <= set(frame.columns))
        self.assertTrue(set(sources.BRK_DERIVED_SERIES) <= set(frame.columns))
        self.assertNotIn("dateindex", requests)
        self.assertEqual(frame["date"].iloc[0], pd.Timestamp("2026-10-30"))
        self.assertTrue(frame["hash_rate"].gt(0).all())
        self.assertTrue(frame["puell_multiple"].gt(0).all())


if __name__ == "__main__":
    unittest.main()
