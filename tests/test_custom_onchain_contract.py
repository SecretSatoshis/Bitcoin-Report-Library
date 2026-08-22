import unittest

import numpy as np
import pandas as pd

import data_format


class CustomOnchainContractTests(unittest.TestCase):
    def source_frame(self) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=731, freq="D", tz="UTC")
        frame = pd.DataFrame(
            {
                "market_cap": 100.0,
                "supply": 20.0,
                "realized_cap": 40.0,
                "realized_price": 7.0,
                "price_close": 10.0,
                "transfer_volume_sum_24h_usd": 10.0,
                "coinbase_sum_24h_usd": 1.0,
                "utxos_over_1y_old_supply": 12.0,
                "coindays_destroyed_sum_24h": 2.0,
                "supply_in_profit": 15.0,
                "supply_in_loss": 5.0,
                "fees_sum_24h": 1.0,
                "coinbase_sum_24h": 5.0,
                "active_addrs_average_24h": 1000.0,
            },
            index=index,
        )
        frame.loc[index[1], "coinbase_sum_24h_usd"] = np.nan
        frame.loc[index[2], "realized_price"] = np.nan
        return frame

    def test_current_report_metric_names_and_formulas_remain_stable(self):
        result = data_format.calculate_custom_on_chain_metrics(self.source_frame())

        self.assertEqual(result["RevAllTimeUSD"].iloc[0], 1.0)
        self.assertEqual(result["RevAllTimeUSD"].iloc[1], 1.0)
        self.assertTrue(pd.isna(result["NVTAdj90"].iloc[88]))
        self.assertEqual(result["NVTAdj90"].iloc[89], 10.0)
        self.assertTrue(pd.isna(result["nvt_price"].iloc[728]))
        self.assertEqual(result["nvt_price"].iloc[729], 5.0)

        self.assertEqual(result["pct_supply_issued"].iloc[-1], 20.0 / 21_000_000)
        self.assertEqual(result["illiquid_supply"].iloc[-1], 12.0)
        self.assertEqual(result["liquid_supply"].iloc[-1], 8.0)
        self.assertIn("miner_revenue_1_Year", result)
        self.assertIn("miner_revenue_4_Year", result)

        self.assertEqual(result["realized_price"].iloc[0], 7.0)
        self.assertEqual(result["realized_price"].iloc[2], 2.0)


if __name__ == "__main__":
    unittest.main()
