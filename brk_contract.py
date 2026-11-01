"""BRK v0.13 wire series mapped to the Report Library's stable column contract.

Definitions and boundaries are pinned in docs/brk-v0.13-migration.md. Aggregates
require every constituent: missing data is never silently replaced by zero.
"""

from __future__ import annotations

import pandas as pd


DAILY_INDEX = "day1"
ALIASES = {
    "hash_rate": "hashrate",
    "supply": "circulating_supply",
    "addr_count": "funded_address_count",
    # This legacy output is a per-block mean of sending OR receiving addresses;
    # it is not the distinct number of sending addresses in a calendar day.
    "active_addrs_average_24h": "active_address_count_avg_24h",
}

# Half-open ranges, ordered youngest first. Months are 30 days; years 365 days.
# The extra 4m/5m and 6y/7y/8y/12y/15y bands must not be skipped or double counted.
AGE_BANDS = (
    "under_1h",
    "1h_to_1d",
    "1d_to_1w",
    "1w_to_1m",
    "1m_to_2m",
    "2m_to_3m",
    "3m_to_4m",
    "4m_to_5m",
    "5m_to_6m",
    "6m_to_9m",
    "9m_to_1y",
    "1y_to_18m",
    "18m_to_2y",
    "2y_to_3y",
    "3y_to_4y",
    "4y_to_5y",
    "5y_to_6y",
    "6y_to_7y",
    "7y_to_8y",
    "8y_to_10y",
    "10y_to_12y",
    "12y_to_15y",
    "over_15y",
)
AGE_CUTOFFS = {
    "1m": 4,
    "3m": 6,
    "6m": 9,
    "1y": 11,
    "2y": 13,
    "3y": 14,
    "4y": 15,
    "5y": 16,
    "10y": 20,
}
BALANCE_BANDS = (
    "1_to_10_sats",
    "10_to_100_sats",
    "100_to_1k_sats",
    "1k_to_10k_sats",
    "10k_to_100k_sats",
    "100k_to_1m_sats",
    "1m_to_10m_sats",
    "10m_to_100m_sats",
    "1_to_10_btc",
    "10_to_100_btc",
    "100_to_1k_btc",
    "1k_to_10k_btc",
    "10k_to_100k_btc",
    "over_100k_btc",
)
BALANCE_CUTOFFS = (
    "1sat",
    "10sats",
    "100sats",
    "1k_sats",
    "10k_sats",
    "100k_sats",
    "1m_sats",
    "10m_sats",
    "1btc",
    "10btc",
    "100btc",
    "1k_btc",
    "10k_btc",
)
SUMS = {
    **{
        f"utxos_under_{cutoff}_old_supply": tuple(
            f"age_{band}_supply" for band in AGE_BANDS[:end]
        )
        for cutoff, end in AGE_CUTOFFS.items()
    },
    "utxos_over_1y_old_supply": tuple(
        f"age_{band}_supply" for band in AGE_BANDS[AGE_CUTOFFS["1y"] :]
    ),
    **{
        f"utxos_over_{cutoff}_old_transfer_volume_cumulative": tuple(
            f"age_{band}_transfer_volume_cumulative"
            for band in AGE_BANDS[AGE_CUTOFFS[cutoff] :]
        )
        for cutoff in ("1y", "2y", "3y", "4y", "5y", "10y")
    },
    **{
        f"addrs_over_{cutoff}_addr_count": tuple(
            f"balance_{band}_address_count" for band in BALANCE_BANDS[start:]
        )
        for start, cutoff in enumerate(BALANCE_CUTOFFS)
    },
}
PUELL_INPUTS = ("subsidy_sum_24h_usd", "subsidy_sum_1y_usd")


def inputs_for(metric: str) -> tuple[str, ...]:
    if metric == "puell_multiple":
        return PUELL_INPUTS
    return SUMS.get(metric, (ALIASES.get(metric, metric),))


def request_series(metrics: list[str]) -> list[str]:
    """Deduplicate shared inputs while retaining request order and stable output names."""
    return list(
        dict.fromkeys(series for metric in metrics for series in inputs_for(metric))
    )


def restore_columns(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    result = {}
    for metric in metrics:
        if metric == "timestamp":
            continue
        names = inputs_for(metric)
        missing = [name for name in names if name not in frame]
        if missing:
            raise RuntimeError(f"BRK inputs missing for {metric}: {', '.join(missing)}")
        if metric == "puell_multiple":
            # The exact replacement formula in v0.13 Bitview Studio's presets.js.
            # v0.12.2 used a per-block ratio; this follows BRK's current definition.
            annual = frame["subsidy_sum_1y_usd"]
            result[metric] = (
                365 * frame["subsidy_sum_24h_usd"] / annual.where(annual > 0)
            )
        elif metric in SUMS:
            result[metric] = frame[list(names)].sum(axis=1, min_count=len(names))
        else:
            result[metric] = frame[names[0]]
    return pd.DataFrame(result, index=frame.index)
