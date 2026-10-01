"""Collect the US spot bitcoin ETFs' published data and build the ETF release files.

Each run reads every fund's current position, the daily histories and NAV histories the funds
publish, and their SEC quarter ends, adds them to what the previous release carried, and
returns the release files:

  etf_daily.csv          one row per fund and trading day (see etf.build)
  etf_totals_daily.csv   one row per trading day across all funds
  etf_quarterly.csv      SEC quarter ends with reported cost, plus an all-funds row
  etf_snapshots.csv      every collected position, carried from release to release
  etf_nav_history.csv    NAV for funds that publish only that, carried likewise
  etf_filing_counts.csv  the "N bitcoin" counts read from each SEC filing, so each is read once

A reader that fails is logged and skipped; its fund is carried forward until the next good day.
Funds that publish no daily history keep the rows earlier releases published.
"""
import json
import warnings

import pandas as pd

from . import funds
from .build import build_tables
from .sec import quarter_ends

CARRIED_FILES = ("etf_snapshots.csv", "etf_nav_history.csv", "etf_filing_counts.csv")
TABLE_FILES = ("etf_daily.csv", "etf_totals_daily.csv", "etf_quarterly.csv")


def _merge(previous: pd.DataFrame | None, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Previous rows plus new ones, the newest winning for the same keys."""
    frame = new if previous is None or previous.empty else pd.concat([previous, new], ignore_index=True)
    return frame.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def collect(report_date, previous: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """The ETF release files through `report_date`, and a log of what each reader returned.

    `previous` holds the last release's files by name (empty on the first run).
    """
    log = {"funds": {}}

    def failed(fund, key, error):
        message = f"{type(error).__name__}: {str(error).strip().splitlines()[-1] if str(error).strip() else ''}"[:200]
        log["funds"].setdefault(fund, {})[key] = message
        warnings.warn(f"ETF {fund} {key} failed: {message}", RuntimeWarning, stacklevel=2)

    snapshots = []
    for fund, read in funds.CURRENT.items():
        try:
            snapshot = read()
            snapshots.append(snapshot.row())
            log["funds"][fund] = {"as_of": snapshot.as_of, "btc_held": snapshot.btc_held}
        except Exception as error:
            failed(fund, "current", error)
    snapshot_frame = _merge(previous.get("etf_snapshots.csv"), pd.DataFrame(snapshots), ["fund", "as_of"])

    histories = []
    for fund, read in funds.HISTORY.items():
        try:
            histories.append(read().assign(fund=fund))
        except Exception as error:
            failed(fund, "history", error)
    if not histories:
        raise RuntimeError("no fund history could be read")
    history = pd.concat(histories, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"])

    navs = []
    for fund, read in funds.NAV_HISTORY.items():
        try:
            navs.append(read().assign(fund=fund))
        except Exception as error:
            failed(fund, "nav_history", error)
    nav_frame = _merge(previous.get("etf_nav_history.csv"),
                       pd.concat(navs)[["fund", "date", "nav"]] if navs else pd.DataFrame(columns=["fund", "date", "nav"]),
                       ["fund", "date"])

    counts = previous.get("etf_filing_counts.csv")
    cache = {} if counts is None else {row.accession: json.loads(row.counts) for row in counts.itertuples()}
    # A 4pm index for quarter ends no fund reports coins for: VanEck's, else 21Shares'.
    priced = history[history.get("index_price", pd.Series(dtype=float)).notna()] if "index_price" in history else history.iloc[:0]
    reference = (priced.sort_values("fund", key=lambda f: f != "HODL").drop_duplicates("date")
                 .set_index("date")["index_price"] if len(priced) else pd.Series(dtype=float))
    try:
        sec = quarter_ends(funds.SEC_CIK, reference, cache)
        sec["quarter_end"] = pd.to_datetime(sec["quarter_end"])
    except Exception as error:
        # Keep the last release's quarter ends; new filings are picked up on the next good run.
        failed("SEC", "quarter_ends", error)
        sec = previous["etf_quarterly.csv"]
        sec = sec[sec.fund != "ALL"].assign(quarter_end=lambda f: pd.to_datetime(f["quarter_end"]))
    count_frame = pd.DataFrame({"accession": list(cache), "counts": [json.dumps(v) for v in cache.values()]})

    tables = build_tables({"histories": history, "snapshots": snapshot_frame, "sec": sec,
                           "nav_history": nav_frame.assign(date=pd.to_datetime(nav_frame["date"])),
                           "previous_daily": previous.get("etf_daily.csv")}, report_date)
    tables.update({"etf_snapshots.csv": snapshot_frame,
                   "etf_nav_history.csv": nav_frame.assign(date=pd.to_datetime(nav_frame["date"]).dt.strftime("%Y-%m-%d")),
                   "etf_filing_counts.csv": count_frame.sort_values("accession").reset_index(drop=True)})
    return tables, log
