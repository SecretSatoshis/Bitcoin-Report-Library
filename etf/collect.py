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
Funds that publish no daily history keep the rows earlier releases published. Days a fund has
not yet published take their flow from the provisional dataset (funds.PROVISIONAL_FLOWS).
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import warnings

import pandas as pd

from . import funds
from .build import UNPUBLISHED, FILLED, _trading_days_behind, build_tables
from .sec import quarter_ends

CARRIED_FILES = ("etf_snapshots.csv", "etf_nav_history.csv", "etf_filing_counts.csv")
TABLE_FILES = ("etf_daily.csv", "etf_totals_daily.csv", "etf_quarterly.csv")
ETF_HEALTH_FILE = "etf_health.json"
TOP5_STALE_WARN_DAYS = 5


def _merge(previous: pd.DataFrame | None, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Previous rows plus new ones, the newest winning for the same keys."""
    frame = new if previous is None or previous.empty else pd.concat([previous, new], ignore_index=True)
    return frame.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def _build_health(
    report_date,
    daily: pd.DataFrame,
    snapshot_frame: pd.DataFrame,
    log: dict,
) -> dict:
    """Build structured scraper health metadata and warn if any top-5 fund is stale >5 trading days."""
    through = pd.Timestamp(report_date)
    days = pd.DatetimeIndex(sorted(pd.to_datetime(daily["date"]).unique())) if not daily.empty else pd.DatetimeIndex([])
    days = days[days <= through]
    snap_dates = (
        snapshot_frame.groupby("fund")["as_of"].max().to_dict()
        if not snapshot_frame.empty and {"fund", "as_of"}.issubset(snapshot_frame.columns)
        else {}
    )
    fund_health: dict[str, dict] = {}
    top5_stale: list[dict] = []
    for fund in funds.CURRENT:
        errors = {k: v for k, v in log["funds"].get(fund, {}).items() if k in ("current", "history", "nav_history")}
        rows = daily[daily["fund"] == fund].copy() if not daily.empty else pd.DataFrame()
        if not rows.empty:
            rows["date"] = pd.to_datetime(rows["date"])
            rows = rows.sort_values("date").set_index("date")
            stale_days, last_own = _trading_days_behind(rows["btc_source"], days)
        else:
            stale_days, last_own = len(days), None
        last_snapshot = snap_dates.get(fund)
        entry = {
            "status": "error" if errors else ("stale" if stale_days > TOP5_STALE_WARN_DAYS else "ok"),
            "last_snapshot_date": str(last_snapshot) if last_snapshot is not None and pd.notna(last_snapshot) else None,
            "last_issuer_date": str(last_own.date()) if last_own is not None else None,
            "consecutive_stale_days": int(stale_days),
            "errors": errors,
        }
        fund_health[fund] = entry
        if fund in funds.TOP5_FUNDS and stale_days > TOP5_STALE_WARN_DAYS:
            top5_stale.append({"fund": fund, "consecutive_stale_days": int(stale_days), "last_issuer_date": entry["last_issuer_date"]})
            warnings.warn(
                f"ETF top-5 fund {fund} has been stale for {stale_days} trading days "
                f"(last issuer data: {entry['last_issuer_date'] or 'never'})",
                RuntimeWarning,
                stacklevel=2,
            )
    return {
        "schema_version": 1,
        "report_date": str(through.date()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "top5_stale_threshold_days": TOP5_STALE_WARN_DAYS,
        "top5_stale": top5_stale,
        "funds": fund_health,
        "other_errors": {k: v for k, v in log["funds"].items() if k not in funds.CURRENT},
    }


def collect(report_date, previous: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """The ETF release files through `report_date`, and a log/health summary of what each reader returned.

    `previous` holds the last release's files by name (empty on the first run).
    """
    funds.clear_caches()
    log = {"funds": {}}

    def failed(fund, key, error):
        message = f"{type(error).__name__}: {str(error).strip().splitlines()[-1] if str(error).strip() else ''}"[:200]
        log["funds"].setdefault(fund, {})[key] = message
        warnings.warn(f"ETF {fund} {key} failed: {message}", RuntimeWarning, stacklevel=2)

    def _run(reader):
        try:
            return reader(), None
        except Exception as error:
            return None, error

    with ThreadPoolExecutor(max_workers=6) as pool:
        current_futs = {fund: pool.submit(_run, read) for fund, read in funds.CURRENT.items()}
        history_futs = {fund: pool.submit(_run, read) for fund, read in funds.HISTORY.items()}
        nav_futs = {fund: pool.submit(_run, read) for fund, read in funds.NAV_HISTORY.items()}
        prov_fut = pool.submit(_run, funds.provisional_flows)

        snapshots = []
        for fund, fut in current_futs.items():
            snapshot, error = fut.result()
            if error is not None:
                failed(fund, "current", error)
            else:
                snapshots.append(snapshot.row())
                log["funds"].setdefault(fund, {}).update({"as_of": snapshot.as_of, "btc_held": snapshot.btc_held})

        histories = []
        for fund, fut in history_futs.items():
            frame, error = fut.result()
            if error is not None:
                failed(fund, "history", error)
            else:
                histories.append(frame.assign(fund=fund))

        navs = []
        for fund, fut in nav_futs.items():
            frame, error = fut.result()
            if error is not None:
                failed(fund, "nav_history", error)
            else:
                navs.append(frame.assign(fund=fund))

        provisional, prov_error = prov_fut.result()
        if prov_error is not None:
            failed("provisional", "flows", prov_error)
            provisional = None

    snapshot_frame = _merge(previous.get("etf_snapshots.csv"), pd.DataFrame(snapshots), ["fund", "as_of"])
    if not histories:
        raise RuntimeError("no fund history could be read")
    history = pd.concat(histories, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"])

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
        sec_previous = previous.get("etf_quarterly.csv")
        if sec_previous is None or sec_previous.empty:
            raise RuntimeError("SEC quarter_ends failed and no previous etf_quarterly.csv is available") from error
        sec = sec_previous[sec_previous.fund != "ALL"].assign(quarter_end=lambda f: pd.to_datetime(f["quarter_end"]))
    count_frame = pd.DataFrame({"accession": list(cache), "counts": [json.dumps(v) for v in cache.values()]})

    tables = build_tables({"histories": history, "snapshots": snapshot_frame, "sec": sec,
                           "nav_history": nav_frame.assign(date=pd.to_datetime(nav_frame["date"])),
                           "previous_daily": previous.get("etf_daily.csv"),
                           "provisional": provisional}, report_date)
    tables.update({"etf_snapshots.csv": snapshot_frame,
                   "etf_nav_history.csv": nav_frame.assign(date=pd.to_datetime(nav_frame["date"]).dt.strftime("%Y-%m-%d")),
                   "etf_filing_counts.csv": count_frame.sort_values("accession").reset_index(drop=True)})
    log["health"] = _build_health(report_date, tables["etf_daily.csv"], snapshot_frame, log)
    return tables, log
