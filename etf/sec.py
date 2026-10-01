"""Quarter-end bitcoin holdings and cost basis for each fund from its SEC filings.

The filings tag the bitcoin's fair value and cost each quarter. The coin count comes from, in
order: a tagged count (CryptoAssetNumberOfUnits, or the investment balance in the schedule of
investments); a count written in the filing's text ("N bitcoin"), taken when it agrees with
fair value / accounting price to within 1%; otherwise fair value / accounting price. The
accounting price for a quarter end is the median fair value / coins of the funds with a tagged
count; quarters with none fall back to VanEck's 4pm index price. `coin_source` says which.
"""
from __future__ import annotations

import html
import re
import time

import pandas as pd

from .common import get

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
# The SEC asks automated clients to identify themselves.
SEC_HEADERS = {"User-Agent": "Secret Satoshis Bitcoin-Report-Library treybrunson@protonmail.com"}
FAIR_VALUE_TAGS = ("CryptoAssetFairValue", "InvestmentOwnedAtFairValue",
                   "InvestmentInPhysicalCommoditiesFairValueDisclosure")
COST_TAGS = ("CryptoAssetCost", "InvestmentOwnedAtCost")
UNITS_TAGS = ("CryptoAssetNumberOfUnits", "InvestmentOwnedBalanceContracts", "InvestmentOwnedBalanceShares")
TEXT_COUNT = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)\s*(?:bitcoins?|BTC)\b", re.IGNORECASE)
TEXT_MATCH_TOLERANCE = 0.01


def _quarter_values(facts: dict, tags) -> dict[str, float]:
    """Point-in-time values from 10-Q/10-K filings by period end.

    Funds switched tags over time, so all listed tags are merged with the first taking
    precedence; within a tag the latest filing for a date wins.
    """
    values: dict[str, float] = {}
    for tag in reversed(tags):
        observations = [o for unit in facts.get(tag, {}).get("units", {}).values() for o in unit
                        if o.get("form") in ("10-Q", "10-K") and "start" not in o]
        for o in sorted(observations, key=lambda o: o.get("filed", "")):
            values[o["end"]] = float(o["val"])
    return values


def _text_counts(cik: int, cache: dict) -> list[float]:
    """Every number written as "<n> bitcoin" in the fund's 10-Q and 10-K filings.

    Each filing is read once; `cache` maps accession numbers to the counts found and gains
    an entry for every filing read.
    """
    recent = get(SUBMISSIONS_URL.format(cik=cik), headers=SEC_HEADERS).json()["filings"]["recent"]
    counts = []
    for form, accession, document in zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"]):
        if form not in ("10-Q", "10-K"):
            continue
        if accession not in cache:
            page = get(ARCHIVE_URL.format(cik=cik, accession=accession.replace("-", ""), document=document),
                       headers=SEC_HEADERS).text
            text = html.unescape(re.sub(r"<[^>]+>", " ", page))
            cache[accession] = sorted({float(m.group(1).replace(",", "")) for m in TEXT_COUNT.finditer(text)})
            time.sleep(0.15)
        counts.extend(cache[accession])
    return counts


def quarter_ends(ciks: dict[str, int], reference: pd.Series, cache: dict) -> pd.DataFrame:
    """One row per fund and quarter end: fair value, cost, coins and cost per coin.

    `cache` holds the text counts of filings already read (accession -> counts).
    """
    rows, text_counts = [], {}
    for fund, cik in ciks.items():
        facts = get(FACTS_URL.format(cik=cik), headers=SEC_HEADERS).json()["facts"].get("us-gaap", {})
        fair = _quarter_values(facts, FAIR_VALUE_TAGS)
        cost = _quarter_values(facts, COST_TAGS)
        units = _quarter_values(facts, UNITS_TAGS)
        for end, value in fair.items():
            rows.append({"fund": fund, "quarter_end": end, "fair_value_usd": value,
                         "cost_usd": cost.get(end), "reported_btc": units.get(end)})
        text_counts[fund] = _text_counts(cik, cache)
        time.sleep(0.15)
    frame = pd.DataFrame(rows)

    # Accounting price per quarter end from funds that tag both fair value and coins.
    tagged = frame[frame.reported_btc.fillna(0) > 100]
    implied = (tagged.fair_value_usd / tagged.reported_btc).groupby(tagged.quarter_end).median()
    frame["price_usd"] = frame.quarter_end.map(implied)
    frame["price_source"] = frame.price_usd.notna().map({True: "fair value / reported coins", False: ""})
    fallback = frame.price_usd.isna()
    frame.loc[fallback, "price_usd"] = frame.loc[fallback, "quarter_end"].map(reference)
    frame.loc[fallback & frame.price_usd.notna(), "price_source"] = "VanEck 4pm index (fallback)"

    estimate = frame.fair_value_usd / frame.price_usd
    frame["coin_source"] = frame.reported_btc.notna().map({True: "tagged", False: "fair value / price"})
    for i in frame.index[frame.reported_btc.isna() & (estimate > 0)]:
        candidates = [c for c in text_counts[frame.at[i, "fund"]]
                      if abs(c / estimate[i] - 1) <= TEXT_MATCH_TOLERANCE]
        if candidates:
            frame.at[i, "reported_btc"] = min(candidates, key=lambda c: abs(c / estimate[i] - 1))
            frame.at[i, "coin_source"] = "filing text"
    frame["btc"] = frame.reported_btc.fillna(estimate)
    frame["cost_per_btc"] = frame.cost_usd / frame.btc
    return frame.sort_values(["fund", "quarter_end"]).reset_index(drop=True)
