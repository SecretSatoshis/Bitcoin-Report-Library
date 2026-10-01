"""One reader per US spot bitcoin ETF, each returning today's published position.

Funds that publish their past also have a history reader returning daily rows of
date, shares_outstanding, nav and net_assets, plus index_price where the fund publishes the
bitcoin price its NAV uses (HODL, ARKB). Funds that publish only their past NAV have a NAV
reader returning date and nav.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import tempfile
import zipfile
from functools import cache
from pathlib import Path

import pandas as pd
import requests

from .common import (
    CHROME_HEADERS, Snapshot, excel_date, get, iso_date, json_after, number, post_json,
    spreadsheet2003_rows, xlsx_rows,
)

HERE = Path(__file__).resolve().parent

IBIT_HOLDINGS = "https://www.ishares.com/us/products/333011/ishares-bitcoin-trust-etf/latest-holdings.csv"
IBIT_DOWNLOAD = ("https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/"
                 "get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares"
                 "&locale=en_US&portfolioId=333011&component=fundDownload&userType=individual")
GRAYSCALE = {
    "GBTC": "https://reporting-prod-20231113144948145500000003.s3.us-east-1.amazonaws.com/product-performance/672e88c7-dac6-4fcd-9069-18eef01a2c73.xlsx",
    "BTC": "https://reporting-prod-20231113144948145500000003.s3.amazonaws.com/product-performance/9ba286d6-3067-4153-b430-81d9d7a25696.xlsx",
}
FBTC_PDF = ("https://www.actionsxchangerepository.fidelity.com/ShowDocument/documentPDF.htm?clientId=Fidelity"
            "&applicationId=MFL&securityId=315948109&docType=DALY&docFormat=pdf&securityIdType=CUSIP"
            "&collectionId=1101373&docName=FBTC_Holdings.pdf&criticalIndicator=Y&pdfReaderStatus=Y")
BITB_PAGE = "https://bitbetf.com/"
ARKB_CSV = "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_21SHARES_BITCOIN_ETF_ARKB_HOLDINGS.csv"
ARKB_HISTORY = "https://api.secondary.21shares.com/api/product_valuation_history/ARKB"
HODL_HOLDINGS = "https://www.vaneck.com/Main/HoldingsBlock/GetDataset/?blockId=348327&pageId=243755&ticker=HODL"
HODL_HISTORY = "https://www.vaneck.com/us/en/investments/bitcoin-etf-hodl/downloads/fundhistoprices/"
MSBT_DATA = "https://www.morganstanley.com/im/json/imwebdata/data/product/EF/100761/"
BRRR_WIDGETS = ("https://www-api.coinshares.com/api/v2/Widgets?ApiKey=094DA478-140C-4E3E-B394-7A19BBE8326B"
                "&names=VALKYRIE_HOLDINGS_BRRR,VALKYRIE_DAILY_BRRR")
EZBC_API = "https://www.franklintempleton.com/api/pds/price-and-performance"
EZBC_VARIABLES = {"productId": "39639", "countryCode": "US", "languageCode": "en_US", "shareClassCode": "SINGLCLASS"}
BTCO_API = "https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/46091J101"
BTCW_PAGE = "https://www.wisdomtree.com/us/products/crypto/btcw"


# --------------------------------------------------------------------------- IBIT

def _ibit_historical() -> list[list[str]]:
    return spreadsheet2003_rows(get(IBIT_DOWNLOAD).content, "Historical")


def ibit_current() -> Snapshot:
    rows = list(csv.reader(io.StringIO(get(IBIT_HOLDINGS).text)))
    meta = {row[0]: row[1] for row in rows if len(row) >= 2}
    header = next(i for i, row in enumerate(rows) if row and row[0] == "Ticker")
    columns = rows[header]
    bitcoin = next(dict(zip(columns, row)) for row in rows[header + 1:] if row and row[0] == "BTC")
    as_of = iso_date(meta["Fund Holdings as of"], "%b %d, %Y")
    nav_row = next(row for row in _ibit_historical()[1:] if row and iso_date(row[0], "%b %d, %Y") == as_of)
    return Snapshot("IBIT", as_of, number(bitcoin["Quantity"]), IBIT_HOLDINGS,
                    shares_outstanding=number(meta["Shares Outstanding"]), nav=number(nav_row[1]),
                    net_assets=number(bitcoin["Market Value"]))


def ibit_history() -> pd.DataFrame:
    # Rows before the fund's first NAV carry "--" placeholders.
    rows = [row for row in _ibit_historical()[1:] if row and row[0] and row[1] not in ("", "--") and row[3] not in ("", "--")]
    frame = pd.DataFrame({
        "date": [iso_date(row[0], "%b %d, %Y") for row in rows],
        "nav": [number(row[1]) for row in rows],
        "shares_outstanding": [number(row[3]) for row in rows],
    })
    frame["net_assets"] = frame["nav"] * frame["shares_outstanding"]
    return frame


# --------------------------------------------------------------------------- Grayscale (GBTC, BTC)

def _grayscale(fund: str) -> tuple[list[list], list[list]]:
    content = get(GRAYSCALE[fund]).content
    return xlsx_rows(content, "Daily Performance"), xlsx_rows(content, "Holdings")


def grayscale_current(fund: str) -> Snapshot:
    daily, holdings = _grayscale(fund)
    head = daily[0]
    latest = dict(zip(head, daily[1]))
    per_share = dict(zip(holdings[0], holdings[1]))
    as_of = excel_date(per_share["Date"])
    if excel_date(latest["Date"]) != as_of:
        raise ValueError(f"{fund}: holdings dated {as_of} but performance dated {excel_date(latest['Date'])}")
    shares = number(latest["Shares Outstanding"])
    btc_per_share = number(per_share["Asset/Share"])
    return Snapshot(fund, as_of, shares * btc_per_share, GRAYSCALE[fund], shares_outstanding=shares,
                    btc_per_share=btc_per_share, nav=number(latest["NAV Per Share"]), net_assets=number(latest["AUM"]))


def grayscale_history(fund: str) -> pd.DataFrame:
    daily, _ = _grayscale(fund)
    head = daily[0]
    records = [dict(zip(head, row)) for row in daily[1:] if row and row[0]]
    return pd.DataFrame({
        "date": [excel_date(r["Date"]) for r in records],
        "nav": [number(r["NAV Per Share"]) for r in records],
        "shares_outstanding": [number(r["Shares Outstanding"]) for r in records],
        "net_assets": [number(r["AUM"]) for r in records],
    })


# --------------------------------------------------------------------------- FBTC

def fbtc_current() -> Snapshot:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(get(FBTC_PDF).content)
        handle.flush()
        text = subprocess.run(["pdftotext", "-layout", handle.name, "-"], capture_output=True, text=True,
                              check=True).stdout
    as_of = iso_date(re.search(r"Holding as of:\s*(\S+)", text).group(1).title(), "%d-%b-%y")
    shares = number(re.search(r"Current Shares Outstanding:\s*([\d,]+)", text).group(1))
    bitcoin = re.search(r"BITCOIN\s+Crypto Asset\s+US Dollar\s+([\d.,]+)\s+\$\s*([\d.,]+)", text)
    total = number(re.search(r"Total:\s+\$\s*([\d.,]+)", text).group(1))
    return Snapshot("FBTC", as_of, number(bitcoin.group(1)), FBTC_PDF, shares_outstanding=shares,
                    nav=total / shares, net_assets=total)


# --------------------------------------------------------------------------- BITB

def bitb_current() -> tuple[Snapshot, list[dict]]:
    page = get(BITB_PAGE, headers={"User-Agent": CHROME_HEADERS["User-Agent"]}).text
    holdings = json_after(page, "holdings")
    bitcoin = next(item for item in holdings["basket"] if item["companyName"] == "BITCOIN")
    wallets = json_after(page, "walletBalances")
    snapshot = Snapshot("BITB", holdings["asOfDate"], float(bitcoin["shares"]), BITB_PAGE,
                        shares_outstanding=float(json_after(page, "sharesOutstanding")),
                        nav=float(json_after(page, "navAndMarketPrice")["nav"]),
                        net_assets=float(bitcoin["marketValue"]))
    wallet_total = sum(w["balance"] for w in wallets if w.get("active"))
    snapshot.note = f"{len(wallets)} custody addresses holding {wallet_total:,.8f} BTC"
    return snapshot, wallets


def bitb_nav_history() -> pd.DataFrame:
    """Daily NAV from the page's chart data, stamped at 4pm New York time in milliseconds."""
    points = json_after(get(BITB_PAGE, headers={"User-Agent": CHROME_HEADERS["User-Agent"]}).text,
                        "navAndMarketPrice")["chart"]["nav"]
    stamps = pd.to_datetime([point[0] for point in points], unit="ms", utc=True).tz_convert("America/New_York")
    return pd.DataFrame({"date": stamps.strftime("%Y-%m-%d"), "nav": [float(point[1]) for point in points]})


# --------------------------------------------------------------------------- ARKB

def arkb_current() -> Snapshot:
    rows = list(csv.DictReader(io.StringIO(get(ARKB_CSV).text)))
    bitcoin = next(row for row in rows if (row.get("company") or "").upper() == "BITCOIN")
    published = pd.Timestamp(iso_date(bitcoin["date"], "%m/%d/%Y"))
    # ARK stamps the file with the next business day; the position is the prior close.
    as_of = (published - pd.offsets.BDay(1)).date().isoformat()
    return Snapshot("ARKB", as_of, number(bitcoin["shares"]), ARKB_CSV, net_assets=number(bitcoin["market value ($)"]),
                    note=f"file dated {published.date()}; whole coins only")


def arkb_history() -> pd.DataFrame:
    """21Shares' daily valuation record: units outstanding, total NAV and the index price."""
    rows = [row for row in get(ARKB_HISTORY).json()["data"]
            if row.get("total_nav") and row.get("total_units_outstanding") and row.get("index")]
    return pd.DataFrame({
        "date": [row["valuation_date"] for row in rows],
        "nav": [float(row["nav_per_share"]) for row in rows],
        "shares_outstanding": [float(row["total_units_outstanding"]) for row in rows],
        "net_assets": [float(row["total_nav"]) for row in rows],
        "index_price": [float(row["index"]) for row in rows],
    }).sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- HODL

def _vaneck_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = CHROME_HEADERS["User-Agent"]
    return session


def hodl_current() -> Snapshot:
    data = _vaneck_session().get(HODL_HOLDINGS, timeout=60).json()
    bitcoin = next(h for h in data["Holdings"] if h["HoldingName"] == "Bitcoin")
    return Snapshot("HODL", data["AsOfDate"][:10], number(bitcoin["Shares"]), HODL_HOLDINGS,
                    net_assets=number(bitcoin["MV"]), note="custody quantity, whole coins")


def hodl_history() -> pd.DataFrame:
    response = _vaneck_session().get(HODL_HISTORY, timeout=60)
    response.raise_for_status()
    sheet = xlsx_rows(response.content, _first_sheet(response.content))
    head = next(i for i, row in enumerate(sheet) if row and row[0] == "Date")
    records = [dict(zip(sheet[head], row)) for row in sheet[head + 1:] if row and row[0]]
    frame = pd.DataFrame({
        "date": [iso_date(r["Date"], "%m/%d/%Y") for r in records],
        "nav": [number(r["NAV"]) for r in records],
        "net_assets": [number(r["AUM"]) for r in records],
        "index_price": [number(r["Index Level"]) for r in records],
        "traded": [r.get("Volume") not in (None, "", "--") for r in records],
    })
    frame["shares_outstanding"] = frame["net_assets"] / frame["nav"]
    return frame


def _first_sheet(content: bytes) -> str:
    workbook = zipfile.ZipFile(io.BytesIO(content)).read("xl/workbook.xml").decode()
    return re.search(r'<(?:\w+:)?sheet\b[^>]*\bname="([^"]+)"', workbook).group(1)


# --------------------------------------------------------------------------- MSBT

def msbt_current() -> Snapshot:
    headers = {**CHROME_HEADERS, "Referer": "https://www.morganstanley.com/"}
    holdings = get(MSBT_DATA + "chart/etfTradeDateHoldingsCurrent.json", headers=headers).json()["en"]
    pricing = get(MSBT_DATA + "detail/en-pricing.json", headers=headers).json()["en"]
    net_assets = get(MSBT_DATA + "detail/en-netAsset.json", headers=headers).json()["en"]["netAsset"]
    bitcoin = next(h for h in holdings["holdings"] if h["securityDescription"] == "BITCOIN")
    nav = number(pricing["shareClasses"][0]["currencies"][0]["pricings"]["nav6f"])
    total = number(net_assets["value2f"]) * 1e6
    return Snapshot("MSBT", iso_date(holdings["effectiveDate"], "%m/%d/%Y"), number(bitcoin["quantity"]),
                    MSBT_DATA + "chart/etfTradeDateHoldingsCurrent.json", shares_outstanding=total / nav,
                    nav=nav, net_assets=total, note="shares = net assets (to $10k) / NAV")


def msbt_nav_history() -> pd.DataFrame:
    headers = {**CHROME_HEADERS, "Referer": "https://www.morganstanley.com/"}
    series = get(MSBT_DATA + "chart/historicalNav.json", headers=headers).json()["en"]["shareClasses"][0][
        "currencies"][0]["series"]
    return pd.DataFrame({"date": [iso_date(d, "%m/%d/%Y") for d in series["category"]],
                         "nav": [number(v) for v in series["data"]]})


# --------------------------------------------------------------------------- BRRR

def brrr_current() -> Snapshot:
    widgets = get(BRRR_WIDGETS, headers={"Referer": "https://coinshares.com/"}).json()
    meta = {}
    for widget in widgets:
        for section in widget["sections"]:
            values = {m["key"]: m["value"] for m in section["meta"]}
            if values.get("securityname") == "BITCOIN" or section["type"] == "VALKYRIE_PERFORMANCE_DAILY":
                meta.update(values)
    return Snapshot("BRRR", meta["RateDate"], number(meta["shares"]), BRRR_WIDGETS,
                    shares_outstanding=number(meta["sharesoutstanding"]), nav=number(meta["NAV"]),
                    net_assets=number(meta["netassets"]))


# --------------------------------------------------------------------------- EZBC

EZBC_FACTS = ("query FundFact($productId: String!, $countryCode: String!, $shareClassCode: String!, "
              "$languageCode: String!) { FundFact(fundid: $productId, shareclasscode: $shareClassCode, "
              "countrycode: $countryCode, languagecode: $languageCode) { elemnamestd elemvalue asofdatestd } }")
EZBC_PRICES = ("query PricingDetails($productId: String!, $countryCode: String!, $languageCode: String!, "
               "$shareClassCode: String!) { PricesHistory(fundid: $productId, shareclasscode: $shareClassCode, "
               "countrycode: $countryCode, languagecode: $languageCode) { prices { asofdatestd navstd "
               "totalnetassets } } }")


def _ezbc_prices() -> list[dict]:
    data = post_json(f"{EZBC_API}?op=PricingDetails&id=4", {"query": EZBC_PRICES, "variables": EZBC_VARIABLES,
                                                            "operationName": "PricingDetails"})["data"]
    history = data["PricesHistory"]
    return (history[0] if isinstance(history, list) else history)["prices"]


def ezbc_current() -> Snapshot:
    facts = post_json(f"{EZBC_API}?op=FundFact&id=2", {"query": EZBC_FACTS, "variables": EZBC_VARIABLES,
                                                       "operationName": "FundFact"})["data"]["FundFact"]
    fact = {row["elemnamestd"]: row for row in facts}
    as_of = fact["BITCOIN_IN_FUND"]["asofdatestd"]
    if fact["SHARES_OUTSTANDING"]["asofdatestd"] != as_of:
        raise ValueError("EZBC: bitcoin and shares outstanding are dated differently")
    price = next((p for p in _ezbc_prices() if p["asofdatestd"] == as_of), None)
    return Snapshot("EZBC", as_of, number(fact["BITCOIN_IN_FUND"]["elemvalue"]), f"{EZBC_API}?op=FundFact",
                    shares_outstanding=number(fact["SHARES_OUTSTANDING"]["elemvalue"]),
                    nav=number(price["navstd"]) if price else None,
                    net_assets=number(price["totalnetassets"]) if price else None)


def ezbc_history() -> pd.DataFrame:
    prices = [p for p in _ezbc_prices() if p.get("navstd") and p.get("totalnetassets")]
    frame = pd.DataFrame({
        "date": [p["asofdatestd"] for p in prices],
        "nav": [number(p["navstd"]) for p in prices],
        "net_assets": [number(p["totalnetassets"]) for p in prices],
    })
    frame["shares_outstanding"] = frame["net_assets"] / frame["nav"]
    return frame


# --------------------------------------------------------------------------- BTCO

def btco_current() -> Snapshot:
    # Invesco's API answers this library's user agent but rejects browser user agents.
    details = get(f"{BTCO_API}?expand=nav&idType=cusip&variationType=fundDetails&productType=ETF").json()
    prices = get(f"{BTCO_API}/prices?idType=cusip&variationType=priceListing&productType=ETF"
                 "&productSubType=ETF-Non-40%20Act").json()
    if details["effectiveDate"] != prices["effectiveDate"]:
        raise ValueError("BTCO: holdings and prices are dated differently")
    return Snapshot("BTCO", details["effectiveDate"], float(details["units"]), f"{BTCO_API} (fundDetails)",
                    shares_outstanding=float(prices["sharesOutstanding"]), nav=float(prices["nav"]),
                    net_assets=float(details["shareclassTotalNetAssets"]))


def btco_nav_history() -> pd.DataFrame:
    points = get(f"{BTCO_API}/navs?idType=cusip&productType=ETF").json()["lineChartData"][0]["data"]
    return pd.DataFrame({"date": [iso_date(p["date"], "%m/%d/%Y") for p in points],
                         "nav": [float(p["value"]) for p in points]}).sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- BTCW

@cache
def _btcw_page() -> dict:
    """WisdomTree sits behind a Cloudflare challenge, so BTCW is read in a headless browser,
    once per run for both today's holdings and the NAV history.

    Needs PLAYWRIGHT_MODULE (the path to playwright or playwright-core's index.mjs) and Node
    18 or newer from NODE_BINARY (default: the newest nvm install, else `node` on PATH).
    """
    module = os.environ.get("PLAYWRIGHT_MODULE")
    if not module:
        raise RuntimeError("PLAYWRIGHT_MODULE is not set")
    result = subprocess.run([_node(), str(HERE / "btcw.mjs"), module, BTCW_PAGE], capture_output=True,
                            text=True, timeout=180)
    if result.returncode:
        detail = next((line for line in result.stderr.splitlines() if "Error:" in line), result.stderr.strip()[-200:])
        raise RuntimeError(f"BTCW browser read failed: {detail.strip()}")
    return json.loads(result.stdout)


def btcw_current() -> Snapshot:
    data = _btcw_page()
    bitcoin = next(h for h in data["holdings"] if h["securityName"] == "BITCOIN")
    return Snapshot("BTCW", bitcoin["dt"][:10], float(bitcoin["shares"]), BTCW_PAGE,
                    shares_outstanding=number(data["sharesOutstanding"]) if data.get("sharesOutstanding") else None,
                    nav=number(data["nav"]) if data.get("nav") else None,
                    net_assets=float(bitcoin["marketValueBase"]))


def btcw_history() -> pd.DataFrame:
    """WisdomTree's daily NAV history: NAV, shares outstanding and AUM (in thousands)."""
    rows = [row for row in _btcw_page()["navHistory"] if row.get("nav") and row.get("sharesOutstanding")]
    frame = pd.DataFrame({
        "date": [row["dt"][:10] for row in rows],
        "nav": [float(row["nav"]) for row in rows],
        "shares_outstanding": [float(row["sharesOutstanding"]) for row in rows],
    })
    frame["net_assets"] = frame["nav"] * frame["shares_outstanding"]
    return frame.sort_values("date").reset_index(drop=True)


def _node() -> str:
    if os.environ.get("NODE_BINARY"):
        return os.environ["NODE_BINARY"]
    installs = sorted(Path.home().glob(".nvm/versions/node/v*/bin/node"),
                      key=lambda p: [int(x) for x in p.parts[-3][1:].split(".")])
    return str(installs[-1]) if installs else "node"


CURRENT = {
    "IBIT": ibit_current,
    "FBTC": fbtc_current,
    "GBTC": lambda: grayscale_current("GBTC"),
    "BTC": lambda: grayscale_current("BTC"),
    "BITB": lambda: bitb_current()[0],
    "ARKB": arkb_current,
    "HODL": hodl_current,
    "MSBT": msbt_current,
    "BRRR": brrr_current,
    "EZBC": ezbc_current,
    "BTCO": btco_current,
    "BTCW": btcw_current,
}
HISTORY = {
    "IBIT": ibit_history,
    "GBTC": lambda: grayscale_history("GBTC"),
    "BTC": lambda: grayscale_history("BTC"),
    "HODL": hodl_history,
    "EZBC": ezbc_history,
    "ARKB": arkb_history,
    "BTCW": btcw_history,
}
# The bitcoin price each fund's NAV uses, as named in its 10-Q. Grayscale moved from the CoinDesk
# Bitcoin Price Index (XBX) to the CoinDesk Bitcoin Benchmark Rate on 2026-04-01.
BRRNY = "CME CF Bitcoin Reference Rate - New York Variant"
MARKETVECTOR = "MarketVector Bitcoin Benchmark Rate"
FUND_INDEX = {
    "IBIT": BRRNY, "BITB": BRRNY, "ARKB": BRRNY, "BRRR": BRRNY, "EZBC": BRRNY, "BTCW": BRRNY,
    "HODL": MARKETVECTOR,
    "GBTC": "CoinDesk Bitcoin Benchmark Rate", "BTC": "CoinDesk Bitcoin Benchmark Rate",
    "MSBT": "CoinDesk Bitcoin Benchmark 4PM NY Settlement Rate",
    "FBTC": "Fidelity Bitcoin Reference Rate",
    "BTCO": "Lukka Prime Bitcoin Reference Rate",
}
# NAV history only; BRRR's published series is a rounded cumulative return, too coarse to use.
NAV_HISTORY = {"BITB": bitb_nav_history, "MSBT": msbt_nav_history, "BTCO": btco_nav_history}
# Funds whose daily share counts change on settlement day (T+1) rather than trade day: shifted
# back one trading day, their daily flows match trade-date flow tables and their quarter ends
# match the SEC exactly. Their figure for a day is the trade-date position of the previous trading day.
SETTLEMENT_DATED = {"IBIT", "BTCW", "GBTC", "BTC", "HODL", "EZBC"}
SEC_CIK = {
    "IBIT": 1980994, "FBTC": 1852317, "GBTC": 1588489, "BTC": 2015034, "BITB": 1763415, "ARKB": 1869699,
    "HODL": 1838028, "MSBT": 2103612, "BRRR": 1841175, "EZBC": 1992870, "BTCO": 1855781, "BTCW": 1850391,
}
