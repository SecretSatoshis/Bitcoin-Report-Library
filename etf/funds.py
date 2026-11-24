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
import threading
import zipfile
from functools import wraps
from pathlib import Path

import pandas as pd

from .common import (
    CHROME_HEADERS, Snapshot, excel_date, get, iso_date, json_after, number, post_json,
    previous_us_trading_day, spreadsheet2003_rows, xlsx_rows,
)

HERE = Path(__file__).resolve().parent

_CACHED_FUNCTIONS: list = []


def _per_run_cache(fn):
    """Thread-safe per-run cache so current + history readers share one HTTP download."""
    lock = threading.Lock()
    store: dict = {}

    @wraps(fn)
    def wrapper(*args):
        with lock:
            if args not in store:
                store[args] = fn(*args)
            return store[args]

    def cache_clear():
        with lock:
            store.clear()

    wrapper.cache_clear = cache_clear
    _CACHED_FUNCTIONS.append(wrapper)
    return wrapper


def clear_caches() -> None:
    """Clear all per-run ETF download caches (called at the start of each collect run)."""
    for fn in _CACHED_FUNCTIONS:
        fn.cache_clear()

# Provisional per-fund daily flows in USD: TFTC's open dataset (CC BY 4.0, attribution "TFTC —
# tftc.io/bitcoin-etf-flows"), which takes them from SoSoValue. They fill only the days the
# issuers have not yet published, and are published as "provisional" (see etf.build).
PROVISIONAL_FLOWS = "https://www.tftc.io/bitcoin-etf-flows/data.json"
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

@_per_run_cache
def _ibit_historical() -> list[list[str]]:
    return spreadsheet2003_rows(get(IBIT_DOWNLOAD).content, "Historical")


def ibit_current() -> Snapshot:
    rows = list(csv.reader(io.StringIO(get(IBIT_HOLDINGS).text)))
    meta = {row[0]: row[1] for row in rows if len(row) >= 2}
    header = next(i for i, row in enumerate(rows) if row and row[0] == "Ticker")
    columns = rows[header]
    bitcoin = next(dict(zip(columns, row)) for row in rows[header + 1:] if row and row[0] == "BTC")
    as_of = iso_date(meta["Fund Holdings as of"], "%b %d, %Y")
    nav_row = next((row for row in _ibit_historical()[1:] if row and iso_date(row[0], "%b %d, %Y") == as_of), None)
    return Snapshot("IBIT", as_of, number(bitcoin["Quantity"]), IBIT_HOLDINGS,
                    shares_outstanding=number(meta["Shares Outstanding"]),
                    nav=number(nav_row[1]) if nav_row is not None else None,
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

@_per_run_cache
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

def _match_first(text: str, patterns: tuple[str, ...], field: str) -> re.Match:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is not None:
            return match
    snippet = re.sub(r"\s+", " ", text).strip()[:200]
    raise ValueError(f"FBTC PDF: could not parse {field} (excerpt: {snippet!r})")


def fbtc_current() -> Snapshot:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(get(FBTC_PDF).content)
        handle.flush()
        proc = subprocess.run(["pdftotext", "-layout", handle.name, "-"], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FBTC pdftotext failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
        text = proc.stdout
    if not text.strip():
        raise ValueError("FBTC PDF: pdftotext produced empty output")
    date_match = _match_first(
        text,
        (
            r"Holding\s+as\s+of[:\s]+(\S+)",
            r"As\s+of[:\s]+(\d{1,2}-[A-Za-z]{3}-\d{2,4}|\d{1,2}/\d{1,2}/\d{2,4})",
        ),
        "as-of date",
    )
    raw_date = date_match.group(1).strip().rstrip(",.")
    as_of = iso_date(raw_date.title(), "%d-%b-%y", "%d-%b-%Y", "%m/%d/%Y", "%Y-%m-%d")

    shares_match = _match_first(
        text,
        (
            r"Current\s+Shares\s+Outstanding[:\s]+([\d,]+(?:\.\d+)?)",
            r"Shares\s+Outstanding[:\s]+([\d,]+(?:\.\d+)?)",
        ),
        "shares outstanding",
    )
    shares = number(shares_match.group(1))

    bitcoin = _match_first(
        text,
        (
            r"BITCOIN\s+Crypto\s+Asset\s+US\s+Dollar\s+([\d.,]+)\s+\$\s*([\d.,]+)",
            r"BITCOIN\b[^\n]*?\b([\d,]+\.\d+)\s+\$?\s*([\d,]+\.\d+)",
        ),
        "bitcoin holding",
    )
    btc_qty = number(bitcoin.group(1))
    btc_mv = number(bitcoin.group(2))

    total_match = re.search(r"Total[:\s]+\$\s*([\d.,]+)", text, re.IGNORECASE)
    total = number(total_match.group(1)) if total_match is not None else btc_mv
    return Snapshot("FBTC", as_of, btc_qty, FBTC_PDF, shares_outstanding=shares,
                    nav=total / shares if shares > 0 else None, net_assets=total)


# --------------------------------------------------------------------------- BITB

def _bitb_json(page: str, *keys: str, expect: type | tuple[type, ...] | None = None):
    """Extract a JSON field from either raw HTML or Next.js RSC-escaped script payloads."""
    candidates = [page]
    if '\\"' in page:
        candidates.append(page.replace('\\\\"', '"').replace('\\"', '"'))
    decoder = json.JSONDecoder()
    for text in candidates:
        for key in keys:
            marker = f'"{key}":'
            pos = 0
            while True:
                start = text.find(marker, pos)
                if start < 0:
                    break
                pos = start + len(marker)
                try:
                    value, _ = decoder.raw_decode(text, pos)
                except json.JSONDecodeError:
                    continue
                if expect is None or isinstance(value, expect):
                    return value
    raise ValueError(f"page data has no {keys[0]!r}")


@_per_run_cache
def _bitb_page() -> str:
    return get(BITB_PAGE, headers={"User-Agent": CHROME_HEADERS["User-Agent"]}).text


def bitb_current() -> tuple[Snapshot, list[dict]]:
    page = _bitb_page()
    holdings = _bitb_json(page, "holdings", expect=dict)
    bitcoin = next(
        item for item in holdings["basket"]
        if (item.get("name") or item.get("companyName")) == "BITCOIN"
    )
    wallets = _bitb_json(page, "walletBalances", expect=list)
    as_of = holdings["asOfDate"]
    btc_held = float(bitcoin["assetInTrust"] if "assetInTrust" in bitcoin else bitcoin["shares"])
    btc_per_share = float(bitcoin["assetPerShare"]) if bitcoin.get("assetPerShare") else None
    net_assets = float(bitcoin["marketValue"])
    shares = (
        round(btc_held / btc_per_share)
        if btc_per_share and btc_per_share > 0
        else float(_bitb_json(page, "sharesOutstanding", expect=(int, float)))
    )
    nav_obj = _bitb_json(page, "navMarketPrice", "navAndMarketPrice", expect=dict)
    raw_nav = nav_obj.get("nav")
    nav_val = float(raw_nav["value"]) if isinstance(raw_nav, dict) else float(raw_nav)
    nav_date = nav_obj.get("asOfDate")
    nav = nav_val if nav_date in (None, as_of) else (net_assets / shares if shares > 0 else None)
    snapshot = Snapshot(
        "BITB",
        as_of,
        btc_held,
        BITB_PAGE,
        shares_outstanding=float(shares),
        btc_per_share=btc_per_share,
        nav=nav,
        net_assets=net_assets,
    )
    wallet_total = sum(w["balance"] for w in wallets if w.get("active"))
    snapshot.note = f"{len(wallets)} custody addresses holding {wallet_total:,.8f} BTC"
    return snapshot, wallets


def bitb_nav_history() -> pd.DataFrame:
    """Daily NAV from the page's chart data (handles both RSC UTC objects and legacy NY arrays)."""
    nav_obj = _bitb_json(_bitb_page(), "navMarketPrice", "navAndMarketPrice", expect=dict)
    points = nav_obj["chart"]["nav"]
    if points and isinstance(points[0], dict):
        stamps = pd.to_datetime([point["t"] for point in points], unit="ms", utc=True)
        navs = [float(point["v"]) for point in points]
    else:
        stamps = pd.to_datetime([point[0] for point in points], unit="ms", utc=True).tz_convert("America/New_York")
        navs = [float(point[1]) for point in points]
    return pd.DataFrame({"date": stamps.strftime("%Y-%m-%d"), "nav": navs})


# --------------------------------------------------------------------------- ARKB

def arkb_current() -> Snapshot:
    rows = list(csv.DictReader(io.StringIO(get(ARKB_CSV).text)))
    bitcoin = next(row for row in rows if (row.get("company") or "").upper() == "BITCOIN")
    published = pd.Timestamp(iso_date(bitcoin["date"], "%m/%d/%Y"))
    # ARK stamps the file with the next trading day; the position is the prior close.
    as_of = previous_us_trading_day(published).date().isoformat()
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

def hodl_current() -> Snapshot:
    data = get(HODL_HOLDINGS, headers={"User-Agent": CHROME_HEADERS["User-Agent"]}).json()
    bitcoin = next(h for h in data["Holdings"] if h["HoldingName"] == "Bitcoin")
    return Snapshot("HODL", data["AsOfDate"][:10], number(bitcoin["Shares"]), HODL_HOLDINGS,
                    net_assets=number(bitcoin["MV"]), note="custody quantity, whole coins")


def hodl_history() -> pd.DataFrame:
    response = get(HODL_HISTORY, headers={"User-Agent": CHROME_HEADERS["User-Agent"]})
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
    match = re.search(r'<(?:\w+:)?sheet\b[^>]*\bname="([^"]+)"', workbook)
    if match is None:
        raise ValueError("xlsx workbook has no named sheet")
    return match.group(1)


# --------------------------------------------------------------------------- MSBT

def _msbt_unwrap(payload: dict, label: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"MSBT {label}: expected JSON object, got {type(payload).__name__}")
    return payload.get("en") if isinstance(payload.get("en"), dict) else payload


def msbt_current() -> Snapshot:
    headers = {**CHROME_HEADERS, "Referer": "https://www.morganstanley.com/"}
    holdings_url = MSBT_DATA + "chart/etfTradeDateHoldingsCurrent.json"
    holdings = _msbt_unwrap(get(holdings_url, headers=headers).json(), "holdings")
    pricing = _msbt_unwrap(get(MSBT_DATA + "detail/en-pricing.json", headers=headers).json(), "pricing")
    net_assets_doc = _msbt_unwrap(get(MSBT_DATA + "detail/en-netAsset.json", headers=headers).json(), "netAsset")
    net_assets = net_assets_doc.get("netAsset") if isinstance(net_assets_doc.get("netAsset"), dict) else net_assets_doc

    holding_rows = holdings.get("holdings")
    if not isinstance(holding_rows, list) or not holding_rows:
        raise ValueError("MSBT holdings payload contains no holdings list")
    bitcoin = next(
        (
            h for h in holding_rows
            if isinstance(h, dict) and "BITCOIN" in str(
                h.get("securityDescription") or h.get("securityName") or h.get("ticker") or ""
            ).upper()
        ),
        None,
    )
    if bitcoin is None:
        raise ValueError("MSBT holdings payload contains no BITCOIN row")

    raw_date = holdings.get("effectiveDate") or holdings.get("asOfDate")
    if not raw_date:
        raise ValueError("MSBT holdings payload missing effectiveDate")
    as_of = iso_date(raw_date, "%m/%d/%Y", "%Y-%m-%d", "%b %d, %Y")

    raw_qty = bitcoin.get("quantity") if bitcoin.get("quantity") is not None else bitcoin.get("shares")
    if raw_qty is None:
        raise ValueError("MSBT bitcoin holding missing quantity")

    share_classes = pricing.get("shareClasses") or []
    currencies = (share_classes[0].get("currencies") or []) if share_classes else []
    pricings = (currencies[0].get("pricings") or {}) if currencies else {}
    raw_nav = pricings.get("nav6f") or pricings.get("nav2f") or pricings.get("nav")
    nav = number(raw_nav) if raw_nav is not None else None

    raw_total = net_assets.get("value2f") if isinstance(net_assets, dict) else None
    if raw_total is None and isinstance(net_assets, dict):
        raw_total = net_assets.get("value")
    total = number(raw_total) * 1e6 if raw_total is not None else None
    shares = total / nav if total is not None and nav and nav > 0 else None

    return Snapshot("MSBT", as_of, number(raw_qty), holdings_url, shares_outstanding=shares,
                    nav=nav, net_assets=total, note="shares = net assets (to $10k) / NAV")


def msbt_nav_history() -> pd.DataFrame:
    headers = {**CHROME_HEADERS, "Referer": "https://www.morganstanley.com/"}
    doc = _msbt_unwrap(get(MSBT_DATA + "chart/historicalNav.json", headers=headers).json(), "historicalNav")
    series = doc["shareClasses"][0]["currencies"][0]["series"]
    return pd.DataFrame({"date": [iso_date(d, "%m/%d/%Y", "%Y-%m-%d") for d in series["category"]],
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


@_per_run_cache
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

@_per_run_cache
def _btcw_page() -> dict:
    """WisdomTree's fund data loads only inside its web page, so BTCW is read in a headless
    browser, once per run for both today's holdings and the NAV history.

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
    """WisdomTree's daily NAV history: NAV and shares outstanding; net assets is their product."""
    rows = [row for row in _btcw_page()["navHistory"] if row.get("nav") and row.get("sharesOutstanding")]
    frame = pd.DataFrame({
        "date": [row["dt"][:10] for row in rows],
        "nav": [float(row["nav"]) for row in rows],
        "shares_outstanding": [float(row["sharesOutstanding"]) for row in rows],
    })
    frame["net_assets"] = frame["nav"] * frame["shares_outstanding"]
    return frame.sort_values("date").reset_index(drop=True)


def provisional_flows() -> pd.DataFrame:
    """Each fund's flow in USD for every day the provisional dataset lists (fund, date, flow_usd)."""
    days = get(PROVISIONAL_FLOWS).json()["days"]
    rows = [{"fund": fund, "date": day["date"], "flow_usd": float(flow)}
            for day in days for fund, flow in (day.get("perEtfUsd") or {}).items()
            if fund in CURRENT and flow is not None]
    if not rows:
        raise ValueError("the provisional flow dataset lists no fund flows")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


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
# Subset whose published holdings snapshots are ALSO settlement-dated (T+1).
# IBIT (latest-holdings.csv) and EZBC (FundFact BITCOIN_IN_FUND) already stamp holdings on
# trade date T while stamping shares outstanding on settlement date T+1.
SETTLEMENT_DATED_HOLDINGS = {"BTCW", "GBTC", "BTC", "HODL"}
TOP5_FUNDS = ("IBIT", "FBTC", "GBTC", "BTC", "BITB")
SEC_CIK = {
    "IBIT": 1980994, "FBTC": 1852317, "GBTC": 1588489, "BTC": 2015034, "BITB": 1763415, "ARKB": 1869699,
    "HODL": 1838028, "MSBT": 2103612, "BRRR": 1841175, "EZBC": 1992870, "BTCO": 1855781, "BTCW": 1850391,
}
