"""Shared pieces for the spot bitcoin ETF readers: the record format, HTTP and file parsing."""

import io
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import cache

import pandas as pd
import requests

try:
    from defusedxml import ElementTree as _SAFE_ET
except ImportError:
    _SAFE_ET = None

_FORBIDDEN_XML_RE = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


def _parse_xml(raw: bytes | str) -> ET.Element:
    """Parse XML while blocking DOCTYPE / ENTITY expansion (XXE / billion-laughs)."""
    raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw
    if _FORBIDDEN_XML_RE.search(raw_bytes):
        raise ValueError("Refusing to parse XML containing DOCTYPE or ENTITY declarations")
    if _SAFE_ET is not None:
        return _SAFE_ET.fromstring(raw_bytes)
    return ET.fromstring(raw_bytes)


def _easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (Anonymous Gregorian algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed_fixed(year: int, month: int, day: int) -> date:
    """NYSE observed rule: Saturday -> Friday, Sunday -> Monday."""
    dt = date(year, month, day)
    if dt.weekday() == 5:
        return dt - timedelta(days=1)
    if dt.weekday() == 6:
        return dt + timedelta(days=1)
    return dt


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Nth (1-based) occurrence of `weekday` (0=Mon..6=Sun) in `year`-`month`."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Last occurrence of `weekday` (0=Mon..6=Sun) in `year`-`month`."""
    next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = next_month - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


@cache
def us_market_holidays(year: int) -> frozenset[date]:
    """NYSE full-day market closures for `year`."""
    holidays = {
        _observed_fixed(year, 1, 1),          # New Year's Day
        _nth_weekday(year, 1, 0, 3),          # Martin Luther King Jr. Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),          # Presidents' Day (3rd Mon Feb)
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),            # Memorial Day (last Mon May)
        _observed_fixed(year, 7, 4),          # Independence Day
        _nth_weekday(year, 9, 0, 1),          # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),         # Thanksgiving Day (4th Thu Nov)
        _observed_fixed(year, 12, 25),        # Christmas Day
    }
    if year >= 2022:
        holidays.add(_observed_fixed(year, 6, 19))  # Juneteenth
    # Next year's Jan 1 observed on Dec 31 of this year (e.g. Jan 1 on Saturday).
    if date(year + 1, 1, 1).weekday() == 5:
        holidays.add(date(year, 12, 31))
    return frozenset(holidays)


def is_us_trading_day(value) -> bool:
    """True when `value` is a weekday and not an NYSE holiday."""
    dt = pd.Timestamp(value).date()
    return dt.weekday() < 5 and dt not in us_market_holidays(dt.year)


def previous_us_trading_day(value) -> pd.Timestamp:
    """The latest US equity market trading day strictly before `value`."""
    dt = pd.Timestamp(value).date() - timedelta(days=1)
    while not is_us_trading_day(dt):
        dt -= timedelta(days=1)
    return pd.Timestamp(dt)


def next_us_trading_day(value) -> pd.Timestamp:
    """The earliest US equity market trading day strictly after `value`."""
    dt = pd.Timestamp(value).date() + timedelta(days=1)
    while not is_us_trading_day(dt):
        dt += timedelta(days=1)
    return pd.Timestamp(dt)

USER_AGENT = "bitcoin-report-library (+https://github.com/SecretSatoshis/Bitcoin-Report-Library)"
# Rate limits and server errors are retried; any other HTTP error will not change on a retry.
TRANSIENT_STATUSES = {408, 425, 429}

# Morgan Stanley's CDN rejects requests that do not carry a browser's header set.
CHROME_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "sec-ch-ua": '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


@dataclass
class Snapshot:
    """One fund's published position on one day. Missing fields stay None."""
    fund: str
    as_of: str
    btc_held: float
    source: str
    shares_outstanding: float | None = None
    btc_per_share: float | None = None
    nav: float | None = None
    net_assets: float | None = None
    note: str = ""
    collected_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def __post_init__(self):
        if not self.btc_held or self.btc_held <= 0:
            raise ValueError(f"{self.fund}: no positive bitcoin holding in the source")
        if self.btc_per_share is None and self.shares_outstanding:
            self.btc_per_share = self.btc_held / self.shares_outstanding

    def row(self) -> dict:
        return asdict(self)


def number(text) -> float:
    """'1,414,960,000.00', ' 10,436 ', '$48.36' -> float."""
    cleaned = re.sub(r"[^0-9.\-]", "", str(text))
    if cleaned in {"", "-", "."}:
        raise ValueError(f"not a number: {text!r}")
    return float(cleaned)


def iso_date(text, *formats) -> str:
    for fmt in formats:
        try:
            return datetime.strptime(str(text).strip(), fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {text!r}")


def get(url: str, *, headers: dict | None = None, timeout: int = 60, attempts: int = 3) -> requests.Response:
    """GET with this library's user agent unless `headers` sets one, retrying transient failures."""
    headers = {"User-Agent": USER_AGENT, **(headers or {})}
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            if attempt == attempts or (status is not None and status < 500 and status not in TRANSIENT_STATUSES):
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def post_json(url: str, payload: dict, *, timeout: int = 60, attempts: int = 3) -> dict:
    """POST JSON with retry and exponential backoff on transient failures."""
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            if attempt == attempts or (status is not None and status < 500 and status not in TRANSIENT_STATUSES):
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def json_after(text: str, key: str):
    """Decode the JSON value that follows the first `"key":` in a page's embedded data."""
    marker = f'"{key}":'
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"page data has no {key!r}")
    value, _ = json.JSONDecoder().raw_decode(text, start + len(marker))
    return value


def xlsx_rows(content: bytes, sheet_name: str) -> list[list]:
    """Rows of one sheet of an .xlsx file as strings or numbers, without a spreadsheet library."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    archive = zipfile.ZipFile(io.BytesIO(content))
    shared = []
    if "xl/sharedStrings.xml" in archive.namelist():
        for item in _parse_xml(archive.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(node.text or "" for node in item.iter(f"{{{ns['m']}}}t")))
    workbook = _parse_xml(archive.read("xl/workbook.xml"))
    rels = _parse_xml(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.get("Id"): rel.get("Target") for rel in rels}
    sheet = next((s for s in workbook.find("m:sheets", ns) if s.get("name") == sheet_name), None)
    if sheet is None:
        raise ValueError(f"workbook has no sheet {sheet_name!r}")
    target = targets[sheet.get(f"{{{ns['r']}}}id")].lstrip("/")
    path = target if target.startswith("xl/") else f"xl/{target}"
    rows = []
    for row in _parse_xml(archive.read(path)).iter(f"{{{ns['m']}}}row"):
        values = {}
        next_index = 1
        for cell in row.findall("m:c", ns):
            ref = cell.get("r")
            if ref:
                column = re.sub(r"\d", "", ref)
                index = 0
                for char in column:
                    index = index * 26 + ord(char) - 64
            else:
                index = next_index
            next_index = index + 1
            kind, raw = cell.get("t"), cell.find("m:v", ns)
            if kind == "s":
                value = shared[int(raw.text)] if raw is not None and raw.text is not None else None
            elif kind == "inlineStr":
                value = "".join(node.text or "" for node in cell.iter(f"{{{ns['m']}}}t"))
            elif raw is None or raw.text is None:
                value = None
            else:
                try:
                    value = float(raw.text)
                except ValueError:
                    value = raw.text
            values[index - 1] = value
        rows.append([values.get(i) for i in range(max(values) + 1)] if values else [])
    return rows


def spreadsheet2003_rows(content: bytes, sheet_name: str) -> list[list[str]]:
    """Rows of one sheet of an Excel 2003 XML workbook (iShares' "Data Download")."""
    text = content.decode("utf-8")
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", text)
    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    ss_index = f"{{{ns['ss']}}}Index"
    root = _parse_xml(text)
    for sheet in root.findall("ss:Worksheet", ns):
        if sheet.get(f"{{{ns['ss']}}}Name") == sheet_name:
            rows: list[list[str]] = []
            for row in sheet.iter(f"{{{ns['ss']}}}Row"):
                cells: list[str] = []
                for cell in row.findall("ss:Cell", ns):
                    explicit = cell.get(ss_index)
                    if explicit is not None:
                        target_idx = int(explicit) - 1
                        if target_idx > len(cells):
                            cells.extend([""] * (target_idx - len(cells)))
                    data_node = cell.find("ss:Data", ns)
                    cells.append(data_node.text if data_node is not None and data_node.text is not None else "")
                rows.append(cells)
            return rows
    raise ValueError(f"workbook has no sheet {sheet_name!r}")


def excel_date(value) -> str:
    """An xlsx date cell (serial number or text) as YYYY-MM-DD."""
    if isinstance(value, float):
        return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=value)).date().isoformat()
    return pd.Timestamp(str(value)).date().isoformat()
