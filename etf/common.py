"""Shared pieces for the spot bitcoin ETF readers: the record format, HTTP and file parsing."""

import io
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests

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


def post_json(url: str, payload: dict, *, timeout: int = 60) -> dict:
    response = requests.post(url, json=payload, timeout=timeout,
                             headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"})
    response.raise_for_status()
    return response.json()


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
        for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(node.text or "" for node in item.iter(f"{{{ns['m']}}}t")))
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.get("Id"): rel.get("Target") for rel in rels}
    sheet = next((s for s in workbook.find("m:sheets", ns) if s.get("name") == sheet_name), None)
    if sheet is None:
        raise ValueError(f"workbook has no sheet {sheet_name!r}")
    target = targets[sheet.get(f"{{{ns['r']}}}id")].lstrip("/")
    path = target if target.startswith("xl/") else f"xl/{target}"
    rows = []
    for row in ET.fromstring(archive.read(path)).iter(f"{{{ns['m']}}}row"):
        values = {}
        for cell in row.findall("m:c", ns):
            column = re.sub(r"\d", "", cell.get("r"))
            index = 0
            for char in column:
                index = index * 26 + ord(char) - 64
            kind, raw = cell.get("t"), cell.find("m:v", ns)
            if kind == "s":
                value = shared[int(raw.text)]
            elif kind == "inlineStr":
                value = "".join(node.text or "" for node in cell.iter(f"{{{ns['m']}}}t"))
            elif raw is None:
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
    root = ET.fromstring(text)
    for sheet in root.findall("ss:Worksheet", ns):
        if sheet.get(f"{{{ns['ss']}}}Name") == sheet_name:
            return [[(cell.find("ss:Data", ns).text if cell.find("ss:Data", ns) is not None else "")
                     for cell in row.findall("ss:Cell", ns)] for row in sheet.iter(f"{{{ns['ss']}}}Row")]
    raise ValueError(f"workbook has no sheet {sheet_name!r}")


def excel_date(value) -> str:
    """An xlsx date cell (serial number or text) as YYYY-MM-DD."""
    if isinstance(value, float):
        return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=value)).date().isoformat()
    return pd.Timestamp(str(value)).date().isoformat()
