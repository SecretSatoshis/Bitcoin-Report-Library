"""Annual reference series, published as annual_reference_data.csv.

One row per series and year:
    series          published name, one of ANNUAL_SERIES
    year            the calendar year the value describes
    value           the value, in `unit`
    unit, source, source_url
    note            how the value was derived, where that needs saying
    retrieved_date  the day the value was fetched; for a hand-kept table, the day it was
                    last checked. Blank for settled history.

A fetched series that fails with an outage reuses its rows from the last published
release, keeping their original retrieved_date, so one slow source cannot stop the daily
release. A series that changes its format, or whose newest year is more than
MAX_AGE_YEARS behind the report date, fails the run.
"""

import io
import warnings

import pandas as pd
import requests

from previous_release import previous_release_files
from data_definitions import (
    API_TIMEOUT,
    BITCOIN_OWNER_ESTIMATES,
    BITCOIN_OWNERS_AS_OF,
    EARLY_INTERNET_USERS,
    EARLY_INTERNET_USERS_URL,
    FRED_MEDIAN_INCOME_URL,
    WORLD_BANK_INDICATOR_URL,
    WORLD_BANK_INDICATORS,
)

ANNUAL_REFERENCE_FILE = "annual_reference_data.csv"
COLUMNS = ["series", "year", "value", "unit", "source", "source_url", "note", "retrieved_date"]

# FRED publishes median income about nine months after the year ends, so the newest
# year is normally one or two years back.
MAX_AGE_YEARS = 3

# Published series: (unit, source, source_url).
ANNUAL_SERIES = {
    "us_median_household_income_usd": (
        "current USD", "FRED / U.S. Census Bureau",
        "https://fred.stlouisfed.org/series/MEHOINUSA646N",
    ),
    "world_internet_users_pct": (
        "% of population", "World Bank (ITU)",
        "https://data.worldbank.org/indicator/IT.NET.USER.ZS",
    ),
    "world_population": (
        "people", "World Bank", "https://data.worldbank.org/indicator/SP.POP.TOTL",
    ),
    "world_internet_users": ("people", "Our World in Data (ITU)", EARLY_INTERNET_USERS_URL),
    # Each owner estimate links its own report.
    "bitcoin_owners_millions": ("millions of people", "Crypto.com", ""),
}


class SourceFormatError(ValueError):
    """The source answered, but not in the shape this module reads: fix the code."""


def _checked(frame: pd.DataFrame, series: str) -> pd.DataFrame:
    """Integer years, each once, with positive finite values."""
    frame = frame.dropna(subset=["value"]).astype({"year": int, "value": float})
    if frame.empty or frame["year"].duplicated().any() or not frame["value"].gt(0).all():
        raise SourceFormatError(f"{series}: empty, duplicated or non-positive values")
    if series.endswith("_pct") and frame["value"].gt(100).any():
        raise SourceFormatError(f"{series}: a percentage above 100")
    return frame.sort_values("year")


def _fred_median_income() -> pd.DataFrame:
    response = requests.get(FRED_MEDIAN_INCOME_URL, timeout=API_TIMEOUT)
    response.raise_for_status()
    # FRED serves an HTML page during maintenance: an outage, not a format change.
    if response.text.lstrip().startswith("<"):
        raise requests.RequestException("FRED returned a web page instead of CSV")
    frame = pd.read_csv(io.StringIO(response.text))
    if not {"observation_date", "MEHOINUSA646N"}.issubset(frame.columns):
        raise SourceFormatError("FRED median income: unexpected columns")
    dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    if dates.isna().any():
        raise SourceFormatError("FRED median income: invalid observation dates")
    values = pd.to_numeric(frame["MEHOINUSA646N"], errors="coerce")
    return pd.DataFrame({"year": dates.dt.year, "value": values})


def _world_bank(code: str) -> pd.DataFrame:
    response = requests.get(
        WORLD_BANK_INDICATOR_URL.format(code=code),
        params={"format": "json", "per_page": 20000},
        timeout=API_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise SourceFormatError(f"World Bank {code}: unexpected response {str(payload)[:200]}")
    if payload[0].get("pages", 1) != 1:
        raise SourceFormatError(f"World Bank {code}: response spans more than one page")
    return pd.DataFrame(
        {"year": [int(row["date"]) for row in payload[1]],
         "value": pd.to_numeric([row["value"] for row in payload[1]], errors="coerce")}
    )


FETCHED_SERIES = {
    "us_median_household_income_usd": _fred_median_income,
    **{series: (lambda code=code: _world_bank(code)) for code, series in WORLD_BANK_INDICATORS.items()},
}


def _previous_release() -> pd.DataFrame:
    """The last published annual file, checked against its release manifest."""
    content = previous_release_files([ANNUAL_REFERENCE_FILE]).get(ANNUAL_REFERENCE_FILE)
    if content is None:
        raise RuntimeError(f"the last release has no {ANNUAL_REFERENCE_FILE}")
    return pd.read_csv(io.BytesIO(content), keep_default_na=False, na_values=[""])


def _rows(series: str, frame: pd.DataFrame, retrieved_date: str) -> pd.DataFrame:
    """A series' checked rows with its metadata; per-row note or source_url columns win."""
    unit, source, url = ANNUAL_SERIES[series]
    frame = _checked(frame, series)
    defaults = {"note": "", "source_url": url}
    return frame.assign(
        series=series, unit=unit, source=source, retrieved_date=retrieved_date,
        **{column: value for column, value in defaults.items() if column not in frame},
    )


def get_annual_reference_data(report_date, previous_release=_previous_release) -> pd.DataFrame:
    """Every annual series in long format; see the module docstring."""
    report_date = pd.Timestamp(report_date)
    today = pd.Timestamp.now(tz="UTC").date().isoformat()
    previous = None
    parts = []
    for series, fetch in FETCHED_SERIES.items():
        try:
            part = _rows(series, fetch(), today)
        except SourceFormatError:
            raise
        except (requests.RequestException, ValueError) as error:
            if previous is None:
                previous = previous_release()
            part = previous[previous["series"] == series]
            if part.empty:
                raise RuntimeError(f"{series}: fetch failed ({error}) and the last release has no copy") from error
            warnings.warn(
                f"{series}: fetch failed ({error}); reusing the last release's rows, "
                f"retrieved {part['retrieved_date'].iloc[0]}",
                RuntimeWarning, stacklevel=2,
            )
        newest = int(part["year"].max())
        if report_date.year - newest > MAX_AGE_YEARS:
            raise RuntimeError(f"{series}: newest year {newest} is more than {MAX_AGE_YEARS} years old")
        parts.append(part)

    parts.append(_rows("bitcoin_owners_millions", BITCOIN_OWNER_ESTIMATES,
                       BITCOIN_OWNERS_AS_OF.date().isoformat()))
    parts.append(_rows("world_internet_users", EARLY_INTERNET_USERS, ""))

    return pd.concat(parts, ignore_index=True)[COLUMNS].sort_values(
        ["series", "year"], kind="stable"
    ).reset_index(drop=True)
