"""Fetch every upstream source and merge them onto the BRK daily calendar.

Sources:
    - BRK (Bitview): on-chain series and daily OHLC candles
    - Yahoo Finance: equity, ETF, index, commodity and dollar-index closes, and historical
      stock market caps (close x shares outstanding)
    - Google Sheets: Coin Metrics monthly miner efficiency

Market fetchers keep each value's true observation date in a temporary column so the
freshness checks can measure real source age rather than trusting a carried-forward value.
"""

import csv
import io
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from yfinance.exceptions import YFException

from data_definitions import (
    API_TIMEOUT,
    BRK_BULK_URL,
    BRK_METRICS,
    MARKET_CAP_HISTORY_START_DATE,
    MINER_DATA_SHEET_URL,
    YAHOO_MARKET_CAP_FX_TICKERS,
    YAHOO_SHARE_TICKER_ALIASES,
)
from data_validation import OHLC_COLUMNS, assert_ohlc_usable, validate_calendar


# Ordinary market feeds should bridge weekends and short exchange holidays, not outages.
# Five calendar days covers those expected gaps while ensuring a stalled source becomes NaN.
MARKET_DATA_MAX_FFILL_DAYS = 5


# Yahoo publishes shares outstanding on each issuer's filing cadence, not daily, so this
# series needs its own budget rather than the ordinary market one. Observed source ages
# across the tracked tickers run from same-day (NVDA, MU) to 162 days (2222.SR, which
# files semi-annually); 220 days clears a semi-annual filer plus its lag while still
# refusing a share count that has gone quiet for the better part of a year. A share count
# carried indefinitely silently understates market cap by the issuer's dilution since.
SHARES_OUTSTANDING_MAX_AGE_DAYS = 220


MINER_EFFICIENCY_VALUE_COLUMN = "cm_efficiency_j_gh"


MINER_EFFICIENCY_SOURCE_DATE_COLUMN = "cm_efficiency_source_date"


MINER_EFFICIENCY_SOURCE_URL_COLUMN = "cm_efficiency_source_url"


MINER_EFFICIENCY_COLUMNS = [
    MINER_EFFICIENCY_VALUE_COLUMN,
    MINER_EFFICIENCY_SOURCE_DATE_COLUMN,
    MINER_EFFICIENCY_SOURCE_URL_COLUMN,
]




BRK_BULK_MAX_ATTEMPTS = 3


BRK_BULK_INITIAL_BACKOFF_SECONDS = 1.0


BRK_SEMANTIC_ERROR_CODES = {
    "weight_exceeded",
    "series_not_found",
    "metric_not_found",
}


# Temporary provenance columns survive source reindexing and the merge into the BRK
# calendar. `forward_fill_market_data` uses them to enforce total source age, then removes
# them so the published schema is unchanged. Miner provenance is intentionally retained.
_SOURCE_OBSERVATION_DATE_PREFIX = "__source_observation_date__"

# Failures Yahoo and its transport raise for unavailable or malformed data. A coding error
# (AttributeError, NameError, ...) is deliberately not caught, so it fails the run instead
# of looking like a ticker with no data.
YAHOO_DATA_ERRORS = (
    YFException, requests.RequestException, ValueError, KeyError, IndexError, TypeError, OSError,
)


def _source_observation_column(value_column: str) -> str:
    return f"{_SOURCE_OBSERVATION_DATE_PREFIX}{value_column}"


def get_brk_ohlc(start: str = "2009-01-03") -> pd.DataFrame:
    """
    Fetch historical Bitcoin OHLC data from BRK.

    The pipeline fetches daily candles only; weekly and monthly candles are aggregated
    from them (candle_data.period_candles) so every period is cut off at the report date.

    Parameters:
    start (str): Start date for the series query.

    Returns:
    pd.DataFrame: DataFrame indexed by BRK date labels with Open, High, Low, Close columns.
    """
    base_url = "https://bitview.space/api/series"
    index = "day1"
    params = {"start": start}

    try:
        date_response = requests.get(
            f"{base_url}/date/{index}", params=params, timeout=API_TIMEOUT
        )
        date_response.raise_for_status()

        ohlc_response = requests.get(
            f"{base_url}/price_ohlc/{index}", params=params, timeout=API_TIMEOUT
        )
        ohlc_response.raise_for_status()

        date_payload, candle_payload = date_response.json(), ohlc_response.json()
        dates, ohlc_rows = date_payload["data"], candle_payload["data"]
        if not dates or not ohlc_rows:
            raise ValueError(f"BRK returned no {index} OHLC observations for start={start}")
        for payload in (date_payload, candle_payload):
            if (payload.get("index") != index
                    or type(payload.get("start")) is not int
                    or type(payload.get("end")) is not int
                    or payload["end"] - payload["start"] != len(payload["data"])):
                raise ValueError("BRK OHLC metadata does not match the requested index/range")
        if any(date_payload[key] != candle_payload[key] for key in ("start", "end")):
            raise ValueError("BRK date/OHLC metadata ranges differ")

        if len(dates) != len(ohlc_rows):
            raise ValueError(
                f"BRK date/OHLC length mismatch: {len(dates)} dates vs {len(ohlc_rows)} rows"
            )

        df = pd.DataFrame(ohlc_rows, columns=OHLC_COLUMNS)
        df["Time"] = pd.to_datetime(dates)
        df.set_index("Time", inplace=True)
        df = df.astype(float)
        validate_calendar(df.index, f"BRK {index} OHLC")
        # BRK's pre-market history is all-zero; internal invalid candles still fail.
        nonzero = df.ne(0).any(axis=1)
        df = df.loc[nonzero.idxmax():] if nonzero.any() else df.iloc[:0]
        assert_ohlc_usable(df, label=f"BRK {index} OHLC")
        return df

    except (requests.RequestException, KeyError, TypeError, ValueError) as e:
        raise RuntimeError(
            f"Failed to fetch usable BRK {index} OHLC data from start={start}: {e}"
        ) from e


def get_price(tickers: dict, start_date: str) -> pd.DataFrame:
    """
    Fetches historical close prices for all tickers using a single yf.download() batch call.

    Batching all tickers into one request is significantly faster than fetching each ticker
    individually.

    Parameters:
    tickers (dict): Dictionary with categories as keys and ticker lists as values.
    start_date (str): Start date for fetching historical data (format: 'YYYY-MM-DD').

    Returns:
    pd.DataFrame: DataFrame containing close prices for all tickers with 'time' column.
    """
    # Anchor on the UTC clock, not the local one, so a local run and a CI run request
    # the same window. yfinance treats `end` as exclusive, so this asks for everything
    # through the current UTC day.
    end_date = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None).strftime(
        "%Y-%m-%d"
    )
    fetch_tickers = [ticker for ticker_list in tickers.values() for ticker in ticker_list]

    if not fetch_tickers:
        return pd.DataFrame(columns=["time"])

    # Continuous daily index for reindexing (fills weekends/holidays via ffill)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Single batch download — orders of magnitude faster than per-ticker loop
    try:
        raw = yf.download(
            fetch_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except YAHOO_DATA_ERRORS as e:
        warnings.warn(f"Yahoo batch price download failed: {e}", RuntimeWarning, stacklevel=2)
        return pd.DataFrame(columns=["time"])

    data_frames = []
    for ticker in fetch_tickers:
        try:
            close_series = raw[ticker]["Close"]
            if close_series.isna().all():
                warnings.warn(f"Yahoo returned no prices for {ticker}", RuntimeWarning, stacklevel=2)
                continue
            value_column = f"{ticker}_close"
            col = close_series.rename(value_column).to_frame()
            col[_source_observation_column(value_column)] = pd.Series(
                col.index, index=col.index
            ).where(col[value_column].notna())
            # Index is already tz-naive with auto_adjust=True; reindex to fill gaps
            col = col.reindex(date_range).ffill(limit=MARKET_DATA_MAX_FFILL_DAYS)
            data_frames.append(col)
        except KeyError:
            warnings.warn(f"{ticker} is missing from the Yahoo batch result", RuntimeWarning, stacklevel=2)

    if not data_frames:
        return pd.DataFrame(columns=["time"])

    # Consolidate the many per-ticker blocks before adding the time column. Without
    # the copy, reset_index has to insert into a highly fragmented frame and pandas
    # emits a PerformanceWarning on every full pipeline run.
    data = pd.concat(data_frames, axis=1).copy().reset_index()
    data.rename(columns={"index": "time"}, inplace=True)
    data["time"] = pd.to_datetime(data["time"]).dt.tz_localize(None)
    return data


def _normalize_yahoo_series(values: pd.Series) -> pd.Series:
    """Return a numeric Yahoo series on timezone-naive calendar dates."""
    values = pd.Series(values).copy()
    index = pd.DatetimeIndex(pd.to_datetime(values.index))
    if index.tz is not None:
        # Preserve Yahoo's exchange-local date. tz_convert(None) can move midnight to the
        # prior/next date, which is especially harmful on a split effective date.
        index = index.tz_localize(None)
    values.index = index.normalize()
    return pd.to_numeric(values, errors="coerce").dropna()


def _select_yahoo_share_observations(
    shares: pd.Series, stock_splits: pd.Series
) -> pd.Series:
    """Collapse Yahoo's duplicate share observations without choosing a wrong split basis."""
    shares = _normalize_yahoo_series(shares)
    shares = shares[shares > 0]
    split_events = _normalize_yahoo_series(stock_splits)
    split_events = split_events[(split_events > 0) & (split_events != 1)]
    split_events = split_events.groupby(level=0).prod()

    selected = {}
    for observation_date, observations in shares.groupby(level=0, sort=True):
        candidates = observations.to_numpy(dtype=float)
        split_ratio = split_events.get(observation_date)
        if split_ratio is not None and selected:
            previous_value = next(reversed(selected.values()))
            target = previous_value * float(split_ratio)
            positive = candidates[candidates > 0]
            distances = np.abs(np.log(positive / target))
            selected[observation_date] = float(positive[np.argmin(distances)])
        else:
            # Alias histories are concatenated old ticker first and current ticker last, so
            # the current ticker wins on a non-split overlap date.
            selected[observation_date] = float(candidates[-1])

    return pd.Series(selected, dtype="float64").sort_index()


def _drop_isolated_yahoo_share_outliers(shares: pd.Series) -> pd.Series:
    """Remove a one-observation share spike/dip when both neighbors agree closely."""
    if len(shares) < 3:
        return shares

    log_shares = np.log(shares)
    previous = log_shares.shift(1)
    following = log_shares.shift(-1)
    isolated = (
        ((log_shares - previous).abs() > np.log(1.20))
        & ((log_shares - following).abs() > np.log(1.20))
        & ((following - previous).abs() < np.log(1.10))
    )
    return shares[~isolated]


def _split_adjust_yahoo_shares(
    shares: pd.Series, stock_splits: pd.Series, search_days: int = 60
) -> pd.Series:
    """Put as-reported shares on the split basis used by Yahoo's Close series."""
    split_events = _normalize_yahoo_series(stock_splits)
    split_events = split_events[(split_events > 0) & (split_events != 1)]
    split_events = split_events.groupby(level=0).prod().sort_index()
    shares = _select_yahoo_share_observations(shares, split_events)
    shares = _drop_isolated_yahoo_share_outliers(shares)
    if shares.empty or split_events.empty:
        return shares

    adjusted = shares.copy()
    for split_date, split_ratio in split_events.items():
        ratios = shares / shares.shift(1)
        window = ratios[
            (ratios.index >= split_date - pd.Timedelta(days=search_days))
            & (ratios.index <= split_date + pd.Timedelta(days=search_days))
        ].dropna()

        transition_date = split_date
        if not window.empty:
            distances = np.abs(np.log(window / float(split_ratio)))
            candidate_date = distances.idxmin()
            candidate_ratio = float(window.loc[candidate_date])
            if abs(candidate_ratio / float(split_ratio) - 1.0) <= 0.35:
                transition_date = candidate_date

        adjusted.loc[adjusted.index < transition_date] *= float(split_ratio)

    return adjusted


def _current_yahoo_market_cap(stock) -> Optional[float]:
    """Return Yahoo's current scalar cap only as a last-date availability fallback."""
    market_cap = None
    try:
        market_cap = stock.fast_info.get("market_cap")
    except YAHOO_DATA_ERRORS:
        market_cap = None
    if market_cap is None:
        try:
            market_cap = stock.info.get("marketCap")
        except YAHOO_DATA_ERRORS:
            market_cap = None
    try:
        market_cap = float(market_cap)
    except (TypeError, ValueError):
        return None
    return market_cap if np.isfinite(market_cap) and market_cap > 0 else None


def _yahoo_close(history: pd.DataFrame) -> pd.Series:
    """Positive daily closes from a Yahoo history frame, one per exchange-local date."""
    close = _normalize_yahoo_series(history["Close"])
    return close[close > 0].groupby(level=0, sort=True).last()


def _yahoo_shares(stock, ticker: str, stock_splits: pd.Series, price_timezone,
                  fetch_start: str, fetch_end: str) -> pd.Series:
    """Split-adjusted historical shares outstanding, stitched across renamed tickers."""
    share_parts = []
    for share_symbol in YAHOO_SHARE_TICKER_ALIASES.get(ticker, [ticker]):
        share_stock = stock if share_symbol == ticker else yf.Ticker(share_symbol)
        # Retired tickers can retain fundamentals while losing chart timezone
        # metadata. Seed them from the current ticker before requesting shares.
        if (
            share_symbol != ticker
            and price_timezone is not None
            and getattr(share_stock, "_tz", None) is None
        ):
            share_stock._tz = str(price_timezone)
        try:
            share_history = share_stock.get_shares_full(start=fetch_start, end=fetch_end)
        except YAHOO_DATA_ERRORS as exc:
            warnings.warn(
                f"Yahoo shares unavailable for {share_symbol}: {exc}", RuntimeWarning, stacklevel=2
            )
            continue
        if share_history is not None and len(share_history) > 0:
            share_parts.append(_normalize_yahoo_series(share_history))
    if not share_parts:
        raise ValueError("Yahoo returned no historical shares outstanding")
    return _split_adjust_yahoo_shares(pd.concat(share_parts), stock_splits)


def _fresh_shares_on_price_dates(shares: pd.Series, close: pd.Series) -> pd.Series:
    """Shares carried onto each price date, blank once the last filing is past its budget.

    The true filing date travels with the value so staleness is measured from the
    observation, never inferred from a repeated daily figure. Beyond the budget the market
    cap becomes NaN rather than silently understating the issuer's dilution.
    """
    combined_index = shares.index.union(close.index).sort_values()
    on_price_dates = shares.reindex(combined_index).ffill().reindex(close.index)
    source_dates = (
        pd.Series(shares.index, index=shares.index)
        .reindex(combined_index)
        .ffill()
        .reindex(close.index)
    )
    row_dates = pd.Series(close.index, index=close.index).dt.normalize()
    share_age_days = (row_dates - source_dates.dt.normalize()).dt.days
    return on_price_dates.where(share_age_days.between(0, SHARES_OUTSTANDING_MAX_AGE_DAYS))


def _yahoo_fx_close(fx_symbol: str, fetch_start: str, fetch_end: str) -> pd.Series:
    """USD conversion closes for a non-USD listing."""
    fx_history = yf.Ticker(fx_symbol).history(start=fetch_start, end=fetch_end, auto_adjust=False)
    if fx_history.empty or "Close" not in fx_history:
        raise ValueError(f"Yahoo returned no {fx_symbol} USD conversion history")
    return _yahoo_close(fx_history)


def _historical_market_cap(stock, ticker: str, fetch_start: str, fetch_end: str,
                           requested_end: pd.Timestamp) -> tuple:
    """Close x split-adjusted shares for one ticker, converted to USD when needed.

    Returns (market_cap series, fx closes or None).
    """
    history = stock.history(start=fetch_start, end=fetch_end, auto_adjust=False, actions=True)
    if history.empty or "Close" not in history:
        raise ValueError("Yahoo returned no closing-price history")
    close = _yahoo_close(history)
    stock_splits = history["Stock Splits"] if "Stock Splits" in history else pd.Series(dtype="float64")
    shares = _yahoo_shares(
        stock, ticker, stock_splits, getattr(history.index, "tz", None), fetch_start, fetch_end
    )

    newest_share_date = shares.index.max()
    newest_age = (requested_end.normalize() - newest_share_date.normalize()).days
    if newest_age > SHARES_OUTSTANDING_MAX_AGE_DAYS:
        warnings.warn(
            f"Shares outstanding for {ticker} last observed {newest_share_date.date()} "
            f"({newest_age} days before {requested_end.date()}); {ticker}_MarketCap will be "
            f"null past the {SHARES_OUTSTANDING_MAX_AGE_DAYS}-day budget",
            RuntimeWarning,
            stacklevel=2,
        )

    market_cap = (close * _fresh_shares_on_price_dates(shares, close)).replace(
        [np.inf, -np.inf], np.nan
    )
    fx_close = None
    fx_symbol = YAHOO_MARKET_CAP_FX_TICKERS.get(ticker)
    if fx_symbol:
        fx_close = _yahoo_fx_close(fx_symbol, fetch_start, fetch_end)
        fx_index = fx_close.index.union(close.index).sort_values()
        fx_on_price_dates = (
            fx_close.reindex(fx_index).ffill(limit=MARKET_DATA_MAX_FFILL_DAYS).reindex(close.index)
        )
        market_cap = market_cap * fx_on_price_dates
    return market_cap.dropna(), fx_close


def _live_market_cap_fallback(stock, ticker: str, fx_close, requested_end: pd.Timestamp):
    """Yahoo's live scalar market cap for the final date only, in USD, or None."""
    current = _current_yahoo_market_cap(stock)
    if current is None or not YAHOO_MARKET_CAP_FX_TICKERS.get(ticker):
        return current
    if fx_close is None or fx_close.empty:
        return None
    recent_fx = fx_close.loc[
        fx_close.index >= requested_end - pd.Timedelta(days=MARKET_DATA_MAX_FFILL_DAYS)
    ]
    return current * float(recent_fx.iloc[-1]) if not recent_fx.empty else None


def get_marketcap(
    tickers: dict, start_date: str, end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Build Yahoo-only historical stock market caps as Close times shares outstanding.

    The existing `TICKER_MarketCap` schema is retained. Values remain null before Yahoo's
    first historical share observation, and renamed ticker histories are stitched under the
    current ticker. Non-USD listings are converted with Yahoo's historical FX close before
    publication. A current Yahoo scalar is used only on the final requested date if the
    historical calculation is unavailable; it is never broadcast backward.
    """
    stocks = list(tickers.get("stocks", []))
    requested_start = pd.to_datetime(start_date).normalize()
    requested_end = (
        pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=1)
        if end_date is None
        else pd.to_datetime(end_date).normalize()
    )
    if requested_end < requested_start:
        raise ValueError("Market-cap end_date cannot be before start_date")

    calendar = pd.date_range(requested_start, requested_end, freq="D", name="time")
    data = pd.DataFrame(index=calendar)
    for ticker in stocks:
        data[f"{ticker}_MarketCap"] = np.nan

    history_start = max(requested_start, pd.to_datetime(MARKET_CAP_HISTORY_START_DATE).normalize())
    if history_start > requested_end:
        return data.reset_index()
    fetch_start = history_start.strftime("%Y-%m-%d")
    # yfinance treats `end` as exclusive.
    fetch_end = (requested_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in stocks:
        value_column = f"{ticker}_MarketCap"
        stock = yf.Ticker(ticker)
        fx_close = None
        try:
            market_cap, fx_close = _historical_market_cap(
                stock, ticker, fetch_start, fetch_end, requested_end
            )
            valid_dates = market_cap.index.intersection(data.index)
            data.loc[valid_dates, value_column] = market_cap.loc[valid_dates].astype(float)
        except YAHOO_DATA_ERRORS as exc:
            warnings.warn(
                f"Could not build Yahoo historical market cap for {ticker}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

        if not data[value_column].notna().any():
            current_market_cap = _live_market_cap_fallback(stock, ticker, fx_close, requested_end)
            if current_market_cap is not None:
                # A live intraday quote in a column otherwise made of settled daily
                # closes. It is confined to the final row and never broadcast backward,
                # but it should not pass unremarked in the run log.
                warnings.warn(
                    f"{value_column} on {requested_end.date()} is Yahoo's live scalar "
                    "market cap, not a settled close: no historical share series was "
                    "available for this ticker",
                    RuntimeWarning,
                    stacklevel=2,
                )
                data.loc[requested_end, value_column] = current_market_cap

    return data.reset_index()


def get_miner_data(google_sheet_url: str = "") -> pd.DataFrame:
    """
    Fetch Coin Metrics monthly Bitcoin network efficiency data from Google Sheets.

    The Google Sheet is expected to contain monthly observations with:
        - time: Month timestamp
        - cm_efficiency_j_gh: Coin Metrics estimated Bitcoin network efficiency in J/GH

    If the sheet instead contains `efficiency_j_th`, this function converts it to
    `cm_efficiency_j_gh` by dividing by 1,000.

    The monthly series is forward-filled to daily frequency so it can be merged
    with daily BRK/on-chain data.

    Parameters:
    google_sheet_url (str): Google Sheets URL to extract data from.
                            Defaults to MINER_DATA_SHEET_URL from config.

    Each daily row retains the date of the actual monthly source observation and the
    configured sheet export URL. Those fields let freshness validation distinguish a
    proven observation from a value repeated merely for daily alignment.

    The latest monthly observation is carried forward until the sheet publishes a new
    value. Its original observation date and URL remain attached so the pipeline can
    warn clearly when that estimate is older than the normal monthly update cadence.

    Returns:
    pd.DataFrame: Daily DataFrame with `time`, `cm_efficiency_j_gh`,
                  `cm_efficiency_source_date`, and `cm_efficiency_source_url`.
                  Returns an empty DataFrame with that schema on error.
    """
    if not google_sheet_url:
        google_sheet_url = MINER_DATA_SHEET_URL

    try:
        # Convert Google Sheets sharing URL to CSV export URL.
        csv_export_url = google_sheet_url.replace("/edit?usp=sharing", "/export?format=csv")
        csv_export_url = csv_export_url.split("#")[0]
        if "/edit?" in csv_export_url:
            csv_export_url = csv_export_url.split("/edit?")[0] + "/export?format=csv"

        response = requests.get(csv_export_url, timeout=API_TIMEOUT)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = [str(col).strip() for col in df.columns]

        if "time" not in df.columns:
            raise ValueError("Miner efficiency sheet must contain a `time` column")

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")

        if "cm_efficiency_j_gh" not in df.columns:
            if "efficiency_j_th" not in df.columns:
                raise ValueError(
                    "Miner efficiency sheet must contain either `cm_efficiency_j_gh` or `efficiency_j_th`"
                )
            df["cm_efficiency_j_gh"] = pd.to_numeric(
                df["efficiency_j_th"], errors="coerce"
            ) / 1000
        else:
            df["cm_efficiency_j_gh"] = pd.to_numeric(
                df["cm_efficiency_j_gh"], errors="coerce"
            )

        df = df.dropna(subset=["cm_efficiency_j_gh"])
        df[MINER_EFFICIENCY_SOURCE_DATE_COLUMN] = df["time"].dt.normalize()
        df[MINER_EFFICIENCY_SOURCE_URL_COLUMN] = csv_export_url
        df = df[["time"] + MINER_EFFICIENCY_COLUMNS]
        df = df.drop_duplicates(subset=["time"], keep="last")

        # Monthly Coin Metrics efficiency is the best available estimate for each day
        # until the next monthly observation. Provenance is carried with the value so
        # an old estimate remains visible rather than masquerading as a new observation.
        df = df.set_index("time").resample("D").ffill().reset_index()

        return df
    except (requests.RequestException, ValueError, KeyError, pd.errors.ParserError) as e:
        warnings.warn(
            f"Could not fetch Coin Metrics miner efficiency from Google Sheets: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["time"] + MINER_EFFICIENCY_COLUMNS)


def _brk_error_code(response: requests.Response) -> Optional[str]:
    """Extract a BRK error code from a non-2xx response when available."""
    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, dict):
        return None

    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("code")

    return None


def _brk_fetch_csv(
    metrics,
    index="dateindex",
    start=0,
    timeout=API_TIMEOUT,
    verbose=False,
    max_attempts=BRK_BULK_MAX_ATTEMPTS,
    initial_backoff_seconds=BRK_BULK_INITIAL_BACKOFF_SECONDS,
):
    """
    Fetch series from BRK bulk API as CSV.

    Parameters:
    metrics (list): List of series names to fetch.
    index (str): Index type for the API request.
    start (int | str): Starting range bound for data retrieval.
    timeout (int): Request timeout in seconds. Defaults to the shared `API_TIMEOUT`.
    verbose (bool): If True, print debug information.
    max_attempts (int): Bounded total attempts for transient failures.
    initial_backoff_seconds (float): First exponential-backoff delay.

    Returns:
    tuple: (header, data_rows) - CSV header and data rows.
    """
    if verbose:
        print(f"[BRK] fetching {len(metrics)} metrics: {metrics}")

    if max_attempts < 1:
        raise ValueError("BRK max_attempts must be at least 1")

    request_params = {
        "series": ",".join(metrics),
        "index": index,
        "start": start,
        "format": "csv",
    }
    r = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(
                BRK_BULK_URL,
                params=request_params,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt >= max_attempts:
                raise
            delay = initial_backoff_seconds * (2 ** (attempt - 1))
            if verbose:
                print(
                    f"[BRK] transient connection failure on attempt {attempt}/{max_attempts}: "
                    f"{exc}; retrying in {delay:g}s"
                )
            time.sleep(delay)
            continue

        code = _brk_error_code(r) if not r.ok else None
        transient_status = r.status_code == 429 or 500 <= r.status_code <= 599
        # Semantic errors need recursive splitting or explicit missing-series handling;
        # retrying the identical oversized/invalid request would only delay that path.
        should_retry = transient_status and code not in BRK_SEMANTIC_ERROR_CODES
        if should_retry and attempt < max_attempts:
            delay = initial_backoff_seconds * (2 ** (attempt - 1))
            if verbose:
                print(
                    f"[BRK] transient HTTP {r.status_code} on attempt "
                    f"{attempt}/{max_attempts}; retrying in {delay:g}s"
                )
            time.sleep(delay)
            continue
        break

    # The loop either obtained a response or re-raised the final connection exception.
    assert r is not None

    if verbose:
        print(f"[BRK] status={r.status_code} bytes={len(r.text)}")

    if not r.ok:
        code = _brk_error_code(r)
        message = f"[BRK] request failed status={r.status_code}"
        if code:
            message += f" code={code}"
        if r.text:
            snippet = r.text[:300].replace("\n", " ")
            message += f" body={snippet}"
        raise requests.HTTPError(message, response=r)

    rows = list(csv.reader(io.StringIO(r.text)))
    if not rows:
        raise ValueError("[BRK] Empty CSV response")

    header = rows[0]
    data_rows = rows[1:]

    if verbose:
        print(f"[BRK] header: {header[:8]}{' ...' if len(header) > 8 else ''}")
        print(f"[BRK] rows: {len(data_rows)}")
        if data_rows:
            print(
                f"[BRK] first row sample: {data_rows[0][:8]}{' ...' if len(data_rows[0]) > 8 else ''}"
            )

    return header, data_rows


def _brk_fetch_csv_resilient(
    metrics,
    index="dateindex",
    start=0,
    timeout=API_TIMEOUT,
    verbose=False,
    missing=None,
    max_attempts=BRK_BULK_MAX_ATTEMPTS,
    initial_backoff_seconds=BRK_BULK_INITIAL_BACKOFF_SECONDS,
):
    """
    Fetch a BRK bulk CSV request, recursively splitting oversized or invalid chunks.

    Parameters:
    missing (list | None): If provided, names of series BRK could not resolve are
                           appended here so the caller can fail loudly rather than
                           silently publishing an all-NaN column.

    Returns:
    list[tuple]: One or more (header, rows) responses.
    """
    try:
        return [
            _brk_fetch_csv(
                metrics,
                index=index,
                start=start,
                timeout=timeout,
                verbose=verbose,
                max_attempts=max_attempts,
                initial_backoff_seconds=initial_backoff_seconds,
            )
        ]
    except requests.HTTPError as exc:
        response = exc.response
        code = _brk_error_code(response) if response is not None else None
        non_ts = [metric for metric in metrics if metric != "timestamp"]

        if code in BRK_SEMANTIC_ERROR_CODES and len(non_ts) > 1:
            midpoint = len(non_ts) // 2
            left = ["timestamp"] + non_ts[:midpoint]
            right = ["timestamp"] + non_ts[midpoint:]

            if verbose:
                print(f"[BRK] splitting chunk ({code}): {non_ts}")

            return (
                _brk_fetch_csv_resilient(
                    left,
                    index=index,
                    start=start,
                    timeout=timeout,
                    verbose=verbose,
                    missing=missing,
                    max_attempts=max_attempts,
                    initial_backoff_seconds=initial_backoff_seconds,
                )
                + _brk_fetch_csv_resilient(
                    right,
                    index=index,
                    start=start,
                    timeout=timeout,
                    verbose=verbose,
                    missing=missing,
                    max_attempts=max_attempts,
                    initial_backoff_seconds=initial_backoff_seconds,
                )
            )

        if code in {"series_not_found", "metric_not_found"} and len(non_ts) == 1:
            if verbose:
                print(f"[BRK] missing series: {non_ts[0]}")
            if missing is not None:
                missing.append(non_ts[0])
            return []

        raise


def get_brk_onchain(
    start_date: str,
    index: str = "dateindex",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Pull BRK metrics, align by timestamp (included in every chunk), and return a pandas
    DataFrame with a 'time' column using native BRK field names.

    BRK also returns the partial, in-progress UTC day. The returned frame is not truncated;
    main.py applies the report-date cutoff to its own exports.
    """

    metric_list = BRK_METRICS[:]  # copy
    if "timestamp" not in metric_list:
        metric_list = ["timestamp"] + metric_list

    # Query from the requested start date instead of genesis to reduce BRK request weight.
    query_start = start_date

    # Start with reasonably sized chunks; resilient fetcher splits again if needed.
    chunk_size = 8
    non_ts = [m for m in metric_list if m != "timestamp"]
    chunks = [
        ["timestamp"] + non_ts[i : i + chunk_size]
        for i in range(0, len(non_ts), chunk_size)
    ]

    data = {}
    ordered_cols = ["timestamp"]

    missing_series = []

    for chunk in chunks:
        responses = _brk_fetch_csv_resilient(
            chunk,
            index=index,
            start=query_start,
            timeout=API_TIMEOUT,
            verbose=verbose,
            missing=missing_series,
        )

        for header, rows in responses:
            if not header or header[0] != "timestamp" or len(set(header)) != len(header):
                raise RuntimeError("BRK bulk: invalid or duplicate headers")
            seen_timestamps = set()
            for r in rows:
                if len(r) != len(header) or not r[0] or r[0] in seen_timestamps:
                    raise RuntimeError("BRK bulk: malformed row or duplicate timestamp")
                ts = r[0]
                seen_timestamps.add(ts)
                d = data.setdefault(ts, {"timestamp": ts})
                for k, v in zip(header[1:], r[1:]):
                    if k in d and d[k] != v:
                        raise RuntimeError(f"BRK bulk: conflicting {k} at {ts}")
                    d[k] = v

            for c in header[1:]:
                if c not in ordered_cols:
                    ordered_cols.append(c)

    # A series BRK could not resolve used to be backfilled as an all-NaN column, which
    # then vanished from the fundamentals table via its `len(series) == 0` skip — the
    # report shipped a row short with no error anywhere. Fail loudly instead: a renamed
    # or retired upstream series is a code change, not a data condition.
    returned = set(ordered_cols[1:])
    unresolved = sorted(set(missing_series) | {m for m in non_ts if m not in returned})
    if unresolved:
        raise RuntimeError(
            "BRK did not return the following required series: "
            + ", ".join(unresolved)
            + ". They were likely renamed or retired upstream. Update BRK_METRICS in "
            "data_definitions.py (check https://bitview.space/api/series/list) rather "
            "than publishing a report with missing metrics."
        )

    frame = pd.DataFrame.from_dict(data, orient="index").drop(columns="timestamp")
    frame.index = pd.to_datetime(frame.index.astype(float).astype(int), unit="s")
    frame = frame.sort_index().reindex(columns=ordered_cols[1:])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[frame.index >= pd.to_datetime(start_date)]
    print(f"[BRK] {len(frame)} days x {len(frame.columns)} series through {frame.index.max().date()}")
    return frame.rename_axis("time").reset_index()


def _normalized_index(data: pd.DataFrame) -> pd.DatetimeIndex:
    """Return a tz-naive normalized DatetimeIndex without mutating the input frame."""
    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    if index.tz is not None:
        index = index.tz_convert(None)
    return index.normalize()


def get_data(
    tickers: dict,
    start_date: str,
) -> pd.DataFrame:
    """
    Primary data orchestration function that fetches and merges all data sources into unified dataset.

    This is the main entry point for data ingestion. It fetches every source, normalizes
    timestamps to UTC midnight, and left-joins them onto the BRK daily calendar.

    Data Sources Integrated:
    1. BRK (Bitview) API: Bitcoin price and on-chain metrics
    2. Yahoo Finance: Stock/ETF/index/commodity/forex prices via yfinance
    3. Yahoo Finance: Market capitalizations for public companies
    4. Google Sheets: Monthly Coin Metrics Bitcoin network efficiency data, forward-filled daily (J/GH)

    Parameters:
    tickers (dict): Asset ticker dictionary from data_definitions.py with keys:
                    'stocks', 'etfs', 'indices', 'commodities', 'forex'.
    start_date (str): Historical data start date in 'YYYY-MM-DD' format. Typically '2010-01-01'
                      to capture maximum history from Yahoo Finance. BRK data starts ~2009.
    """
    # Fetch data
    coindata = get_brk_onchain(start_date)
    prices = get_price(tickers, start_date)
    marketcaps = get_marketcap(tickers, start_date)
    miner_data = get_miner_data()  # Monthly Coin Metrics network efficiency, forward-filled daily

    datasets = [
        ("coindata", coindata),
        ("prices", prices),
        ("marketcaps", marketcaps),
        ("miner_data", miner_data),
    ]

    processed_datasets = {}
    for name, dataset in datasets:
        if not dataset.empty and "time" in dataset.columns:
            dataset["time"] = pd.to_datetime(dataset["time"]).dt.tz_localize(None)
            dataset.set_index("time", inplace=True)
            processed_datasets[name] = dataset

    # coindata is the base frame every other source is left-joined onto — it defines the
    # index. Look it up by name: positional access would silently fall through to the
    # next available source and anchor the whole pipeline to yfinance's trading-day index.
    if "coindata" not in processed_datasets:
        raise RuntimeError(
            "BRK on-chain data (coindata) is missing or empty — it is the base frame for "
            "the merged dataset and cannot be substituted. Aborting rather than building "
            "a report on a different index."
        )

    data = processed_datasets["coindata"]
    validate_calendar(data.index, "BRK on-chain data")
    for name, dataset in processed_datasets.items():
        if name == "coindata":
            continue
        # Two sources producing the same column is a configuration error. pandas would
        # rename both copies with _x/_y suffixes and silently change the schema.
        overlap = sorted(set(data.columns) & set(dataset.columns))
        if overlap:
            raise RuntimeError(f"Sources returned duplicate columns: {', '.join(overlap)}")
        data = pd.merge(data, dataset, left_index=True, right_index=True, how="left")

    # Optional assets retain a stable schema when their providers return no data.
    optional = [f"{ticker}_close" for group in tickers.values() for ticker in group]
    optional += [f"{ticker}_MarketCap" for ticker in tickers.get("stocks", [])]
    data = data.reindex(columns=list(dict.fromkeys([*data.columns, *optional])))

    return data
