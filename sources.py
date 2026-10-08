"""Fetch every upstream source and merge them onto the BRK daily calendar.

Sources:
    - BRK (Bitview): on-chain series and daily OHLC candles
    - Yahoo Finance: stock, ETF, index, futures and dollar-index closes, and historical
      stock market caps (close x shares outstanding)
    - Google Sheets: Coin Metrics monthly miner efficiency

The annual reference series are fetched in annual_data.py and the ETF tables built in etf/.

Market fetchers keep each value's real observation date in a temporary column so the
freshness checks measure true source age, not the age of a carried-forward value.
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
    BRK_DAILY_FLOWS,
    BRK_METRICS,
    BRK_PRICE_DEPENDENT_METRICS,
    BRK_REALIZED_PRICE_METRICS,
    MARKET_CAP_HISTORY_START_DATE,
    MINER_DATA_SHEET_URL,
    YAHOO_MARKET_CAP_FX_TICKERS,
    YAHOO_SHARE_TICKER_ALIASES,
)
from data_validation import OHLC_COLUMNS, assert_ohlc_usable, validate_calendar


# Bridges weekends and short exchange holidays; a source stalled longer becomes NaN.
MARKET_DATA_MAX_FFILL_DAYS = 5


# Share counts follow each issuer's filing cadence, not a daily one. 220 days covers a
# semi-annual filer plus its reporting lag (2222.SR runs ~162 days); an older count would
# understate market cap by any dilution since.
SHARES_OUTSTANDING_MAX_AGE_DAYS = 220


# Miner efficiency keeps its source date and URL in the published data.
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
# Errors that retrying the same request cannot fix: the request is split or the series
# reported missing instead.
BRK_SEMANTIC_ERROR_CODES = {
    "weight_exceeded",
    "series_not_found",
    "metric_not_found",
}

# Prefix of the temporary per-column observation dates on market data. They survive the
# merge, drive the freshness checks, and are dropped by `forward_fill_market_data`.
_SOURCE_OBSERVATION_DATE_PREFIX = "__source_observation_date__"

# Errors Yahoo raises for unavailable or malformed data. Coding errors (AttributeError,
# NameError) are not listed, so they fail the run instead of looking like missing data.
YAHOO_DATA_ERRORS = (
    YFException, requests.RequestException, ValueError, KeyError, IndexError, TypeError, OSError,
)


def _source_observation_column(value_column: str) -> str:
    return f"{_SOURCE_OBSERVATION_DATE_PREFIX}{value_column}"


def get_brk_ohlc(start: str = "2009-01-03") -> pd.DataFrame:
    """Daily Bitcoin OHLC candles from BRK, indexed by date ("Time").

    Leading all-zero candles from before Bitcoin had a price are dropped. Weekly and
    monthly candles are built from these in candle_data.
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
        # Drop the all-zero pre-market history; an invalid candle after it still fails.
        nonzero = df.ne(0).any(axis=1)
        df = df.loc[nonzero.idxmax():] if nonzero.any() else df.iloc[:0]
        assert_ohlc_usable(df, label=f"BRK {index} OHLC")
        return df

    except (requests.RequestException, KeyError, TypeError, ValueError) as e:
        raise RuntimeError(
            f"Failed to fetch usable BRK {index} OHLC data from start={start}: {e}"
        ) from e


def get_price(tickers: dict, start_date: str) -> pd.DataFrame:
    """Daily closes for every ticker in `tickers`, in one batched Yahoo download.

    Returns a frame with a `date` column and one `{ticker}_close` column per ticker,
    each with its observation-date marker.
    """
    # Use the UTC date so local and CI runs request the same window. `end` is exclusive,
    # so this requests everything before the current UTC day.
    end_date = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None).strftime(
        "%Y-%m-%d"
    )
    fetch_tickers = [ticker for ticker_list in tickers.values() for ticker in ticker_list]

    if not fetch_tickers:
        return pd.DataFrame(columns=["date"])

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

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
        return pd.DataFrame(columns=["date"])

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
            # Put trading days on the daily calendar, bridging weekends and holidays.
            col = col.reindex(date_range).ffill(limit=MARKET_DATA_MAX_FFILL_DAYS)
            data_frames.append(col)
        except KeyError:
            warnings.warn(f"{ticker} is missing from the Yahoo batch result", RuntimeWarning, stacklevel=2)

    if not data_frames:
        return pd.DataFrame(columns=["date"])

    # copy() consolidates the per-ticker blocks; otherwise reset_index raises pandas'
    # fragmentation PerformanceWarning.
    data = pd.concat(data_frames, axis=1).copy().reset_index()
    data.rename(columns={"index": "date"}, inplace=True)
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    return data


def _normalize_yahoo_series(values: pd.Series) -> pd.Series:
    """Return a numeric Yahoo series on timezone-naive calendar dates."""
    values = pd.Series(values).copy()
    index = pd.DatetimeIndex(pd.to_datetime(values.index))
    if index.tz is not None:
        # Keep the exchange-local date. tz_convert(None) can shift midnight to another
        # day, which misplaces split dates.
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
            # Alias histories are concatenated oldest ticker first, so the current
            # ticker wins a shared date.
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
        # A retired ticker can lack timezone metadata, which the shares request needs.
        # Borrow it from the current ticker.
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
    """Shares carried onto each price date; NaN once the last filing is past its budget.

    Age is measured from the filing date, not from the last row the value was carried to.
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
            f"({newest_age} days before {requested_end.date()}); {ticker}_market_cap will be "
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


def get_market_caps(
    tickers: dict, start_date: str, end_date: Optional[str] = None
) -> pd.DataFrame:
    """Daily `{ticker}_market_cap` in USD: Yahoo close x split-adjusted shares outstanding.

    Values are NaN before Yahoo's first share observation. Renamed tickers are stitched
    under the current ticker, and non-USD listings are converted at the historical FX
    close. If no history can be built, Yahoo's live market cap fills the final date only.
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

    calendar = pd.date_range(requested_start, requested_end, freq="D", name="date")
    data = pd.DataFrame(index=calendar)
    for ticker in stocks:
        data[f"{ticker}_market_cap"] = np.nan

    history_start = max(requested_start, pd.to_datetime(MARKET_CAP_HISTORY_START_DATE).normalize())
    if history_start > requested_end:
        return data.reset_index()
    fetch_start = history_start.strftime("%Y-%m-%d")
    # yfinance treats `end` as exclusive.
    fetch_end = (requested_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in stocks:
        value_column = f"{ticker}_market_cap"
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
                # A live quote among settled closes: final row only, and logged.
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
    """Coin Metrics monthly network efficiency (J/GH) from Google Sheets, filled daily.

    The sheet has `time` and `cm_efficiency_j_gh` columns. Each daily row keeps its source
    observation date and URL so the freshness check can measure the estimate's real age.
    Returns an empty frame with the same columns if the sheet cannot be read.
    """
    if not google_sheet_url:
        google_sheet_url = MINER_DATA_SHEET_URL

    try:
        csv_export_url = google_sheet_url.split("/edit")[0] + "/export?format=csv"

        response = requests.get(csv_export_url, timeout=API_TIMEOUT)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = [str(col).strip() for col in df.columns]

        missing = {"time", MINER_EFFICIENCY_VALUE_COLUMN} - set(df.columns)
        if missing:
            raise ValueError(f"Miner efficiency sheet is missing {sorted(missing)}")

        df["date"] = pd.to_datetime(df["time"], errors="coerce")
        df[MINER_EFFICIENCY_VALUE_COLUMN] = pd.to_numeric(
            df[MINER_EFFICIENCY_VALUE_COLUMN], errors="coerce"
        )
        df = df.dropna(subset=["date", MINER_EFFICIENCY_VALUE_COLUMN]).sort_values("date")
        df[MINER_EFFICIENCY_SOURCE_DATE_COLUMN] = df["date"].dt.normalize()
        df[MINER_EFFICIENCY_SOURCE_URL_COLUMN] = csv_export_url
        df = df[["date"] + MINER_EFFICIENCY_COLUMNS]
        df = df.drop_duplicates(subset=["date"], keep="last")

        df = df.set_index("date").resample("D").ffill().reset_index()

        return df
    except (requests.RequestException, ValueError, KeyError, pd.errors.ParserError) as e:
        warnings.warn(
            f"Could not fetch Coin Metrics miner efficiency from Google Sheets: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["date"] + MINER_EFFICIENCY_COLUMNS)


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
    """Fetch series from the BRK bulk API as CSV; returns (header, rows).

    Connection errors, 429s and 5xx responses are retried with exponential backoff, up to
    `max_attempts` in total.
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

    assert r is not None  # the loop re-raises its final connection error

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
    """Fetch a BRK bulk request, splitting it in half on an oversized or invalid chunk.

    Returns a list of (header, rows) responses. Series BRK cannot resolve are appended
    to `missing` when it is given.
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
    """Every BRK_METRICS series and the BRK_DAILY_FLOWS from `start_date`, with a `date`
    column.

    Series are fetched in chunks and joined on `timestamp`, starting a day early so the
    first day has a flow. The partial current UTC day is included; main.py applies the
    report-date cutoff.
    """
    metric_list = BRK_METRICS[:]
    if "timestamp" not in metric_list:
        metric_list = ["timestamp"] + metric_list

    # Chunks that are still too large are split again by the resilient fetcher.
    chunk_size = 8
    non_ts = [m for m in metric_list if m != "timestamp"]
    chunks = [
        ["timestamp"] + non_ts[i : i + chunk_size]
        for i in range(0, len(non_ts), chunk_size)
    ]

    data = {}
    ordered_cols = ["timestamp"]

    missing_series = []

    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for chunk in chunks:
        responses = _brk_fetch_csv_resilient(
            chunk,
            index=index,
            start=fetch_start,
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

    # A missing series means it was renamed or retired upstream: fail rather than
    # publish an empty column.
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
    frame = _add_daily_flows(frame)
    frame = frame.loc[frame.index >= pd.to_datetime(start_date)]
    frame = _blank_pre_price_placeholders(frame)
    print(f"[BRK] {len(frame)} days x {len(frame.columns)} series through {frame.index.max().date()}")
    return frame.rename_axis("date").reset_index()


# Every column _add_daily_flows adds. They are on-chain series, so the market-data fill
# never touches them.
BRK_DERIVED_SERIES = (*BRK_DAILY_FLOWS, "net_realized_pnl_sum_24h", "nvt", "hash_price_ths")


def _add_daily_flows(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily flows from running totals, and the BRK ratios built on daily flows.

    Net realized P&L, NVT and hash price are recomputed from the calendar-day flows with
    BRK's own formulas, so they carry no rolling-window overcount.
    """
    flows = {name: frame[cumulative].diff() for name, cumulative in BRK_DAILY_FLOWS.items()}
    volume = flows["transfer_volume_sum_24h_usd"]
    flows["net_realized_pnl_sum_24h"] = flows["realized_profit_sum_24h"] - flows["realized_loss_sum_24h"]
    flows["nvt"] = frame["market_cap"] / volume.where(volume > 0)
    flows["hash_price_ths"] = flows["coinbase_sum_24h_usd"] / (frame["hash_rate"] / 1e12)
    return pd.concat([frame, pd.DataFrame(flows, index=frame.index)], axis=1)


def _blank_pre_price_placeholders(frame: pd.DataFrame) -> pd.DataFrame:
    """Blank BRK's 0 placeholders for price-dependent series before Bitcoin had a price,
    and any realized price of 0."""
    frame = frame.copy()
    priced = frame["price_close"].gt(0) if "price_close" in frame else pd.Series(True, frame.index)
    before_price = ~priced.cummax()
    columns = [c for c in BRK_PRICE_DEPENDENT_METRICS if c in frame.columns]
    frame.loc[before_price, columns] = np.nan
    realized = [c for c in BRK_REALIZED_PRICE_METRICS if c in frame.columns]
    frame[realized] = frame[realized].where(frame[realized].ne(0))
    return frame


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
    """Fetch every source and left-join it onto the BRK daily calendar.

    `tickers` is data_definitions.TICKERS; market caps are built for its `stocks` group.
    """
    coindata = get_brk_onchain(start_date)
    prices = get_price(tickers, start_date)
    market_caps = get_market_caps(tickers, start_date)
    miner_data = get_miner_data()

    datasets = [
        ("coindata", coindata),
        ("prices", prices),
        ("market_caps", market_caps),
        ("miner_data", miner_data),
    ]

    processed_datasets = {}
    for name, dataset in datasets:
        if not dataset.empty and "date" in dataset.columns:
            dataset["date"] = pd.to_datetime(dataset["date"]).dt.tz_localize(None)
            dataset.set_index("date", inplace=True)
            processed_datasets[name] = dataset

    # BRK defines the calendar; no other source can stand in for it.
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
        # pandas would rename a shared column to _x/_y, silently changing the schema.
        overlap = sorted(set(data.columns) & set(dataset.columns))
        if overlap:
            raise RuntimeError(f"Sources returned duplicate columns: {', '.join(overlap)}")
        data = pd.merge(data, dataset, left_index=True, right_index=True, how="left")

    # Keep a column for every configured ticker even when Yahoo returned nothing.
    optional = [f"{ticker}_close" for group in tickers.values() for ticker in group]
    optional += [f"{ticker}_market_cap" for ticker in tickers.get("stocks", [])]
    data = data.reindex(columns=list(dict.fromkeys([*data.columns, *optional])))

    return data
