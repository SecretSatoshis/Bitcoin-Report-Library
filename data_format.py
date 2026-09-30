"""
Data Format Module - Bitcoin Analytics Data Pipeline

This module handles all data fetching, transformation, and metric calculation for Bitcoin
market and on-chain analytics. It integrates multiple data sources and computes derived
metrics used throughout the reporting pipeline.

Data Sources:
    - BRK (Bitview): On-chain metrics, difficulty, supply data
    - Yahoo Finance: Equities, ETFs, indices, commodities, forex
    - BRK: Bitcoin OHLC price data
    - Google Sheets: Miner efficiency data
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO
import time
import csv, io
import warnings
from typing import Optional
from data_definitions import (
    BRK_BULK_URL,
    BRK_METRICS,
    ELECTRICITY_BASE_TARIFF_USD_PER_KWH,
    ELECTRICITY_TARIFFS_USD_PER_KWH,
    MINER_DATA_SHEET_URL,
    API_TIMEOUT,
    SATS_PER_BTC,
    market_cap_history_start_date,
    yahoo_market_cap_fx_tickers,
    yahoo_share_ticker_aliases,
    BITCOIN_GENESIS_DATE,
    REFERENCE_DATA_VINTAGES,
    PRICE_OUTLOOK_YEAR,
    REFERENCE_DATA_MAX_AGE_DAYS,
    METCALFE_ADDRESS_COLUMNS,
    HASH_RIBBON_FAST_WINDOW,
    HASH_RIBBON_SLOW_WINDOW,
)


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

# Coin Metrics miner efficiency is a monthly observation. Allow at most two monthly
# publication intervals before refusing to carry it further; freshness is validated from
# the retained source observation date, never inferred from a repeated daily value.
MINER_EFFICIENCY_MAX_AGE_DAYS = 62
MINER_EFFICIENCY_VALUE_COLUMN = "cm_efficiency_j_gh"
MINER_EFFICIENCY_SOURCE_DATE_COLUMN = "cm_efficiency_source_date"
MINER_EFFICIENCY_SOURCE_URL_COLUMN = "cm_efficiency_source_url"
MINER_EFFICIENCY_COLUMNS = [
    MINER_EFFICIENCY_VALUE_COLUMN,
    MINER_EFFICIENCY_SOURCE_DATE_COLUMN,
    MINER_EFFICIENCY_SOURCE_URL_COLUMN,
]
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]

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


def _source_observation_column(value_column: str) -> str:
    return f"{_SOURCE_OBSERVATION_DATE_PREFIX}{value_column}"


# Get Data


def assert_ohlc_usable(ohlc_data: pd.DataFrame, label: str = "OHLC") -> None:
    """Raise before publication when an OHLC frame has no complete numeric candle."""
    if ohlc_data is None or ohlc_data.empty:
        raise RuntimeError(f"{label} data is empty; refusing to overwrite OHLC outputs")

    missing = [column for column in OHLC_COLUMNS if column not in ohlc_data.columns]
    if missing:
        raise RuntimeError(
            f"{label} data is missing required columns {missing}; refusing to overwrite OHLC outputs"
        )

    from data_validation import validate_candles
    validate_candles(ohlc_data, label)


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

        df = pd.DataFrame(ohlc_rows, columns=["Open", "High", "Low", "Close"])
        df["Time"] = pd.to_datetime(dates)
        df.set_index("Time", inplace=True)
        df = df.astype(float)
        from data_validation import validate_calendar
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
    except Exception as e:
        print(f"[yfinance] Batch download failed: {e}")
        return pd.DataFrame(columns=["time"])

    data_frames = []
    for ticker in fetch_tickers:
        try:
            close_series = raw[ticker]["Close"]
            if close_series.isna().all():
                print(f"[yfinance] No data returned for {ticker} — skipping")
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
            print(f"[yfinance] Could not extract {ticker} from batch result — skipping")

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
    except Exception:
        market_cap = None
    if market_cap is None:
        try:
            market_cap = stock.info.get("marketCap")
        except Exception:
            market_cap = None
    try:
        market_cap = float(market_cap)
    except (TypeError, ValueError):
        return None
    return market_cap if np.isfinite(market_cap) and market_cap > 0 else None


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
        pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        - pd.Timedelta(days=1)
        if end_date is None
        else pd.to_datetime(end_date).normalize()
    )
    if requested_end < requested_start:
        raise ValueError("Market-cap end_date cannot be before start_date")

    calendar = pd.date_range(requested_start, requested_end, freq="D", name="time")
    data = pd.DataFrame(index=calendar)
    for ticker in stocks:
        data[f"{ticker}_MarketCap"] = np.nan

    history_start = max(
        requested_start, pd.to_datetime(market_cap_history_start_date).normalize()
    )
    if history_start > requested_end:
        return data.reset_index()

    fetch_start = history_start.strftime("%Y-%m-%d")
    # yfinance treats `end` as exclusive.
    fetch_end = (requested_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in stocks:
        value_column = f"{ticker}_MarketCap"
        stock = None
        fx_symbol = yahoo_market_cap_fx_tickers.get(ticker)
        fx_close = None
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(
                start=fetch_start,
                end=fetch_end,
                auto_adjust=False,
                actions=True,
            )
            if history.empty or "Close" not in history:
                raise ValueError("Yahoo returned no closing-price history")

            price_timezone = getattr(history.index, "tz", None)
            close = _normalize_yahoo_series(history["Close"])
            close = close[close > 0].groupby(level=0, sort=True).last()
            stock_splits = (
                history["Stock Splits"]
                if "Stock Splits" in history
                else pd.Series(dtype="float64")
            )

            share_parts = []
            for share_symbol in yahoo_share_ticker_aliases.get(ticker, [ticker]):
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
                    share_history = share_stock.get_shares_full(
                        start=fetch_start, end=fetch_end
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Yahoo shares unavailable for {share_symbol}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                if share_history is not None and len(share_history) > 0:
                    share_parts.append(_normalize_yahoo_series(share_history))

            if not share_parts:
                raise ValueError("Yahoo returned no historical shares outstanding")

            shares = _split_adjust_yahoo_shares(
                pd.concat(share_parts), stock_splits
            )
            combined_index = shares.index.union(close.index).sort_values()
            shares_on_price_dates = (
                shares.reindex(combined_index).ffill().reindex(close.index)
            )

            # Carry the true filing date alongside the value so staleness is measured
            # from the observation, never inferred from a repeated daily figure. Beyond
            # the budget the market cap becomes NaN rather than silently understating
            # the issuer's dilution.
            share_source_dates = (
                pd.Series(shares.index, index=shares.index)
                .reindex(combined_index)
                .ffill()
                .reindex(close.index)
            )
            row_dates = pd.Series(close.index, index=close.index).dt.normalize()
            share_age_days = (
                row_dates - share_source_dates.dt.normalize()
            ).dt.days
            fresh = share_age_days.between(0, SHARES_OUTSTANDING_MAX_AGE_DAYS)
            shares_on_price_dates = shares_on_price_dates.where(fresh)

            newest_share_date = shares.index.max()
            newest_age = (requested_end.normalize() - newest_share_date.normalize()).days
            if newest_age > SHARES_OUTSTANDING_MAX_AGE_DAYS:
                warnings.warn(
                    f"Shares outstanding for {ticker} last observed "
                    f"{newest_share_date.date()} ({newest_age} days before "
                    f"{requested_end.date()}); {ticker}_MarketCap will be null past the "
                    f"{SHARES_OUTSTANDING_MAX_AGE_DAYS}-day budget",
                    RuntimeWarning,
                    stacklevel=2,
                )

            market_cap = (close * shares_on_price_dates).replace(
                [np.inf, -np.inf], np.nan
            )

            if fx_symbol:
                fx_history = yf.Ticker(fx_symbol).history(
                    start=fetch_start,
                    end=fetch_end,
                    auto_adjust=False,
                )
                if fx_history.empty or "Close" not in fx_history:
                    raise ValueError(
                        f"Yahoo returned no {fx_symbol} USD conversion history"
                    )
                fx_close = _normalize_yahoo_series(fx_history["Close"])
                fx_close = fx_close[fx_close > 0].groupby(level=0, sort=True).last()
                fx_index = fx_close.index.union(close.index).sort_values()
                fx_on_price_dates = (
                    fx_close.reindex(fx_index)
                    .ffill(limit=MARKET_DATA_MAX_FFILL_DAYS)
                    .reindex(close.index)
                )
                market_cap = market_cap * fx_on_price_dates

            market_cap = market_cap.dropna()
            valid_dates = market_cap.index.intersection(data.index)
            data.loc[valid_dates, value_column] = market_cap.loc[valid_dates].astype(
                float
            )
        except Exception as exc:
            warnings.warn(
                f"Could not build Yahoo historical market cap for {ticker}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

        if not data[value_column].notna().any():
            current_market_cap = _current_yahoo_market_cap(stock)
            if current_market_cap is not None and fx_symbol:
                if fx_close is None or fx_close.empty:
                    current_market_cap = None
                else:
                    recent_fx = fx_close.loc[
                        fx_close.index >= requested_end
                        - pd.Timedelta(days=MARKET_DATA_MAX_FFILL_DAYS)
                    ]
                    current_market_cap = (
                        current_market_cap * float(recent_fx.iloc[-1])
                        if not recent_fx.empty
                        else None
                    )
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
        df = pd.read_csv(StringIO(response.text))
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
    except Exception as e:
        print(f"Failed to fetch Coin Metrics network efficiency data from Google Sheets: {e}")
        return pd.DataFrame(columns=["time"] + MINER_EFFICIENCY_COLUMNS)


# Bitcoin's halving schedule. Index 0 is genesis (start of the first subsidy era);
# every later entry is an observed halving date. This is the single source of truth for
# both block subsidy and halving-era segmentation.
BITCOIN_HALVING_DATES = [
    "2009-01-03",  # Genesis — 50 BTC/block
    "2012-11-28",
    "2016-07-09",
    "2020-05-11",
    "2024-04-20",
]

# 210,000 blocks at a 10-minute nominal target is ~1,458 days, but observed intervals
# have run shorter (1,425 / 1,319 / 1,402 / 1,440) because hash rate growth outpaces
# difficulty retargeting. The two most recent intervals average ~1,421 days and the trend
# is back toward nominal, so 1,435 lands the next halving in late March 2028 — in line
# with block-height projections. This only needs to be close enough to segment eras
# correctly; replace the estimate with the observed date once a halving occurs.
HALVING_INTERVAL_DAYS = 1435

GENESIS_BLOCK_SUBSIDY = 50.0


def bitcoin_halving_dates(through=None) -> list:
    """
    Return halving dates (genesis first), projected forward as far as needed.

    Known halvings are returned verbatim. If `through` extends past the last known
    halving, additional dates are projected on the observed ~1,400-day cadence so that
    era segmentation keeps splitting correctly without a manual source edit.

    Returns:
    list[pd.Timestamp]: Ascending halving dates, always covering `through`.
    """
    dates = [pd.Timestamp(d) for d in BITCOIN_HALVING_DATES]
    if through is None:
        return dates

    through = pd.Timestamp(through)
    while dates[-1] <= through:
        dates.append(dates[-1] + pd.Timedelta(days=HALVING_INTERVAL_DAYS))
    return dates


def _bitcoin_block_subsidy_from_time(time_index) -> pd.Series:
    """
    Infer Bitcoin's protocol block subsidy in BTC per block from the date.

    Derived from BITCOIN_HALVING_DATES so it stays correct past the next halving
    instead of pinning every future date to the current subsidy.

    Returns:
    pd.Series: Block subsidy in BTC per block, indexed like `time_index`.
    """
    dates = pd.to_datetime(time_index)
    if len(dates) == 0:
        return pd.Series(dtype=float, index=time_index)

    halvings = bitcoin_halving_dates(through=dates.max())

    # Each halving at position i (i >= 1) halves the genesis subsidy i times.
    reward = np.full(len(dates), GENESIS_BLOCK_SUBSIDY, dtype=float)
    for i, halving_date in enumerate(halvings[1:], start=1):
        reward[np.asarray(dates >= halving_date)] = GENESIS_BLOCK_SUBSIDY / (2**i)

    return pd.Series(reward, index=time_index)


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
    tuple: (header, data_rows, raw_text) - CSV header, data rows, and raw response text.
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

    return header, data_rows, r.text


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
    list[tuple]: One or more (header, rows, raw_text) responses.
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
    from_: int = 0,
    verbose: bool = True,
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
    query_start = start_date or from_

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

        for header, rows, _raw_csv in responses:
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

    if verbose:
        print(f"[BRK] merged rows: {len(data)}")
        print(f"[BRK] merged cols: {len(ordered_cols)}")
        print(f"[BRK] cols: {ordered_cols}")

    # build a single CSV (date derived later in your pipeline; we keep time + metrics)
    header_line = ",".join(ordered_cols)
    row_lines = []
    for ts in sorted(data, key=lambda x: int(float(x))):
        row = [data[ts].get(c, "") for c in ordered_cols]
        row_lines.append((int(float(ts)), ",".join(map(str, row))))
    merged_csv = "\n".join([header_line] + [line for _, line in row_lines])

    # load into pandas
    df = pd.read_csv(StringIO(merged_csv), low_memory=False)

    # A series BRK could not resolve used to be backfilled as an all-NaN column, which
    # then vanished from the fundamentals table via its `len(series) == 0` skip — the
    # report shipped a row short with no error anywhere. Fail loudly instead: a renamed
    # or retired upstream series is a code change, not a data condition.
    absent = [metric for metric in non_ts if metric not in df.columns]
    unresolved = sorted(set(missing_series) | set(absent))
    if unresolved:
        raise RuntimeError(
            "BRK did not return the following required series: "
            + ", ".join(unresolved)
            + ". They were likely renamed or retired upstream. Update BRK_METRICS in "
            "data_definitions.py (check https://bitview.space/api/series/list) rather "
            "than publishing a report with missing metrics."
        )

    # timestamp -> time
    df["time"] = pd.to_datetime(df["timestamp"].astype(float).astype(int), unit="s")
    df.drop(columns=["timestamp"], inplace=True)

    # numeric coercion
    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time"] = df["time"].dt.tz_localize(None)

    if start_date:
        df = df[df["time"] >= pd.to_datetime(start_date)]

    if verbose:
        print(f"[BRK] final df shape: {df.shape}")
        print(f"[BRK] final cols: {list(df.columns)}")
        print(df.tail(3))

    return df


# On-chain series that must be present on the report-date row. BRK can answer 200 with an
# empty or partial payload; numeric coercion turns those cells into NaN, and a blanket
# forward fill would then republish yesterday's numbers as today's — indistinguishable
# from a genuinely flat day. These are checked explicitly instead.
REQUIRED_ONCHAIN_METRICS = [
    "price_close",
    "market_cap",
    "supply",
    "realized_cap",
    "hash_rate",
    "addr_count",
    "addrs_over_100k_sats_addr_count",
    "addrs_over_1m_sats_addr_count",
    "addrs_over_10m_sats_addr_count",
    # Investor sentiment: supply in profit. NUPL is derived from market_cap and realized_cap.
    "supply_in_profit",
]


def _ordinary_market_columns(data: pd.DataFrame) -> list:
    """Return externally observed non-miner series subject to the short fill budget."""
    onchain_columns = {metric for metric in BRK_METRICS if metric != "timestamp"}
    excluded = onchain_columns | set(MINER_EFFICIENCY_COLUMNS)
    return [
        column
        for column in data.columns
        if column not in excluded
        and not column.startswith(_SOURCE_OBSERVATION_DATE_PREFIX)
    ]


def _normalized_index(data: pd.DataFrame) -> pd.DatetimeIndex:
    """Return a tz-naive normalized DatetimeIndex without mutating the input frame."""
    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    if index.tz is not None:
        index = index.tz_convert(None)
    return index.normalize()


def warn_on_stale_market_data(
    data: pd.DataFrame,
    report_date,
    max_age_days: int = MARKET_DATA_MAX_FFILL_DAYS,
) -> list:
    """
    Warn about ordinary market series whose last proven observation is too old.

    This check must run before `forward_fill_market_data`. Price fetchers retain
    temporary source-date markers, so an already repeated weekend value cannot masquerade
    as a new source observation. Monthly miner efficiency has its own explicit policy.

    Stale values remain NaN after the bounded fill, but a single unavailable ticker does not
    abort the report. The returned issue strings also make the warning machine-testable.

    Returns:
    list[str]: Stale or missing series descriptions; empty when all are fresh.
    """
    if max_age_days < 0:
        raise ValueError("Market-data max_age_days cannot be negative")
    if data.empty:
        issues = ["market data frame is empty"]
        warnings.warn(issues[0], RuntimeWarning, stacklevel=2)
        return issues

    report_date = pd.to_datetime(report_date).normalize()
    normalized_index = _normalized_index(data)
    available_mask = normalized_index <= report_date
    if not available_mask.any():
        issues = [f"no market data exists on or before {report_date.date()}"]
        warnings.warn(issues[0], RuntimeWarning, stacklevel=2)
        return issues

    stale = []
    missing = []
    for column in _ordinary_market_columns(data):
        values = data.loc[available_mask, column]
        if not values.notna().any():
            missing.append(column)
            continue

        marker_column = _source_observation_column(column)
        if marker_column in data.columns:
            source_dates = pd.to_datetime(
                data.loc[available_mask, marker_column], errors="coerce"
            ).dropna()
            if source_dates.empty:
                missing.append(f"{column} (source date missing)")
                continue
            observation_date = source_dates.iloc[-1].normalize()
        else:
            observed_positions = np.flatnonzero(available_mask & data[column].notna())
            observation_date = normalized_index[observed_positions[-1]]

        age_days = (report_date - observation_date).days
        if age_days < 0 or age_days > max_age_days:
            stale.append(
                f"{column} (source {observation_date.date()}, {age_days} days old)"
            )

    if missing or stale:
        details = []
        if stale:
            details.append("stale=" + ", ".join(stale))
        if missing:
            details.append("missing=" + ", ".join(missing))
        message = (
            "Market-data freshness warning: "
            + "; ".join(details)
            + f". Maximum allowed age is {max_age_days} calendar days; stale values "
            "will remain NaN."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return stale + missing


def forward_fill_market_data(
    data: pd.DataFrame,
    market_max_age_days: int = MARKET_DATA_MAX_FFILL_DAYS,
    miner_max_age_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Forward-fill only the columns that legitimately have gaps.

    Ordinary market data may bridge at most `market_max_age_days` calendar days. Monthly
    miner efficiency carries the last published estimate by default while retaining its
    true observation date; callers may pass `miner_max_age_days` to impose a hard limit.
    On-chain series are never filled. Historical stock market caps now vary with daily
    prices and therefore use the same bounded policy as other market data.

    Returns:
    pd.DataFrame: Copy with bounded fills and temporary market source markers removed.
    """
    if market_max_age_days < 0 or (
        miner_max_age_days is not None and miner_max_age_days < 0
    ):
        raise ValueError("Forward-fill age limits cannot be negative")

    data = data.copy()
    normalized_index = _normalized_index(data)
    row_dates = pd.Series(normalized_index, index=data.index)

    for column in _ordinary_market_columns(data):
        filled = data[column].ffill(limit=market_max_age_days)
        marker_column = _source_observation_column(column)
        if marker_column in data.columns:
            source_dates = pd.to_datetime(
                data[marker_column], errors="coerce"
            ).ffill(limit=market_max_age_days)
            age_days = (row_dates - source_dates.dt.normalize()).dt.days
            filled = filled.where(age_days.between(0, market_max_age_days))
        data[column] = filled

    miner_columns = [c for c in MINER_EFFICIENCY_COLUMNS if c in data.columns]
    if miner_columns:
        data[miner_columns] = data[miner_columns].ffill(limit=miner_max_age_days)
        if (
            miner_max_age_days is not None
            and MINER_EFFICIENCY_SOURCE_DATE_COLUMN in data.columns
        ):
            source_dates = pd.to_datetime(
                data[MINER_EFFICIENCY_SOURCE_DATE_COLUMN], errors="coerce"
            )
            age_days = (row_dates - source_dates.dt.normalize()).dt.days
            valid = age_days.between(0, miner_max_age_days)
            data[miner_columns] = data[miner_columns].where(valid, axis=0)

    source_marker_columns = [
        column
        for column in data.columns
        if column.startswith(_SOURCE_OBSERVATION_DATE_PREFIX)
    ]
    if source_marker_columns:
        data.drop(columns=source_marker_columns, inplace=True)
    return data


def warn_on_stale_miner_efficiency(
    data: pd.DataFrame,
    report_date,
    max_age_days: int = MINER_EFFICIENCY_MAX_AGE_DAYS,
) -> list:
    """
    Validate miner-efficiency provenance and warn when its last observation is old.

    The value must be present on the latest dataset row at or before the report date, its
    retained source observation date must be no more than `max_age_days` old, and its
    source URL provenance must be present. Repeated daily rows never reset source age.

    Missing, unusable, or future-dated values still raise because there is no valid
    estimate to use. An otherwise valid old observation is carried forward and reported
    as a RuntimeWarning instead of aborting report generation.

    Returns:
    list[str]: Warning descriptions; empty when the observation is current.
    """
    if max_age_days < 0:
        raise ValueError("Miner-efficiency max_age_days cannot be negative")

    absent = [column for column in MINER_EFFICIENCY_COLUMNS if column not in data.columns]
    if absent:
        raise RuntimeError(
            "Miner-efficiency data is missing required provenance columns: "
            + ", ".join(absent)
        )

    report_date = pd.to_datetime(report_date).normalize()
    normalized_index = _normalized_index(data)
    available_mask = normalized_index <= report_date
    if not available_mask.any():
        raise RuntimeError(
            f"No miner-efficiency data exists on or before {report_date.date()}"
        )

    as_of_position = np.flatnonzero(available_mask)[-1]
    as_of = normalized_index[as_of_position]
    as_of_row = data.iloc[as_of_position]

    value_history = data.loc[available_mask, MINER_EFFICIENCY_VALUE_COLUMN]
    valid_positions = np.flatnonzero(available_mask & data[MINER_EFFICIENCY_VALUE_COLUMN].notna())
    if value_history.dropna().empty or len(valid_positions) == 0:
        raise RuntimeError("Miner-efficiency series has no usable observation")

    latest_value_position = valid_positions[-1]
    source_date = pd.to_datetime(
        data.iloc[latest_value_position][MINER_EFFICIENCY_SOURCE_DATE_COLUMN],
        errors="coerce",
    )
    provenance = data.iloc[latest_value_position][MINER_EFFICIENCY_SOURCE_URL_COLUMN]
    if pd.isna(source_date):
        raise RuntimeError("Miner-efficiency source observation date is missing")
    if pd.isna(provenance) or not str(provenance).strip():
        raise RuntimeError("Miner-efficiency source provenance URL is missing")

    source_date = source_date.normalize()
    age_days = (report_date - source_date).days
    if age_days < 0:
        raise RuntimeError(
            f"Miner-efficiency source observation {source_date.date()} is after report date "
            f"{report_date.date()}"
        )
    if age_days > max_age_days:
        message = (
            f"Miner-efficiency data is stale: using last available source observation "
            f"{source_date.date()}, which is {age_days} days old on report date "
            f"{report_date.date()} (warning threshold {max_age_days}). "
            f"Provenance: {provenance}"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        issues = [message]
    else:
        issues = []

    if pd.isna(as_of_row[MINER_EFFICIENCY_VALUE_COLUMN]):
        raise RuntimeError(
            f"Miner-efficiency value was not carried to latest dataset row {as_of.date()} "
            "despite an available source observation"
        )

    return issues


def assert_onchain_freshness(data: pd.DataFrame, report_date, metrics=None) -> None:
    """
    Verify the report-date row actually carries on-chain data.

    Raises:
    RuntimeError: If the report date is missing from the index, or any required on-chain
                  metric is null on that row — i.e. the pipeline is about to publish a
                  report built on absent upstream data.
    """
    metrics = metrics or REQUIRED_ONCHAIN_METRICS
    report_date = pd.to_datetime(report_date).normalize()

    available = data.index[data.index <= report_date]
    if len(available) == 0:
        raise RuntimeError(
            f"No data on or before the report date ({report_date.date()}). "
            "Upstream fetch returned nothing usable."
        )

    # Every report table reads the exact report-date row, so any lag must fail here with
    # the real cause rather than later as a KeyError or a "missing fundamental".
    as_of = available.max()
    if as_of != report_date:
        raise RuntimeError(
            f"On-chain data is stale: latest row is {as_of.date()}, report date is "
            f"{report_date.date()}. Refusing to publish a report on stale data."
        )

    row = data.loc[as_of]
    missing = [m for m in metrics if m in data.columns and pd.isna(row[m])]
    absent = [m for m in metrics if m not in data.columns]

    if missing or absent:
        raise RuntimeError(
            f"Required on-chain metrics are unusable on {as_of.date()}: "
            f"null={missing or 'none'}, absent={absent or 'none'}. BRK likely returned a "
            "partial payload. Refusing to publish rather than carrying forward stale values."
        )


# Series that feed a cumulative sum. A hole in one of these is not a missing day — it
# permanently shifts every subsequent total, and the resulting curve looks entirely
# plausible, so it has to be caught at ingest rather than eyeballed downstream.
CUMULATIVE_ONCHAIN_INPUTS = ["coinbase_sum_24h_usd"]

# Series that every `*_btc_price` and per-coin metric divides by. They are never filled,
# so a hole must fail ingest rather than publish a gap (or, worse, a repeated value).
GAP_CHECKED_ONCHAIN_INPUTS = CUMULATIVE_ONCHAIN_INPUTS + ["supply"]


def assert_price_outlook_current(report_date, outlook_year: int = PRICE_OUTLOOK_YEAR) -> None:
    """
    Verify the published case levels forecast the year the report belongs to.

    The bull/base/bear levels are hand-maintained and revised once a year. Without this
    check, the first run of a new year silently republishes last year's forecast — and
    both the dashboard cards and the homepage tracker label it with the new year.

    Raises:
    RuntimeError: If the outlook year does not match the report date's year.
    """
    report_date = pd.to_datetime(report_date).normalize()
    if int(outlook_year) != report_date.year:
        raise RuntimeError(
            f"Price outlook is for {outlook_year} but the report date is "
            f"{report_date.date()}. Publish the new Year Ahead Outlook and update "
            "PRICE_OUTLOOK_YEAR and price_outlook_levels in data_definitions.py."
        )


def assert_reference_data_fresh(
    report_date, vintages=None, max_age_days: int = REFERENCE_DATA_MAX_AGE_DAYS
) -> None:
    """
    Verify the hand-maintained reference figures have been re-checked recently enough.

    These are broadcast across the entire daily history, so a stale figure is presented as
    though it held in 2010. Unlike the fetched sources there is nothing to observe their
    age from — only the vintage a maintainer recorded when last confirming them.

    Raises:
    RuntimeError: If any reference figure is older than the budget on the report date.
    """
    vintages = REFERENCE_DATA_VINTAGES if vintages is None else vintages
    report_date = pd.to_datetime(report_date).normalize()

    stale = []
    for name, as_of in vintages.items():
        as_of_date = pd.to_datetime(as_of, errors="coerce")
        if pd.isna(as_of_date):
            stale.append(f"{name} (invalid vintage {as_of!r})")
            continue
        as_of_date = as_of_date.normalize()
        if getattr(as_of_date, "tz", None) is not None:
            as_of_date = as_of_date.tz_convert(None)
        age_days = (report_date - as_of_date).days
        if age_days < 0:
            stale.append(f"{name} (future vintage {as_of_date.date()})")
            continue
        if age_days > max_age_days:
            stale.append(f"{name} (as of {as_of_date.date()}, {age_days} days old)")

    if stale:
        raise RuntimeError(
            "Hand-maintained reference data is stale: "
            + "; ".join(stale)
            + f". Maximum allowed age is {max_age_days} days. Re-check the figures in "
            "data_definitions.py and bump their *_AS_OF vintage."
        )


def assert_no_internal_onchain_gaps(
    data: pd.DataFrame, report_date, columns=None
) -> None:
    """
    Verify gap-checked on-chain inputs have no holes between first and last observation.

    Leading nulls before a series begins are expected and contribute zero. A gap *inside*
    the observed range is not recoverable by zero-filling: `RevAllTimeUSD` and every
    thermocap series derived from it would be understated for all later dates with no
    visible artefact. `supply` is checked for the same reason: it is never filled, and
    every per-coin price series divides by it.

    Raises:
    RuntimeError: If any monitored column has an internal gap on or before the report date.
    """
    columns = columns or GAP_CHECKED_ONCHAIN_INPUTS
    report_date = pd.to_datetime(report_date).normalize()
    from data_validation import validate_calendar
    validate_calendar(data.index, "On-chain data")
    normalized_index = _normalized_index(data)
    in_range = data.loc[normalized_index <= report_date]

    problems = []
    for column in columns:
        if column not in in_range.columns:
            problems.append(f"{column} is absent")
            continue
        values = in_range[column]
        observed = values.notna()
        if not observed.any():
            problems.append(f"{column} has no observations")
            continue
        interior = values.loc[observed.idxmax() : observed[::-1].idxmax()]
        gaps = interior.index[interior.isna()]
        if len(gaps):
            shown = ", ".join(str(d.date()) for d in gaps[:5])
            more = f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else ""
            problems.append(f"{column} has an internal gap on {shown}{more}")

    if problems:
        raise RuntimeError(
            "Gap-checked on-chain inputs are incomplete: "
            + "; ".join(problems)
            + ". Refusing to publish rather than filling a running total or divisor."
        )


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

    from data_validation import validate_calendar
    data = processed_datasets["coindata"]
    validate_calendar(data.index, "BRK on-chain data")
    for name, dataset in processed_datasets.items():
        if name == "coindata":
            continue
        data = pd.merge(data, dataset, left_index=True, right_index=True, how="left")

    # Optional assets retain a stable schema when their providers return no data.
    optional = [f"{ticker}_close" for group in tickers.values() for ticker in group]
    optional += [f"{ticker}_MarketCap" for ticker in tickers.get("stocks", [])]
    data = data.reindex(columns=list(dict.fromkeys([*data.columns, *optional])))

    # Handle duplicates
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()]
    if data.index.duplicated().any():
        data = data[~data.index.duplicated()]

    return data


# Metric Calculation


def calculate_custom_on_chain_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive Bitcoin on-chain valuation and network health metrics.

    This function computes derived metrics including valuation models (MVRV, NVT, Thermocap,
    realized-cap multiples), price moving averages, NUPL, supply profitability, reserve risk,
    average/delta cap and volatility. Only columns a consumer reads are published; the
    intermediates behind them (all-time miner revenue, adjusted BDD, HODL bank) stay local.

    Parameters:
    data (pd.DataFrame): DataFrame with DatetimeIndex containing BRK API on-chain metrics.
                         Must include columns listed above for full metric calculation.
    """
    # Bind the source columns this function leans on repeatedly.
    market_cap = data["market_cap"]
    supply = data["supply"]
    realized_cap = data["realized_cap"]
    price_close = data["price_close"]
    transfer_volume = data["transfer_volume_sum_24h_usd"]
    miner_revenue_usd = data["coinbase_sum_24h_usd"]

    # --- Intermediates that later metrics build on -------------------------------
    # Only leading nulls survive `assert_no_internal_onchain_gaps`, and those legitimately
    # contribute zero: they precede the series' first observation. An interior hole would
    # have aborted the run already.
    rev_all_time = miner_revenue_usd.fillna(0).cumsum()
    nvt_adj = market_cap / transfer_volume

    # Early source rows carry a 0.0 price placeholder from before Bitcoin had a market
    # price. Dividing by those publishes inf, which downstream consumers cannot chart:
    # any max()/min() over the column returns inf, and JSON encoders serialize
    # non-finite floats as null. Treat non-positive prices as missing instead.
    positive_price = price_close.where(price_close > 0)

    mvrv_ratio = market_cap / realized_cap  # published as CapMVRVCur
    nvt_price = (nvt_adj.rolling(window=365 * 2).median() * transfer_volume) / supply
    ma_200_day = price_close.rolling(window=200).mean()

    # BRK provides utxos_over_1y_old_supply in BTC; divide by circulating supply for %
    supply_pct_1_year_plus = (data["utxos_over_1y_old_supply"] / supply) * 100
    illiquid_supply = (supply_pct_1_year_plus / 100) * supply

    # Reserve Risk pipeline: adjusted BDD -> VOCD -> MVOCD -> HODL bank -> reserve risk
    adjusted_bdd = data["coindays_destroyed_sum_24h"] / supply
    vocd = price_close * adjusted_bdd
    mvocd = vocd.rolling(window=30).median()
    daily_hodl_value = (price_close - mvocd).clip(lower=0)
    hodl_bank = daily_hodl_value.cumsum()

    # Average Cap and Delta Cap
    # The divisor is the network's true age, not a row counter. The fetched history
    # starts 2010-01-01 — 363 days after genesis — so counting rows understates the
    # denominator and overstates Average Cap by ~6%, with the error shrinking as the
    # window lengthens (which distorts the curve's shape, not just its level).
    cumulative_market_cap = market_cap.cumsum()
    days_since_start = pd.Series(
        (_normalized_index(data) - BITCOIN_GENESIS_DATE).days + 1,
        index=data.index,
    ).clip(lower=1)
    average_cap = cumulative_market_cap / days_since_start
    delta_cap = realized_cap - average_cap

    daily_returns = price_close.pct_change(fill_method=None)


    new_columns = {
        "sat_per_dollar": SATS_PER_BTC / positive_price,
        "CapMVRVCur": mvrv_ratio,
        "nupl": (market_cap - realized_cap) / market_cap,
        "nvt_price": nvt_price,
        "7_day_ma_price_close": price_close.rolling(window=7).mean(),
        "50_day_ma_price_close": price_close.rolling(window=50).mean(),
        "200_day_ma_price_close": ma_200_day,
        "200_week_ma_price_close": price_close.rolling(window=200 * 7).mean(),
        "200_day_multiple": price_close / ma_200_day,
        "thermocap_price": rev_all_time / supply,
        "thermocap_price_multiple_4": (4 * rev_all_time) / supply,
        "thermocap_price_multiple_8": (8 * rev_all_time) / supply,
        "thermocap_price_multiple_16": (16 * rev_all_time) / supply,
        "thermocap_price_multiple_32": (32 * rev_all_time) / supply,
        "realizedcap_multiple_2": (2 * realized_cap) / supply,
        "realizedcap_multiple_3": (3 * realized_cap) / supply,
        "realizedcap_multiple_5": (5 * realized_cap) / supply,
        "supply_pct_1_year_plus": supply_pct_1_year_plus,
        "pct_supply_issued": supply / 21000000,
        "pct_fee_of_reward": (data["fees_sum_24h"] / data["coinbase_sum_24h"]) * 100,
        "illiquid_supply": illiquid_supply,
        "liquid_supply": supply - illiquid_supply,
        # active_addrs_average_24h is already a daily total — no block-count scaling.
        "daily_active_addresses_sending": data["active_addrs_average_24h"],
        "vocd": vocd,
        "mvocd": mvocd,
        "reserve_risk_calc": price_close / hodl_bank,
        "average_cap_price": average_cap / supply,
        "delta_cap_price": delta_cap / supply,
        "VtyDayRet30d": daily_returns.rolling(30).std() * np.sqrt(365),
        "VtyDayRet180d": daily_returns.rolling(180).std() * np.sqrt(365),
        "supply_in_profit_pct": (data["supply_in_profit"] / supply) * 100,
        "supply_in_loss_pct": (data["supply_in_loss"] / supply) * 100,
    }

    # Realized price: the value at which each coin last moved. BRK usually supplies it,
    # so fill gaps rather than overwrite; only derive the whole column if it is absent.
    calculated_realized_price = realized_cap / supply
    if "realized_price" in data.columns:
        data["realized_price"] = data["realized_price"].fillna(calculated_realized_price)
    else:
        new_columns["realized_price"] = calculated_realized_price

    # Attach every derived column in one concat. Assigning them individually inserts
    # ~60 separate blocks into the frame, which triggers pandas' fragmentation warning
    # and makes each successive assignment slower as the column count grows.
    data = pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)

    print("Custom Metrics Created")
    return data


def calculate_moving_averages(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Add 30-day and 365-day moving averages for each metric in `metrics`
    (data_definitions.moving_avg_metrics), as `30_day_ma_{metric}` and `365_day_ma_{metric}`.
    """
    moving_averages = {
        f"{window}_day_ma_{metric}": data[metric].rolling(window=window).mean()
        for window in (30, 365)
        for metric in metrics
    }

    data = pd.concat([data, pd.DataFrame(moving_averages)], axis=1)
    return data


def calculate_metal_market_caps(
    data: pd.DataFrame, gold_silver_supply: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate market caps for gold and silver and add them to the DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing existing financial data.
    gold_silver_supply (pd.DataFrame): DataFrame containing supply data for gold and silver.

    Returns:
    pd.DataFrame: DataFrame with added columns for metal market caps.

    Notes:
    The published ``*_marketcap_billion_usd`` column names are retained for
    compatibility, but their values are absolute USD market caps. The suffix is
    legacy naming and is not a scaling instruction.
    """
    new_columns = {}
    for _, row in gold_silver_supply.iterrows():
        metal = row["Metal"]
        supply_billion_troy_ounces = row["Supply in Billion Troy Ounces"]

        # Skip if the supply data is missing
        if pd.isna(supply_billion_troy_ounces):
            print(f"Warning: Supply data for {metal} is NaN.")
            continue

        # Determine the correct price column based on the metal type
        if metal == "Gold":
            if "GC=F_close" not in data:
                print("Warning: Gold price data column is missing.")
                continue
            # Use the last available price, forward filling missing values
            price_usd_per_ounce = data["GC=F_close"].ffill()
        elif metal == "Silver":
            if "SI=F_close" not in data:
                print("Warning: Silver price data column is missing.")
                continue
            # Use the last available price, forward filling missing values
            price_usd_per_ounce = data["SI=F_close"].ffill()

        # Calculate the market cap using the last available price
        metric_name = f"{metal.lower()}_marketcap_billion_usd"
        market_cap = supply_billion_troy_ounces * price_usd_per_ounce.iloc[-1]
        # Create a new series for the calculated market cap, indexed to match the data DataFrame
        new_columns[metric_name] = pd.Series(market_cap, index=data.index)

    # Concatenate the new columns to the original data
    data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
    return data


def calculate_btc_price_to_surpass_metal_categories(
    data: pd.DataFrame, gold_supply_breakdown: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the BTC price needed to surpass various metal market caps.

    Parameters:
    data (pd.DataFrame): DataFrame containing existing financial data with BRK native field names.
    gold_supply_breakdown (pd.DataFrame): DataFrame containing breakdown percentages for gold supply.

    Returns:
    pd.DataFrame: DataFrame with added columns for BTC prices needed to surpass metal categories.
    """
    # On-chain supply is never filled: `assert_no_internal_onchain_gaps` guarantees it has
    # no interior holes, and any row without a positive supply publishes NaN rather than a
    # value divided by a copied-forward or zero supply.
    supply = data["supply"].where(data["supply"] > 0)

    new_columns = {}  # Use a dictionary to store new columns

    # Calculating BTC prices required to match or surpass gold market cap
    gold_marketcap_billion_usd = data["gold_marketcap_billion_usd"].iloc[-1]
    new_columns["gold_marketcap_btc_price"] = gold_marketcap_billion_usd / supply

    # Iterating through gold supply breakdown to calculate BTC prices for specific categories
    for _, row in gold_supply_breakdown.iterrows():
        category = row["Gold Supply Breakdown"].replace(" ", "_").lower()
        percentage_of_market = row["Percentage Of Market"] / 100.0
        new_columns[f"gold_{category}_marketcap_btc_price"] = (
            gold_marketcap_billion_usd * percentage_of_market
        ) / supply

    # Silver market cap calculations
    silver_marketcap_billion_usd = data["silver_marketcap_billion_usd"].iloc[-1]
    new_columns["silver_marketcap_btc_price"] = silver_marketcap_billion_usd / supply

    # Convert the dictionary to a DataFrame and concatenate it with the original DataFrame
    new_columns_df = pd.DataFrame(new_columns, index=data.index)
    data = pd.concat([data, new_columns_df], axis=1)

    return data


def calculate_btc_price_to_surpass_fiat(
    data: pd.DataFrame, fiat_money_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the BTC price needed to surpass the fiat supply of different countries.

    Parameters:
    data (pd.DataFrame): DataFrame containing existing financial data with BRK native field names.
    fiat_money_data (pd.DataFrame): DataFrame containing fiat supply data for different countries.

    Returns:
    pd.DataFrame: DataFrame with added columns for BTC prices needed to surpass fiat supplies.
    """
    fiat_marketcap = {}

    for _, row in fiat_money_data.iterrows():
        country = row["Country"].replace(" ", "_")
        fiat_supply_usd_trillion = row["US Dollar Trillion"]

        # Convert the fiat supply from trillions to units
        fiat_supply_usd = fiat_supply_usd_trillion * 1e12

        # Compute the price of Bitcoin needed to surpass this country's fiat supply
        fiat_marketcap[f"{country}_btc_price"] = fiat_supply_usd / data["supply"]

    data = pd.concat([data, pd.DataFrame(fiat_marketcap)], axis=1)
    return data


def calculate_btc_price_for_stock_mkt_caps(
    data: pd.DataFrame, stock_tickers: list
) -> pd.DataFrame:
    """
    Calculate the BTC price needed to surpass market caps of different stocks.

    Parameters:
    data (pd.DataFrame): DataFrame containing existing financial data with BRK native field names.
    stock_tickers (list): List of stock tickers to calculate market cap-based BTC prices for.

    Returns:
    pd.DataFrame: DataFrame with added columns for BTC prices needed to surpass stock market caps.
    """
    stock_marketcap_prices = {
        f"{ticker}_mc_btc_price": data[f"{ticker}_MarketCap"] / data["supply"]
        for ticker in stock_tickers
    }

    data = pd.concat([data, pd.DataFrame(stock_marketcap_prices)], axis=1)
    return data


## Onchain Models Calculation


def calculate_network_model_metrics(data, model_end_date=None):
    """Calculate the strategy notebook's Metcalfe, power-law, and hash-ribbon series.

    Coefficients are fitted using positive observations on or before
    ``model_end_date`` (normally the report date), then those equations are evaluated
    across the full frame. Metcalfe fixes the exponent at 2 and fits its market-cap
    scale. The power law fits ``price = scale * age**exponent`` in log-log space.
    Hash Ribbons compare 30- and 60-day simple moving averages of inferred hash rate.
    """
    required = {
        "price_close",
        "market_cap",
        "supply",
        "hash_rate",
        *METCALFE_ADDRESS_COLUMNS.keys(),
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Network model input is missing required columns: {missing}")

    result = data.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if index.tz is not None:
        index = index.tz_convert(None)
    result.index = index
    result = result.sort_index()

    fit_end = (
        pd.to_datetime(model_end_date).normalize()
        if model_end_date is not None
        else result.index.max().normalize()
    )
    fit_mask = result.index.normalize() <= fit_end
    price = pd.to_numeric(result["price_close"], errors="coerce")
    supply = pd.to_numeric(result["supply"], errors="coerce")
    # The strategy notebook defines market_cap_usd directly from the same price
    # and supply series used by the model rather than fitting against a separate
    # upstream market-cap field.
    model_market_cap = price * supply
    days_since_genesis = pd.Series(
        (result.index.normalize() - BITCOIN_GENESIS_DATE).days.astype(float),
        index=result.index,
    )
    new_columns = {}

    power_fit = fit_mask & price.gt(0) & days_since_genesis.gt(0)
    if power_fit.sum() < 2:
        raise ValueError("Power-law model requires at least two positive-price rows")
    exponent, log_scale = np.polyfit(
        np.log(days_since_genesis.loc[power_fit]),
        np.log(price.loc[power_fit]),
        1,
    )
    power_law_price = np.exp(log_scale) * days_since_genesis.where(
        days_since_genesis > 0
    ).pow(exponent)
    new_columns.update(
        {
            "power_law_price": power_law_price,
            "power_law_price_multiple": price.div(
                power_law_price.where(power_law_price > 0)
            ),
            "power_law_exponent": pd.Series(float(exponent), index=result.index),
            "power_law_scale": pd.Series(float(np.exp(log_scale)), index=result.index),
        }
    )

    for address_column, suffix in METCALFE_ADDRESS_COLUMNS.items():
        addresses = pd.to_numeric(result[address_column], errors="coerce")
        metcalfe_fit = (
            fit_mask & model_market_cap.gt(0) & supply.gt(0) & addresses.gt(0)
        )
        if not metcalfe_fit.any():
            raise ValueError(
                f"Metcalfe model requires positive observations for {address_column}"
            )
        scale = np.exp(
            (
                np.log(model_market_cap.loc[metcalfe_fit])
                - 2 * np.log(addresses.loc[metcalfe_fit])
            ).mean()
        )
        value = (scale * addresses.pow(2)).div(supply.where(supply > 0))
        new_columns[f"metcalfe_value_{suffix}"] = value
        new_columns[f"metcalfe_scale_{suffix}"] = pd.Series(
            float(scale), index=result.index
        )

    new_columns["metcalfe_value"] = new_columns["metcalfe_value_any_balance"]
    new_columns["metcalfe_price_multiple"] = price.div(
        new_columns["metcalfe_value"].where(new_columns["metcalfe_value"] > 0)
    )

    hash_rate = pd.to_numeric(result["hash_rate"], errors="coerce")
    fast = hash_rate.rolling(HASH_RIBBON_FAST_WINDOW).mean()
    slow = hash_rate.rolling(HASH_RIBBON_SLOW_WINDOW).mean()
    ribbon_valid = fast.notna() & slow.notna() & slow.ne(0)
    capitulation = pd.Series(pd.NA, index=result.index, dtype="boolean")
    capitulation.loc[ribbon_valid] = fast.loc[ribbon_valid] < slow.loc[ribbon_valid]
    new_columns.update(
        {
            f"{HASH_RIBBON_FAST_WINDOW}_day_ma_hash_rate": fast,
            f"{HASH_RIBBON_SLOW_WINDOW}_day_ma_hash_rate": slow,
            "hash_ribbon_capitulation": capitulation,
        }
    )

    # calculate_moving_averages already creates the 30-day hash-rate column.
    # Replace it with the identical strategy calculation instead of duplicating it.
    existing = [column for column in new_columns if column in result.columns]
    if existing:
        result = result.drop(columns=existing)
    return pd.concat([result, pd.DataFrame(new_columns, index=result.index)], axis=1)


def electric_price_models(data):
    """
    Calculate electricity-based Bitcoin valuation models.

    BRK inputs:
        - hash_rate
        - difficulty
        - subsidy_sum_24h
        - fees_sum_24h
        - price_close

    Google Sheet input:
        - cm_efficiency_j_gh: Coin Metrics Labs monthly estimated Bitcoin network
          efficiency in J/GH, forward-filled daily.

    Model outputs:
        - Electricity_Cost_{3c..7c}: Power expense per BTC earned (subsidy plus fees)
          under tariff scenarios.
        - Electricity_Cost: Alias for the base $0.05/kWh power-expense scenario.
        - Hayes_Network_Price_Per_BTC: Hayes cost-of-production price per BTC, using
          the protocol block subsidy inferred from halving dates.
        - Hayes_Network_Price_Multiple: price_close / Hayes_Network_Price_Per_BTC.
    """
    SECONDS_PER_DAY = 24 * 60 * 60
    SHA_256_CONSTANT = 2**32

    hash_rate_th_s = data["hash_rate"] / 1e12

    # Main efficiency input: Coin Metrics monthly network efficiency in J/GH,
    # forward-filled daily from the Google Sheet.
    efficiency_j_gh = data["cm_efficiency_j_gh"]

    # Hayes uses deterministic protocol subsidy inferred from halving dates.
    block_reward = _bitcoin_block_subsidy_from_time(data.index)

    # H/s ÷ 1e9 gives GH/s; multiplying by J/GH gives J/s (watts), then kWh per day.
    daily_electricity_consumption_kwh = data["hash_rate"] / 1e9 * efficiency_j_gh * 24 / 1000
    miner_revenue_btc = data["subsidy_sum_24h"] + data["fees_sum_24h"]

    for tariff in ELECTRICITY_TARIFFS_USD_PER_KWH:
        cents = int(round(tariff * 100))
        data[f"Electricity_Cost_{cents}c"] = (
            daily_electricity_consumption_kwh * tariff
        ).div(miner_revenue_btc.where(miner_revenue_btc > 0))

    base_cents = int(round(ELECTRICITY_BASE_TARIFF_USD_PER_KWH * 100))
    data["Electricity_Cost"] = data[f"Electricity_Cost_{base_cents}c"]

    btc_per_day_network_expected = (
        data["hash_rate"]
        * SECONDS_PER_DAY
        * block_reward
        / (data["difficulty"] * SHA_256_CONSTANT)
    )

    e_day_network = (
        ELECTRICITY_BASE_TARIFF_USD_PER_KWH
        * 24
        * efficiency_j_gh
        * hash_rate_th_s
    )

    data["Hayes_Network_Price_Per_BTC"] = np.where(
        btc_per_day_network_expected > 0,
        e_day_network / btc_per_day_network_expected,
        np.nan,
    )

    data["Hayes_Network_Price_Multiple"] = np.where(

        data["Hayes_Network_Price_Per_BTC"] != 0,

        data["price_close"] / data["Hayes_Network_Price_Per_BTC"],

        np.nan,

    )

    return data


# Timeframe Calculations


def calculate_rolling_cagr_for_all_columns(data, years):
    """
    Calculate the rolling Compound Annual Growth Rate (CAGR) for all columns in the DataFrame over the specified number of years.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.
    years (int): Number of years over which to calculate the CAGR.

    Returns:
    pd.DataFrame: DataFrame containing the calculated CAGR for each column.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")

    # Ensure that all data is numeric by coercing non-numeric values to NaN
    data = data.apply(pd.to_numeric, errors="coerce")

    # Look up the same calendar date `years` earlier, as calculate_yoy_change does. A
    # fixed 365-day row shift ignores leap days, so a "4 Year" window spans 1,460 days
    # instead of 1,461. DateOffset maps February 29 to February 28 in a common year.
    start_value = data.reindex(data.index - pd.DateOffset(years=years))
    start_value.index = data.index

    # Replace zero start values with NaN to avoid ZeroDivisionError
    # (CAGR from zero is mathematically undefined)
    start_value = start_value.replace(0, np.nan)

    # Calculate CAGR using the formula: ((End Value / Start Value)^(1/years)) - 1
    # Division by zero or negative values will produce NaN/inf, which is mathematically correct
    cagr = ((data / start_value) ** (1 / years) - 1) * 100  # Convert to percentage
    # Replace inf values with NaN for cleaner output
    cagr = cagr.replace([np.inf, -np.inf], np.nan)

    cagr.columns = [f"{col}_{years}_Year_CAGR" for col in cagr.columns]

    return cagr


def _safe_pct_change(numerator, denominator):
    """
    Percentage change in percentage points, treating a zero denominator as missing.

    Pre-2012 source rows carry 0.0 placeholders for metrics that did not exist yet. A
    plain division there yields inf, which is worse than a gap: it poisons every
    downstream min()/max() over the column, and JSON encoders serialize non-finite
    floats as null, so charts silently lose whatever depends on the column's range.

    Returns:
    Same shape as `numerator`, with NaN wherever the denominator was 0 or missing.
    """
    denominator = denominator.where(denominator != 0)
    return ((numerator / denominator) - 1) * 100


def _previous_period_positive_close(data, period):
    """Align each row with the last positive observation before its period began.

    The lookup is independent per column: a missing or zero value at a calendar
    boundary does not hide an earlier valid close for that metric. ``period`` is
    either ``"month"`` or ``"year"``.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")
    if period not in {"month", "year"}:
        raise ValueError("period must be either 'month' or 'year'.")

    period_frequency = "M" if period == "month" else "Y"
    period_starts = pd.DatetimeIndex(
        data.index.to_period(period_frequency).start_time
    )
    lookup_dates = period_starts - pd.Timedelta(nanoseconds=1)

    # Forward-filling only the positive subset makes this a per-column lookup of
    # the latest valid observation, even when the immediately prior row is null/zero.
    sorted_data = data.sort_index()
    positive = sorted_data.where(sorted_data > 0).ffill()
    previous_close = positive.reindex(lookup_dates, method="ffill")
    previous_close.index = data.index
    return previous_close


def calculate_ytd_change(data):
    """
    Calculate the Year-to-Date (YTD) percentage change for each column in the DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.

    Returns:
    pd.DataFrame: DataFrame containing the YTD percentage change (percentage points).
    """
    # Standard YTD is measured from the final valid close before January 1, not
    # from January's first observation (which would erase the first day's move).
    prior_year_close = _previous_period_positive_close(data, "year")
    ytd_change = _safe_pct_change(data, prior_year_close)
    ytd_change.columns = [f"{col}_YTD_change" for col in ytd_change.columns]

    return ytd_change


def calculate_mtd_change(data):
    """
    Calculate the Month-to-Date (MTD) percentage change for each column in the DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.

    Returns:
    pd.DataFrame: DataFrame containing the MTD percentage change for each column.
    """
    # Standard MTD is measured from the final valid close before the first of the
    # month, preserving the first day's move in every published MTD value.
    prior_month_close = _previous_period_positive_close(data, "month")
    mtd_change = _safe_pct_change(data, prior_month_close)
    mtd_change.columns = [f"{col}_MTD_change" for col in mtd_change.columns]

    return mtd_change


def calculate_yoy_change(data):
    """
    Calculate the Year-over-Year (YoY) percentage change for each column in the DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.

    Returns:
    pd.DataFrame: DataFrame containing the YoY percentage change for each column.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")

    # Look up the same calendar date one year earlier. A 365-row shift drifts after
    # leap days and is also wrong when a daily source has a missing row. DateOffset
    # maps February 29 to February 28 in a non-leap prior year.
    prior_year_dates = data.index - pd.DateOffset(years=1)
    prior_year = data.reindex(prior_year_dates)
    prior_year.index = data.index
    yoy_change = _safe_pct_change(data, prior_year)
    yoy_change.columns = [f"{col}_YOY_change" for col in yoy_change.columns]

    return yoy_change


def calculate_all_changes(data: pd.DataFrame, yoy_columns: list, periods: Optional[list] = None) -> pd.DataFrame:
    """
    Calculate 7-day, 90-day, MTD and YTD changes for every column, and YoY changes
    for `yoy_columns` only.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.
    yoy_columns (list): Columns that also get a year-over-year change.
    periods (list of int, optional): Fixed day windows. Defaults to [7, 90].

    Returns:
    pd.DataFrame: The change columns only, in percentage points.
    """
    if periods is None:
        periods = [7, 90]

    return pd.concat(
        [
            calculate_time_changes(data, periods),
            calculate_ytd_change(data),
            calculate_mtd_change(data),
            calculate_yoy_change(data[yoy_columns]),
        ],
        axis=1,
    )


def calculate_time_changes(data, periods):
    """
    Calculate percentage changes for the given periods for each column in the DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.
    periods (list of int): List of time periods (in days) for which to calculate percentage changes.

    Returns:
    pd.DataFrame: DataFrame containing the calculated percentage changes for each specified period.
    """
    # Return all fixed-window changes in percentage-point format, consistent with MTD/YTD.
    changes = pd.concat(
        [
            _safe_pct_change(data, data.shift(period)).add_suffix(f"_{period}_change")
            for period in periods
        ],
        axis=1,
    )

    return changes


# Create Market Statistics


CORRELATION_PERIODS = [7, 30, 90, 365]

# Fewest paired returns a window may hold and still publish a correlation.
MIN_CORRELATION_RETURNS = 3


def observed_market_values(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Return `columns` with every value that was not a real source observation masked.

    Market fetchers bridge weekends and holidays with bounded fills and record each value's
    true observation date in a temporary marker column. Keeping only rows whose marker
    equals the row's own date recovers the asset's actual trading days, so a carried-forward
    Friday close cannot pose as a flat Saturday. Columns without a marker (on-chain series
    such as ``price_close``) are returned unchanged. Must run before
    ``forward_fill_market_data`` removes the markers.
    """
    frame = data.reindex(columns=columns).copy()
    row_dates = pd.Series(_normalized_index(data), index=data.index)
    for column in columns:
        marker_column = _source_observation_column(column)
        if marker_column not in data.columns:
            continue
        source_dates = pd.to_datetime(data[marker_column], errors="coerce").dt.normalize()
        frame[column] = frame[column].where(source_dates.eq(row_dates))
    return frame


def _paired_return_correlation(btc, asset, as_of, period):
    """Correlate BTC and one asset over returns measured between the asset's own observations.

    Both returns in each pair span the same interval (e.g. Friday to Monday for an equity),
    so weekends neither add fake zero returns nor misalign the Monday move. The window must
    be fully covered and the asset must have traded recently; otherwise the result is NaN.
    """
    pair = pd.concat([btc, asset], axis=1).loc[:as_of].dropna()
    if pair.empty:
        return np.nan
    window_start = as_of - pd.Timedelta(days=period)
    stale_before = as_of - pd.Timedelta(days=MARKET_DATA_MAX_FFILL_DAYS)
    if pair.index[0] > window_start or pair.index[-1] < stale_before:
        return np.nan

    returns = (
        pair.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .loc[lambda frame: frame.index > window_start]
        .dropna()
    )
    if len(returns) < MIN_CORRELATION_RETURNS:
        return np.nan
    return returns.iloc[:, 0].corr(returns.iloc[:, 1])


# Calculate Custom Datasets


def create_btc_correlation_data(
    report_date, tickers, correlations_data, periods=CORRELATION_PERIODS
):
    """
    Calculate Bitcoin's return correlation with every tracked asset as of the report date.

    For each asset, returns are measured between consecutive dates on which the asset has a
    real observation, and BTC's return is measured over exactly the same span. Windows are
    calendar-day lookbacks (7, 30, 90, 365 days) ending at the as-of date, which is the
    report date or, if absent, the latest earlier row.

    Parameters:
    report_date (str or pd.Timestamp): As-of date for the correlation snapshot.
    tickers (dict): Asset ticker dictionary from data_definitions.py.
    correlations_data (pd.DataFrame): DatetimeIndex frame with price_close and {ticker}_close
                                      columns holding only real observations — use
                                      ``observed_market_values`` before forward-filling.

    Returns:
    dict: Keys "price_close_{period}_days". Each value is a one-row DataFrame indexed
          ["price_close"] with one column per asset ({ticker}_close); values run -1 to +1
          and are NaN when the window lacks coverage. Bitcoin's own correlation is 1.0.
    """
    report_date = pd.to_datetime(report_date)
    all_tickers = [ticker for ticker_list in tickers.values() for ticker in ticker_list]
    ticker_list_with_suffix = ["price_close"] + [
        f"{ticker}_close" for ticker in all_tickers
    ]
    ticker_list_with_suffix = list(dict.fromkeys(ticker_list_with_suffix))

    filtered_data = correlations_data.reindex(columns=ticker_list_with_suffix).dropna(
        subset=["price_close"]
    )
    filtered_data = filtered_data.apply(pd.to_numeric, errors="coerce").sort_index()

    btc_correlations = {
        f"price_close_{p}_days": pd.DataFrame(
            index=["price_close"], columns=ticker_list_with_suffix, dtype=float
        )
        for p in periods
    }
    available = filtered_data.index[filtered_data.index <= report_date]
    if len(available) == 0:
        return btc_correlations
    as_of = available.max()

    btc = filtered_data["price_close"]
    for period in periods:
        result = btc_correlations[f"price_close_{period}_days"]
        for column in ticker_list_with_suffix:
            if column == "price_close":
                result.loc["price_close", column] = 1.0
                continue
            result.loc["price_close", column] = _paired_return_correlation(
                btc, filtered_data[column], as_of, period
            )

    return btc_correlations


# =============================================================================
# CHART-READY COMPUTE FUNCTIONS
# =============================================================================


def _ordinal(n: int) -> str:
    """Return 1 -> '1st', 2 -> '2nd', 11 -> '11th'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _era_label(position: int) -> str:
    """Halving era name by position (0 = genesis era)."""
    return "Genesis Era" if position == 0 else f"{_ordinal(position + 1)} Era"


def _build_period_bounds(boundaries, data_end, label_func):
    """
    Turn ascending boundary dates into (label, start, end) windows.

    Windows are half-open — [start, end) — so a date that both closes one period and
    opens the next is attributed to exactly one of them. The final window extends one
    day past `data_end` so the last observation is retained.

    Returns:
    list[tuple]: (label, start Timestamp, end Timestamp) per period, oldest first.
    """
    data_end = pd.Timestamp(data_end)
    horizon = data_end + pd.Timedelta(days=1)
    periods = []

    for i, start in enumerate(boundaries):
        start = pd.Timestamp(start)
        if start > data_end:
            break
        end = min(pd.Timestamp(boundaries[i + 1]), horizon) if i + 1 < len(boundaries) else horizon
        periods.append((label_func(i), start, end))

    return periods


# Market cycle boundaries define the search window for each cycle. The exported
# series is anchored to the lowest positive price actually observed inside that
# window, since a hand-entered boundary can lead the final low by a few days (and
# an open cycle can make a later low before it is complete).
BITCOIN_CYCLE_LOW_DATES = [
    "2010-07-25",
    "2011-11-18",
    "2015-01-15",
    "2018-12-16",
    "2022-11-20",
    "2026-02-06",
]

# Drawdown cycles run from an all-time high until that high is reclaimed, so unlike
# market cycles they are NOT contiguous — the gaps between them are the periods spent
# at new highs. Each entry is (label, ATH date, recovery date); the open cycle's end is
# supplied from the data rather than hardcoded.
BITCOIN_DRAWDOWN_CYCLES = [
    ("Drawdown Cycle 1", "2011-06-08", "2013-02-28"),
    ("Drawdown Cycle 2", "2013-11-29", "2017-03-03"),
    ("Drawdown Cycle 3", "2017-12-17", "2020-12-16"),
    ("Drawdown Cycle 4", "2021-11-10", "2024-03-04"),
    ("Drawdown Cycle 5", "2025-10-06", None),  # open — ends at latest data
]


def compute_drawdowns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Long-form drawdown series:
      - days_since_ath
      - drawdown_pct (0 at ATH, negative when below ATH)
      - Cycle (label)

    Aligns each cycle to its start ATH date.
    """
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if df.empty:
        return pd.DataFrame(columns=["days_since_ath", "drawdown_pct", "Cycle"])

    data_end = df.index.max()
    out = []

    for cycle_name, start_date, end_date in BITCOIN_DRAWDOWN_CYCLES:
        start_dt = pd.to_datetime(start_date)
        # An open cycle ends at the latest observation. Deriving it from the data rather
        # than today's clock keeps the export deterministic and stops the window running
        # past the data whenever the pipeline is re-run or replayed.
        end_dt = data_end if end_date is None else pd.to_datetime(end_date)

        period = df.loc[(df.index >= start_dt) & (df.index <= end_dt)].copy()
        if period.empty:
            continue

        # ATH path within the period
        period["ath"] = period["price_close"].cummax()

        # Drawdown percent
        period["drawdown_pct"] = (period["price_close"] / period["ath"] - 1.0) * 100.0

        # Days since the cycle's ATH start anchor (your start_dt)
        period["days_since_ath"] = (period.index - start_dt).days

        period["Cycle"] = cycle_name

        out.append(period[["days_since_ath", "drawdown_pct", "Cycle"]])

    if not out:
        return pd.DataFrame(columns=["days_since_ath", "drawdown_pct", "Cycle"])

    return pd.concat(out, ignore_index=True)


def compute_cycle_lows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market cycle performance indexed from cycle lows.

    Returns DataFrame with:
      - days_since_cycle_low
      - index_value (1.0 at cycle low, 2.0 = 2x gain, etc.)
      - Cycle (label)
    """
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if df.empty:
        return pd.DataFrame(columns=["days_since_cycle_low", "index_value", "Cycle"])

    cycle_periods = _build_period_bounds(
        BITCOIN_CYCLE_LOW_DATES,
        df.index.max(),
        lambda i: f"Market Cycle {i + 1}",
    )

    out = []
    for cycle_name, start_dt, end_dt in cycle_periods:
        # Half-open window: each cycle low both ends one cycle and begins the next, so an
        # inclusive end would emit that date twice under two different cycle labels.
        period = df.loc[(df.index >= start_dt) & (df.index < end_dt)].copy()
        if period.empty:
            continue

        valid_prices = period["price_close"].dropna()
        valid_prices = valid_prices[valid_prices > 0]
        if valid_prices.empty:
            continue

        # Use the actual lowest positive observation in the window. Starting the
        # export at that row guarantees the documented 1.0 floor and avoids
        # presenting a provisional/configured boundary as a confirmed cycle low.
        low_date = valid_prices.idxmin()
        low_px = float(valid_prices.loc[low_date])
        period = period.loc[period.index >= low_date].copy()

        period["days_since_cycle_low"] = (period.index - low_date).days
        period["index_value"] = period["price_close"] / low_px
        period["Cycle"] = cycle_name

        out.append(period[["days_since_cycle_low", "index_value", "Cycle"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["days_since_cycle_low", "index_value", "Cycle"]
    )


def compute_halving_days(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single long-form dataframe with:
      - days_since_halving
      - index_value (cycle index; 1.0 at halving, 2.0 = 2x, etc.)
      - Era (string name)
    """
    # Ensure datetime index
    data = data.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Ensure sorted
    data = data.sort_index()

    if data.empty:
        return pd.DataFrame(columns=["days_since_halving", "index_value", "Era"])

    # Eras are derived from the halving schedule rather than hardcoded, so the next
    # halving splits a new era automatically instead of silently stretching the current
    # one across two halvings.
    data_end = data.index.max()
    eras = _build_period_bounds(bitcoin_halving_dates(through=data_end), data_end, _era_label)

    out = []

    for era_name, start_dt, end_dt in eras:
        # Half-open window: the halving date itself belongs to the era it starts, so
        # closing the previous era inclusively would duplicate every boundary date.
        period = data.loc[(data.index >= start_dt) & (data.index < end_dt)].copy()
        if period.empty:
            continue

        # A halving comparison must have a real price on the halving boundary.
        # Genesis predates the first positive source price, so silently anchoring it
        # hundreds of days later mislabels both the x-axis and the 1.0 baseline.
        # Omit such an era instead; normal halving eras retain day 0 at index 1.0.
        valid_prices = period["price_close"].dropna()
        valid_prices = valid_prices[valid_prices > 0]
        if valid_prices.empty or start_dt not in valid_prices.index:
            continue

        start_px = float(valid_prices.loc[start_dt])

        period["days_since_halving"] = (period.index - start_dt).days
        period["index_value"] = period["price_close"] / start_px  # 1.0 at halving
        period["Era"] = era_name

        out.append(period[["days_since_halving", "index_value", "Era"]])

    if not out:
        return pd.DataFrame(columns=["days_since_halving", "index_value", "Era"])

    return pd.concat(out, ignore_index=True)
