"""
Data Format Module - Bitcoin Analytics Data Pipeline

This module handles all data fetching, transformation, and metric calculation for Bitcoin
market and on-chain analytics. It integrates multiple data sources and computes derived
metrics used throughout the reporting pipeline.

Data Sources:
    - BRK (Bitview): On-chain metrics, difficulty, supply data
    - Yahoo Finance: Equities, ETFs, indices, commodities, forex
    - CoinGecko: Altcoin prices, market caps, dominance
    - BRK: Bitcoin OHLC price data
    - Alternative.me: Fear & Greed Index
    - Google Sheets: Miner efficiency data
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
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
    PUE,
    ELEC_TO_TOTAL_COST_RATIO,
    MINER_DATA_SHEET_URL,
    API_TIMEOUT,
    SATS_PER_BTC,
    market_cap_history_start_date,
    yahoo_market_cap_fx_tickers,
    yahoo_share_ticker_aliases,
    BITCOIN_GENESIS_DATE,
    METCALFE_ADDRESS_COLUMNS,
    HASH_RIBBON_FAST_WINDOW,
    HASH_RIBBON_SLOW_WINDOW,
)
import os


# Ordinary market feeds should bridge weekends and short exchange holidays, not outages.
# Five calendar days covers those expected gaps while ensuring a stalled source becomes NaN.
MARKET_DATA_MAX_FFILL_DAYS = 5

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


def get_fear_and_greed_index() -> pd.DataFrame:
    """
    Fetches the Fear and Greed Index data from the Alternative.me API.

    Returns:
    pd.DataFrame: DataFrame containing the Fear and Greed Index data.
    """
    # URL to fetch the Fear and Greed Index data (limit=0 fetches all historical data)
    url = "https://api.alternative.me/fng/?limit=0"

    try:
        # Attempt to send a GET request to the URL
        response = requests.get(
            url, timeout=API_TIMEOUT
        )  # Set a timeout to avoid indefinite waits
        response.raise_for_status()  # Raise an error for unsuccessful status codes

        # Convert the JSON response to a dictionary
        data = response.json()
        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data["data"])
        df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["fear_greed_value"] = pd.to_numeric(df["value"], errors="coerce")
        df["fear_greed_classification"] = df["value_classification"]
        df = df[["fear_greed_value", "fear_greed_classification", "time"]]
        return df

    except (requests.exceptions.RequestException, KeyError) as e:
        # If an error occurs, return an empty DataFrame and print the error
        print(f"Failed to fetch Fear and Greed Index data. Reason: {e}")
        return pd.DataFrame(
            columns=["fear_greed_value", "fear_greed_classification", "time"]
        )


def get_bitcoin_dominance() -> pd.DataFrame:
    """
    Fetches the current Bitcoin dominance from the CoinGecko API.

    Returns:
    pd.DataFrame: DataFrame containing Bitcoin dominance and timestamp.
    """
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        bitcoin_dominance = data["data"]["market_cap_percentage"]["btc"]
        timestamp = pd.to_datetime(data["data"]["updated_at"], unit="s")

        df = pd.DataFrame(
            {"bitcoin_dominance": [bitcoin_dominance], "time": [timestamp]}
        )

        return df

    except requests.RequestException as e:
        print(f"Failed to fetch Bitcoin dominance: {e}")
        return pd.DataFrame(columns=["bitcoin_dominance", "time"])
    except (KeyError, ValueError) as e:
        print(f"Failed to parse Bitcoin dominance data: {e}")
        return pd.DataFrame(columns=["bitcoin_dominance", "time"])


def assert_ohlc_usable(ohlc_data: pd.DataFrame, label: str = "OHLC") -> None:
    """Raise before publication when an OHLC frame has no complete numeric candle."""
    if ohlc_data is None or ohlc_data.empty:
        raise RuntimeError(f"{label} data is empty; refusing to overwrite OHLC outputs")

    missing = [column for column in OHLC_COLUMNS if column not in ohlc_data.columns]
    if missing:
        raise RuntimeError(
            f"{label} data is missing required columns {missing}; refusing to overwrite OHLC outputs"
        )

    numeric = ohlc_data[OHLC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        raise RuntimeError(
            f"{label} data contains no complete numeric candles; refusing to overwrite OHLC outputs"
        )


def get_brk_ohlc(index: str = "week1", start: str = "2017-01-01") -> pd.DataFrame:
    """
    Fetch historical Bitcoin OHLC data from BRK.

    Parameters:
    index (str): BRK index to fetch, such as "week1" or "day1".
    start (str): Start date for the series query.

    Returns:
    pd.DataFrame: DataFrame indexed by BRK date labels with Open, High, Low, Close columns.
    """
    base_url = "https://bitview.space/api/series"
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

        dates = date_response.json()["data"]
        ohlc_rows = ohlc_response.json()["data"]

        if not dates or not ohlc_rows:
            raise ValueError(
                f"BRK returned no {index} OHLC observations for start={start}"
            )

        if len(dates) != len(ohlc_rows):
            raise ValueError(
                f"BRK date/OHLC length mismatch: {len(dates)} dates vs {len(ohlc_rows)} rows"
            )

        df = pd.DataFrame(ohlc_rows, columns=["Open", "High", "Low", "Close"])
        df["Time"] = pd.to_datetime(dates)
        df.set_index("Time", inplace=True)
        df = df.astype(float).sort_index()
        assert_ohlc_usable(df, label=f"BRK {index} OHLC")
        return df

    except (requests.RequestException, KeyError, TypeError, ValueError) as e:
        raise RuntimeError(
            f"Failed to fetch usable BRK {index} OHLC data from start={start}: {e}"
        ) from e


def get_btc_trade_volume_14d() -> pd.DataFrame:
    """
    Fetches the past 14 days of Bitcoin trade volume from CoinGecko.

    Returns:
    pd.DataFrame: DataFrame with daily Bitcoin trade volume.
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "14", "interval": "daily"}

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()

        volume_data = response.json()["total_volumes"]
        df = pd.DataFrame(volume_data, columns=["time", "btc_trading_volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        return df

    except requests.RequestException as e:
        print(f"Failed to fetch Bitcoin trading volume: {e}")
        return pd.DataFrame(columns=["time", "btc_trading_volume"])
    except (KeyError, ValueError) as e:
        print(f"Failed to parse Bitcoin trading volume data: {e}")
        return pd.DataFrame(columns=["time", "btc_trading_volume"])


def get_crypto_data(ticker_list: list) -> pd.DataFrame:
    """
    Fetches historical daily data for a list of cryptocurrencies from the CoinGecko API.

    Parameters:
    ticker_list (list): List of CoinGecko-compatible cryptocurrency tickers.

    Returns:
    pd.DataFrame: DataFrame containing merged close prices, volumes, and market caps.
    """
    data_frames = []  # Collect all DataFrames for efficient concatenation
    max_retries = 5  # Maximum number of retries per ticker
    initial_retry_delay = 60  # Initial delay in seconds for retry attempts

    for ticker in ticker_list:
        success = False
        retries = 0
        retry_delay = initial_retry_delay

        while not success and retries < max_retries:
            try:
                # Define API endpoint and parameters
                url = f"https://api.coingecko.com/api/v3/coins/{ticker}/market_chart"
                params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
                response = requests.get(url, params=params, timeout=API_TIMEOUT)
                response.raise_for_status()

                # Parse JSON response into DataFrames
                json_data = response.json()
                prices = pd.DataFrame(
                    json_data["prices"], columns=["time", f"{ticker}_close"]
                )
                volumes = pd.DataFrame(
                    json_data["total_volumes"], columns=["time", f"{ticker}_volume"]
                )
                market_caps = pd.DataFrame(
                    json_data["market_caps"], columns=["time", f"{ticker}_market_cap"]
                )

                # Convert timestamps to datetime
                prices["time"] = pd.to_datetime(prices["time"], unit="ms")
                volumes["time"] = pd.to_datetime(volumes["time"], unit="ms")
                market_caps["time"] = pd.to_datetime(market_caps["time"], unit="ms")

                # Merge DataFrames on the 'time' column
                merged_data = pd.merge(prices, volumes, on="time")
                merged_data = pd.merge(merged_data, market_caps, on="time")
                merged_data.set_index("time", inplace=True)

                # Retain the true API observation date through daily reindexing. Without
                # this marker, a second fill after merging could mistake a repeated value
                # for a fresh observation and extend it beyond the configured age limit.
                for value_column in list(merged_data.columns):
                    source_dates = pd.Series(
                        merged_data.index, index=merged_data.index
                    ).where(merged_data[value_column].notna())
                    merged_data[_source_observation_column(value_column)] = source_dates

                # Collect the merged data
                data_frames.append(merged_data)

                success = True  # Set success flag to True after successful data fetch

            except requests.HTTPError as http_err:
                if http_err.response.status_code == 429:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff for rate limits
                    retries += 1
                else:
                    print(f"HTTP error for {ticker}: {http_err}")
                    break  # Break the loop for non-429 HTTP errors
            except Exception as err:
                print(f"An error occurred for {ticker}: {err}")
                break

        if not success:
            print(f"Failed to fetch data for {ticker} after {max_retries} retries.")

        # Delay between requests to avoid hitting API rate limits
        time.sleep(1)

    # Concatenate all DataFrames at once (O(n) instead of O(n²))
    if data_frames:
        data = pd.concat(data_frames, axis=1)
        # Resample to fill any missing daily data
        data = (
            data.resample("D")
            .ffill(limit=MARKET_DATA_MAX_FFILL_DAYS)
            .reset_index()
        )
    else:
        data = pd.DataFrame()

    return data


def get_price(tickers: dict, start_date: str) -> pd.DataFrame:
    """
    Fetches historical close prices for all tickers using a single yf.download() batch call.

    Batching all tickers into one request is significantly faster than fetching each ticker
    individually. CoinGecko-sourced crypto tickers (ethereum, ripple, etc.) are excluded
    because they are fetched separately by get_crypto_data().

    Parameters:
    tickers (dict): Dictionary with categories as keys and ticker lists as values.
    start_date (str): Start date for fetching historical data (format: 'YYYY-MM-DD').

    Returns:
    pd.DataFrame: DataFrame containing close prices for all tickers with 'time' column.
    """
    end_date = datetime.today().strftime("%Y-%m-%d")
    excluded_crypto_tickers = {
        "ethereum",
        "ripple",
        "dogecoin",
        "binancecoin",
        "tether",
    }

    # Build flat list of tickers, excluding those sourced from CoinGecko
    fetch_tickers = [
        ticker
        for category, ticker_list in tickers.items()
        for ticker in ticker_list
        if not (category == "crypto" and ticker.lower() in excluded_crypto_tickers)
    ]

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
        pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
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
    save_csv: bool = True,
    out_path: str = "csv/brk_onchain_raw.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pull BRK metrics, align by timestamp (included in every chunk), optionally save raw CSV,
    then return a pandas DataFrame with a 'time' column using native BRK field names.
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

    raw_parts = []  # keep each raw CSV response if you want to debug / concatenate
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

        for header, rows, raw_csv in responses:
            raw_parts.append(raw_csv.strip())

            for r in rows:
                ts = r[0]
                d = data.setdefault(ts, {"timestamp": ts})
                for k, v in zip(header[1:], r[1:]):
                    d[k] = v

            for c in header[1:]:
                if c not in ordered_cols:
                    ordered_cols.append(c)

    if verbose:
        print(f"[BRK] merged rows: {len(data)}")
        print(f"[BRK] merged cols: {len(ordered_cols)}")
        print(f"[BRK] cols: {ordered_cols}")

    # build a single CSV (date derived later in your pipeline; we keep time + metrics)
    lines = []
    lines.append(",".join(ordered_cols))
    for ts in sorted(data, key=lambda x: int(float(x))):
        row = [data[ts].get(c, "") for c in ordered_cols]
        lines.append(",".join(map(str, row)))
    merged_csv = "\n".join(lines)

    if save_csv:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as f:
            f.write(merged_csv + "\n")
        if verbose:
            print(f"[BRK] saved raw CSV -> {out_path}")

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
]


def _ordinary_market_columns(data: pd.DataFrame) -> list:
    """Return externally observed non-miner series subject to the short fill budget."""
    onchain_columns = {metric for metric in BRK_METRICS if metric != "timestamp"}
    excluded = onchain_columns | set(MINER_EFFICIENCY_COLUMNS) | {"block_reward"}
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

    This check must run before `forward_fill_market_data`. Price and crypto fetchers retain
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

    as_of = available.max()
    if (report_date - as_of).days > 1:
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


def get_data(tickers: dict, start_date: str) -> pd.DataFrame:
    """
    Primary data orchestration function that fetches and merges all data sources into unified dataset.

    This is the main entry point for data ingestion. It coordinates API calls to 8 different data
    sources, normalizes timestamps to UTC midnight, and performs left-join merges to create a
    complete time-series dataset. The resulting DataFrame contains 400+ columns spanning on-chain
    metrics, market prices, market caps, sentiment indicators, and crypto altcoin data.

    Data Sources Integrated:
    1. BRK (Bitview) API: Bitcoin on-chain metrics (difficulty, hash rate, supply, fees, etc.)
    2. Yahoo Finance: Stock/ETF/commodity/forex prices via yfinance library
    3. Yahoo Finance: Market capitalizations for public companies
    4. Alternative.me: Fear & Greed Index sentiment indicator
    5. Google Sheets: Monthly Coin Metrics Bitcoin network efficiency data, forward-filled daily (J/GH)
    6. CoinGecko: Bitcoin dominance percentage
    7. CoinGecko: 14-day Bitcoin trade volume
    8. CoinGecko: Altcoin prices (ETH, XRP, DOGE, BNB, USDT)

    Parameters:
    tickers (dict): Asset ticker dictionary from data_definitions.py with keys:
                    'stocks', 'etfs', 'indices', 'commodities', 'forex', 'crypto'.
                    Example: {"stocks": ["AAPL", "MSFT"], "crypto": ["ethereum"]}
    start_date (str): Historical data start date in 'YYYY-MM-DD' format. Typically '2010-01-01'
                      to capture maximum history from Yahoo Finance. BRK data starts ~2009.
    """
    # Fetch data
    coindata = get_brk_onchain(start_date)
    prices = get_price(tickers, start_date)
    marketcaps = get_marketcap(tickers, start_date)
    fear_greed_index = get_fear_and_greed_index()
    miner_data = get_miner_data()  # Monthly Coin Metrics network efficiency, forward-filled daily
    bitcoin_dominance = get_bitcoin_dominance()
    btc_trade_volume_14d = get_btc_trade_volume_14d()
    crypto_data = get_crypto_data(tickers["crypto"])

    if not bitcoin_dominance.empty and "time" in bitcoin_dominance.columns:
        bitcoin_dominance["time"] = pd.to_datetime(bitcoin_dominance["time"]).dt.normalize()
        if not coindata.empty and "time" in coindata.columns:
            latest_data_date = pd.to_datetime(coindata["time"]).max().normalize()
            dominance_value = bitcoin_dominance["bitcoin_dominance"].iloc[-1]
            dominance_source_date = bitcoin_dominance["time"].iloc[-1]
            bitcoin_dominance = pd.DataFrame(
                {
                    "bitcoin_dominance": [dominance_value, dominance_value],
                    "time": [latest_data_date - pd.Timedelta(days=1), latest_data_date],
                    _source_observation_column("bitcoin_dominance"): [
                        dominance_source_date,
                        dominance_source_date,
                    ],
                }
            ).drop_duplicates(subset=["time"], keep="last")

    datasets = [
        ("coindata", coindata),
        ("prices", prices),
        ("marketcaps", marketcaps),
        ("fear_greed_index", fear_greed_index),
        ("miner_data", miner_data),
        ("bitcoin_dominance", bitcoin_dominance),
        ("btc_trade_volume_14d", btc_trade_volume_14d),
        ("crypto_data", crypto_data),
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
    for name, dataset in processed_datasets.items():
        if name == "coindata":
            continue
        data = pd.merge(data, dataset, left_index=True, right_index=True, how="left")

    # Handle duplicates
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()]
    if data.index.duplicated().any():
        data = data[~data.index.duplicated()]

    # Hayes needs per-block subsidy in BTC/block. Infer it from Bitcoin's
    # deterministic halving schedule rather than external data.
    data["block_reward"] = _bitcoin_block_subsidy_from_time(data.index)

    return data


# Metric Calculation


def calculate_custom_on_chain_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive Bitcoin on-chain valuation and network health metrics.

    This function computes 40+ derived metrics including valuation models (MVRV, NVT, Thermocap,
    Stock-to-Flow), price moving averages, profitability indicators (NUPL), and miner revenue
    multiples. These metrics are essential for Bitcoin fundamental analysis and market cycle timing.

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
    rev_all_time = miner_revenue_usd.fillna(0).cumsum()
    nvt_adj = market_cap / transfer_volume
    nvt_adj_90 = market_cap / transfer_volume.rolling(90).mean()

    # Early source rows carry a 0.0 price placeholder from before Bitcoin had a market
    # price. Dividing by those publishes inf, which downstream consumers cannot chart:
    # any max()/min() over the column returns inf, and JSON encoders serialize
    # non-finite floats as null. Treat non-positive prices as missing instead.
    positive_price = price_close.where(price_close > 0)

    mvrv_ratio = market_cap / realized_cap
    nvt_price = (nvt_adj.rolling(window=365 * 2).median() * transfer_volume) / supply
    nvt_price_adj = (nvt_adj_90.rolling(window=365).median() * transfer_volume) / supply
    nvt_price_multiple = price_close / nvt_price
    ma_200_day = price_close.rolling(window=200).mean()

    miner_revenue_1y = miner_revenue_usd.rolling(window=365).sum()
    miner_revenue_4y = miner_revenue_usd.rolling(window=4 * 365).sum()

    # BRK provides utxos_over_1y_old_supply in BTC; divide by circulating supply for %
    supply_pct_1_year_plus = (data["utxos_over_1y_old_supply"] / supply) * 100
    illiquid_supply = (supply_pct_1_year_plus / 100) * supply

    # Reserve Risk pipeline: adjusted BDD -> VOCD -> MVOCD -> HODL bank -> reserve risk
    adjusted_bdd = data["coindays_destroyed_sum_24h"] / supply
    adjusted_bdd_mean = adjusted_bdd.expanding().mean()
    vocd = price_close * adjusted_bdd
    mvocd = vocd.rolling(window=30).median()
    daily_hodl_value = (price_close - mvocd).clip(lower=0)
    hodl_bank = daily_hodl_value.cumsum()

    # Average Cap and Delta Cap
    cumulative_market_cap = market_cap.cumsum()
    days_since_start = pd.Series(range(1, len(data) + 1), index=data.index)
    average_cap = cumulative_market_cap / days_since_start
    delta_cap = realized_cap - average_cap

    daily_returns = price_close.pct_change(fill_method=None)

    # Some upstream rows report supply-in-profit/loss in sats rather than BTC.
    supply_in_profit_btc = pd.Series(
        np.where(
            data["supply_in_profit"] > supply * 10,
            data["supply_in_profit"] / SATS_PER_BTC,
            data["supply_in_profit"],
        ),
        index=data.index,
    )
    supply_in_loss_btc = pd.Series(
        np.where(
            data["supply_in_loss"] > supply * 10,
            data["supply_in_loss"] / SATS_PER_BTC,
            data["supply_in_loss"],
        ),
        index=data.index,
    )

    new_columns = {
        "RevAllTimeUSD": rev_all_time,
        "NVTAdj": nvt_adj,
        "NVTAdj90": nvt_adj_90,
        "sat_per_dollar": SATS_PER_BTC / positive_price,
        "mvrv_ratio": mvrv_ratio,
        "CapMVRVCur": mvrv_ratio,
        "nupl": (market_cap - realized_cap) / market_cap,
        "nvt_price": nvt_price,
        "nvt_price_adj": nvt_price_adj,
        "nvt_price_multiple": nvt_price_multiple,
        "nvt_price_multiple_ma": nvt_price_multiple.rolling(window=14).mean(),
        "7_day_ma_price_close": price_close.rolling(window=7).mean(),
        "50_day_ma_price_close": price_close.rolling(window=50).mean(),
        "100_day_ma_price_close": price_close.rolling(window=100).mean(),
        "200_day_ma_price_close": ma_200_day,
        "200_week_ma_price_close": price_close.rolling(window=200 * 7).mean(),
        "200_day_multiple": price_close / ma_200_day,
        "thermocap_multiple": market_cap / rev_all_time,
        "thermocap_price": rev_all_time / supply,
        "thermocap_price_multiple_4": (4 * rev_all_time) / supply,
        "thermocap_price_multiple_8": (8 * rev_all_time) / supply,
        "thermocap_price_multiple_16": (16 * rev_all_time) / supply,
        "thermocap_price_multiple_32": (32 * rev_all_time) / supply,
        "miner_revenue_1_Year": miner_revenue_1y,
        "miner_revenue_4_Year": miner_revenue_4y,
        "ss_multiple_1": market_cap / miner_revenue_1y,
        "ss_price_1": miner_revenue_1y / supply,
        "ss_multiple_4": market_cap / miner_revenue_4y,
        "ss_price_4": miner_revenue_4y / supply,
        "realizedcap_multiple_2": (2 * realized_cap) / supply,
        "realizedcap_multiple_3": (3 * realized_cap) / supply,
        "realizedcap_multiple_5": (5 * realized_cap) / supply,
        "realizedcap_multiple_7": (7 * realized_cap) / supply,
        "supply_pct_1_year_plus": supply_pct_1_year_plus,
        "pct_supply_issued": supply / 21000000,
        "pct_fee_of_reward": (data["fees_sum_24h"] / data["coinbase_sum_24h"]) * 100,
        "illiquid_supply": illiquid_supply,
        "liquid_supply": supply - illiquid_supply,
        # active_addrs_average_24h is already a daily total — no block-count scaling.
        "daily_active_addresses_sending": data["active_addrs_average_24h"],
        "adjusted_bdd": adjusted_bdd,
        "adjusted_bdd_mean": adjusted_bdd_mean,
        "adjusted_bdd_above_avg": adjusted_bdd > adjusted_bdd_mean,
        "vocd": vocd,
        "mvocd": mvocd,
        "daily_hodl_value": daily_hodl_value,
        "hodl_bank_calc": hodl_bank,
        "reserve_risk_calc": price_close / hodl_bank,
        "cumulative_market_cap": cumulative_market_cap,
        "days_since_start": days_since_start,
        "average_cap": average_cap,
        "delta_cap": delta_cap,
        "average_cap_price": average_cap / supply,
        "delta_cap_price": delta_cap / supply,
        "VtyDayRet30d": daily_returns.rolling(30).std() * np.sqrt(365),
        "VtyDayRet180d": daily_returns.rolling(180).std() * np.sqrt(365),
        "supply_in_profit_btc": supply_in_profit_btc,
        "supply_in_loss_btc": supply_in_loss_btc,
        "supply_in_profit_pct": (supply_in_profit_btc / supply) * 100,
        "supply_in_loss_pct": (supply_in_loss_btc / supply) * 100,
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
    Calculate 7-day, 30-day, and 365-day moving averages for specified metrics.

    This function creates smoothed time series for on-chain metrics to reduce daily volatility
    and identify trends. Moving averages are used throughout the pipeline for analysis and
    visualization. The function adds 3 new columns per input metric.

    Parameters:
    data (pd.DataFrame): DataFrame with DatetimeIndex containing the metrics to smooth.
                         Must include all column names specified in the metrics list.
    metrics (list): List of column names to calculate moving averages for. Typically includes:
                    hash_rate, daily_active_addresses_sending, tx_count_sum_24h, transfer_volume_sum_24h_usd, fees_average_24h_usd,
                    subsidy_sum_24h, coinbase_sum_24h_usd, nvt_price, nvt_price_adj.
                    Defined in data_definitions.moving_avg_metrics.
    """
    moving_averages = {
        f"7_day_ma_{metric}": data[metric].rolling(window=7).mean()
        for metric in metrics
    }
    moving_averages.update(
        {
            f"30_day_ma_{metric}": data[metric].rolling(window=30).mean()
            for metric in metrics
        }
    )
    moving_averages.update(
        {
            f"365_day_ma_{metric}": data[metric].rolling(window=365).mean()
            for metric in metrics
        }
    )

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


def calculate_gold_market_cap_breakdown(
    data: pd.DataFrame, gold_supply_breakdown: pd.DataFrame
) -> pd.DataFrame:
    """
    Break down the gold market cap into different categories and add the results to the DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing existing financial data.
    gold_supply_breakdown (pd.DataFrame): DataFrame containing breakdown percentages for gold supply.

    Returns:
    pd.DataFrame: DataFrame with added columns for each category's market cap.
    """
    # Use the latest value of gold market cap
    gold_marketcap_billion_usd = data["gold_marketcap_billion_usd"].iloc[-1]

    for _, row in gold_supply_breakdown.iterrows():
        category = row["Gold Supply Breakdown"]
        percentage_of_market = row["Percentage Of Market"]
        category_marketcap_billion_usd = gold_marketcap_billion_usd * (
            percentage_of_market / 100.0
        )

        # Create the metric name for the category
        metric_name = (
            "gold_marketcap_" + category.replace(" ", "_").lower() + "_billion_usd"
        )

        # Assign the calculated value to all rows in the new column
        data[metric_name] = category_marketcap_billion_usd

    # Explicitly check if the index is a DatetimeIndex; fix if needed
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except ValueError as e:
            print(f"Failed to convert index back to DatetimeIndex: {e}")

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
    # Ensure 'supply' is forward filled to avoid NaN values
    data["supply"] = data["supply"].ffill()

    # Early return if 'supply' for the latest row is zero or NaN to avoid division by zero
    if data["supply"].iloc[-1] == 0 or pd.isna(data["supply"].iloc[-1]):
        print(
            "Warning: 'supply' is zero or NaN for the latest row. Skipping calculations."
        )
        return data

    new_columns = {}  # Use a dictionary to store new columns

    # Calculating BTC prices required to match or surpass gold market cap
    gold_marketcap_billion_usd = data["gold_marketcap_billion_usd"].iloc[-1]
    new_columns["gold_marketcap_btc_price"] = (
        gold_marketcap_billion_usd / data["supply"]
    )

    # Iterating through gold supply breakdown to calculate BTC prices for specific categories
    for _, row in gold_supply_breakdown.iterrows():
        category = row["Gold Supply Breakdown"].replace(" ", "_").lower()
        percentage_of_market = row["Percentage Of Market"] / 100.0
        new_columns[f"gold_{category}_marketcap_btc_price"] = (
            gold_marketcap_billion_usd * percentage_of_market
        ) / data["supply"]

    # Silver market cap calculations
    silver_marketcap_billion_usd = data["silver_marketcap_billion_usd"].iloc[-1]
    new_columns["silver_marketcap_btc_price"] = (
        silver_marketcap_billion_usd / data["supply"]
    )

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
        fiat_marketcap[f"{country}_cap"] = fiat_supply_usd

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


def calculate_stock_to_flow_metrics(data):
    """
    Calculate Bitcoin Stock-to-Flow (S2F) valuation model using PlanB's power law regression.

    Stock-to-Flow measures Bitcoin's scarcity by dividing existing supply (stock) by annual new
    issuance (flow). PlanB's model uses the power law: Market Value = exp(14.6) * S2F^3.3, which
    historically correlated with Bitcoin's price. The model predicts price increases as Bitcoin
    becomes scarcer through halvings (reducing flow every 4 years).

    Model Details:
    - Intercept: 14.6 (from PlanB's regression analysis)
    - Power coefficient: 3.3 (non-linear relationship between scarcity and value)
    - SF ratio calculated using 365-day supply change to smooth daily volatility
    - 365-day MA applied to predicted price for trend identification

    Parameters:
    data (pd.DataFrame): DataFrame with DatetimeIndex containing:
                         - supply: Total Bitcoin supply (from BRK API)
                         - price_close: Actual Bitcoin price (for multiple calculation)
    """
    # Initialize a dictionary to hold new columns
    new_columns = {}

    # Use PlanB's Stock-to-Flow model directly
    # PlanB model parameters: intercept and power coefficient are pre-determined
    intercept = 14.6
    power = 3.3

    # Calculate S2F using yearly supply difference to align with PlanB's original model.
    # The first 365 rows have no prior-year supply to difference against. Leaving that
    # warmup window as NaN keeps it out of the export; filling it with 0 would make the
    # denominator zero and publish inf into SF, SF_Predicted_Market_Value and
    # SF_Predicted_Price — and inf propagates into SF_Multiple as a plausible-looking 0.0.
    annual_supply_growth = data["supply"].diff(periods=365)
    annual_supply_growth = annual_supply_growth.where(annual_supply_growth > 0)
    new_columns["SF"] = data["supply"] / annual_supply_growth

    # Applying the PlanB linear regression formula
    new_columns["SF_Predicted_Market_Value"] = (
        np.exp(intercept) * new_columns["SF"] ** power
    )

    # Calculating the predicted market price using supply
    new_columns["SF_Predicted_Price"] = (
        new_columns["SF_Predicted_Market_Value"] / data["supply"]
    )

    # Apply a 365-day moving average to the predicted S2F price to smooth the curve
    new_columns["SF_Predicted_Price_MA365"] = (
        new_columns["SF_Predicted_Price"].rolling(window=365).mean()
    )

    # Calculating the S/F multiple using the actual price and the predicted price
    new_columns["SF_Multiple"] = data["price_close"] / new_columns["SF_Predicted_Price"]

    # Concatenate all new columns to the DataFrame at once
    data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)

    return data


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
    new_columns = {"days_since_genesis": days_since_genesis}

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
            "hash_ribbon_ratio": fast.div(slow.where(slow.ne(0))),
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
        - inflation_rate
        - subsidy_sum_24h
        - price_close

    Google Sheet input:
        - cm_efficiency_j_gh: Coin Metrics Labs monthly estimated Bitcoin network
          efficiency in J/GH, forward-filled daily.

    Internally derived:
        - block_reward from Bitcoin halving dates

    Model outputs:
        - daily_electricity_consumption_kwh: Daily network electricity consumption.
        - network_power_watts: Estimated fleet power draw.
        - miner_revenue_btc: Observed daily subsidy plus fees paid to miners.
        - Electricity_Cost_{3c..7c}: Power expense per BTC earned under tariff scenarios.
        - Electricity_Cost: Alias for the base $0.05/kWh power-expense scenario.
        - Electricity_Cost_PUE_Subsidy_Only: Legacy report calculation using PUE
          and subsidy only.
        - Bitcoin_Production_Cost: Legacy all-in production-cost estimate.
        - Hayes_Network_Price_Per_BTC: Hayes cost-of-production price per BTC.
        - Energy_Value: Capriole / Charles Edwards Energy Value in USD.
        - Energy_Value_Multiple: price_close / Energy_Value.
        - Lagged_Energy_Value and CM_Energy_Value: Backward-compatible aliases for Energy_Value.
    """
    FIAT_FACTOR = 2.0e-15
    SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
    SECONDS_PER_DAY = 24 * 60 * 60
    SHA_256_CONSTANT = 2**32

    hash_rate_th_s = data["hash_rate"] / 1e12

    # Main efficiency input: Coin Metrics monthly network efficiency in J/GH,
    # forward-filled daily from the Google Sheet.
    efficiency_j_gh = data["cm_efficiency_j_gh"]

    # Hayes uses deterministic protocol subsidy inferred from halving dates.
    data["block_reward"] = _bitcoin_block_subsidy_from_time(data.index)

    # H/s ÷ 1e9 gives GH/s; multiplying by J/GH gives J/s (watts).
    data["network_power_watts"] = data["hash_rate"] / 1e9 * efficiency_j_gh
    data["daily_electricity_consumption_kwh"] = (
        data["network_power_watts"] * 24 / 1000
    )
    data["miner_revenue_btc"] = data["subsidy_sum_24h"] + data["fees_sum_24h"]

    valid_revenue = data["miner_revenue_btc"] > 0
    for tariff in ELECTRICITY_TARIFFS_USD_PER_KWH:
        cents = int(round(tariff * 100))
        data[f"Electricity_Cost_{cents}c"] = (
            data["daily_electricity_consumption_kwh"] * tariff
        ).div(
            data["miner_revenue_btc"].where(valid_revenue)
        )

    base_cents = int(round(ELECTRICITY_BASE_TARIFF_USD_PER_KWH * 100))
    data["Electricity_Cost"] = data[f"Electricity_Cost_{base_cents}c"]
    data["power_only_breakeven_tariff_usd_per_kwh"] = (
        data["price_close"] * data["miner_revenue_btc"]
    ).div(
        data["daily_electricity_consumption_kwh"].where(
            data["daily_electricity_consumption_kwh"] > 0
        )
    )

    # Preserve the report's former PUE/subsidy-only calculation under an explicit
    # name. Bitcoin_Production_Cost continues to derive from this legacy model so
    # its historical meaning does not change.
    legacy_total_electricity_cost = (
        data["daily_electricity_consumption_kwh"]
        * ELECTRICITY_BASE_TARIFF_USD_PER_KWH
        * PUE
    )
    data["Electricity_Cost_PUE_Subsidy_Only"] = legacy_total_electricity_cost.div(
        data["subsidy_sum_24h"].where(data["subsidy_sum_24h"] > 0)
    )
    data["Bitcoin_Production_Cost"] = (
        data["Electricity_Cost_PUE_Subsidy_Only"] / ELEC_TO_TOTAL_COST_RATIO
    )

    btc_per_day_network_expected = (
        data["hash_rate"]
        * SECONDS_PER_DAY
        * data["block_reward"]
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

    supply_growth_rate_s = data["inflation_rate"] / 100 / SECONDS_PER_YEAR
    miner_efficiency_w_th = efficiency_j_gh * 1000
    energy_input_watts = hash_rate_th_s * miner_efficiency_w_th

    data["Energy_Value"] = np.where(
        supply_growth_rate_s != 0,
        (energy_input_watts / supply_growth_rate_s) * FIAT_FACTOR,
        np.nan,
    )

    data["Energy_Value_Multiple"] = np.where(
        data["Energy_Value"] != 0,
        data["price_close"] / data["Energy_Value"],
        np.nan,
    )

    # Backward compatibility for existing chart/table references.
    data["Lagged_Energy_Value"] = data["Energy_Value"]
    data["CM_Energy_Value"] = data["Energy_Value"]

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
    # Ensure that all data is numeric by coercing non-numeric values to NaN
    data = data.apply(pd.to_numeric, errors="coerce")

    # Calculate the start value for CAGR by shifting data backward by the number of years in days
    days_per_year = 365
    start_value = data.shift(int(years * days_per_year))

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


def calculate_rolling_cagr_for_all_metrics(data):
    """
    Calculate rolling Compound Annual Growth Rate (CAGR) for all metrics across 4-year and 2-year windows.

    This function computes annualized growth rates for every numeric column in the dataset, providing
    historical context for current values. CAGR smooths volatility and shows long-term trends, making
    it useful for valuation model projections (e.g., projecting Bitcoin price based on 4-year CAGR of
    realized price or thermocap price). The output is used in create_eoy_model_table for forecasting.

    CAGR Formula: ((End Value / Start Value)^(1/years) - 1) * 100

    Parameters:
    data (pd.DataFrame): DataFrame with DatetimeIndex containing numeric metrics to calculate growth rates.
                         Typically includes price_close, realized_price, thermocap_price, hash_rate, etc.

    Returns:
    pd.DataFrame: DataFrame with DatetimeIndex containing CAGR columns:
        - {metric}_4_Year_CAGR: Annualized growth rate over past 4 years (1460 days) for each metric
        - {metric}_2_Year_CAGR: Annualized growth rate over past 2 years (730 days) for each metric
        Values are percentages (5.0 = 5% annual growth). NaN for insufficient history.
    """
    # Calculate 4-year CAGR for all columns
    cagr_4yr = calculate_rolling_cagr_for_all_columns(data, 4)

    # Calculate 2-year CAGR for all columns
    cagr_2yr = calculate_rolling_cagr_for_all_columns(data, 2)

    # Concatenate the results to return a DataFrame containing both 4-year and 2-year CAGR metrics
    return pd.concat([cagr_4yr, cagr_2yr], axis=1)


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


def calculate_all_changes(data: pd.DataFrame, periods: Optional[list] = None) -> pd.DataFrame:
    """
    Calculate time-based changes for each column in the DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.
    periods (list of int, optional): List of time periods (in days) to calculate changes for.
                                     Defaults to [7, 90] which are the most commonly used.

    Returns:
    pd.DataFrame: DataFrame containing all calculated changes.
    """
    # Default to only the periods actually used in reports
    if periods is None:
        periods = [7, 90]

    # Calculate changes for the specified periods
    changes = calculate_time_changes(data, periods)

    # Calculate YTD, MTD, and YOY changes (needed for reports and charts)
    ytd_change = calculate_ytd_change(data)
    mtd_change = calculate_mtd_change(data)
    yoy_change = calculate_yoy_change(data)

    # Concatenate all changes into a single DataFrame
    changes = pd.concat([changes, ytd_change, mtd_change, yoy_change], axis=1)

    return changes


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


def calculate_statistics(data, start_date):
    """
    Calculate statistical metrics, including percentiles and z-scores, for the given data after a specified start date.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing financial or numerical data.
    start_date (str): The start date from which to filter data.

    Returns:
    tuple: Two DataFrames containing percentiles and z-scores, respectively.
    """
    # Convert start_date to datetime to ensure consistent filtering
    start_date = pd.to_datetime(start_date)

    # Filter data to only include rows after start_date
    data = data[data.index >= start_date]

    # Calculate percentiles and z-scores for numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    # Calculate percentiles for each numeric column
    percentiles = numeric_data.rank(pct=True)
    percentiles.columns = [str(col) + "_percentile" for col in percentiles.columns]

    # Calculate z-scores for each numeric column (standard score)
    z_scores = (numeric_data - numeric_data.mean()) / numeric_data.std()
    z_scores.columns = [str(col) + "_zscore" for col in z_scores.columns]

    return percentiles, z_scores


def run_data_analysis(data: pd.DataFrame, start_date: str, periods: Optional[list] = None, include_statistics: bool = False) -> pd.DataFrame:
    """
    Calculate time-based percentage changes (7d, 90d, MTD, YTD) for all metrics in the dataset.

    This is the primary analysis function that enriches raw data with calculated percentage changes
    across multiple time periods. These change columns are essential for performance tables and
    time-series analysis. Optionally includes percentile and z-score statistics.

    The function calculates:
    - Fixed period changes: 7-day, 90-day percentage changes
    - Month-to-date (MTD) changes: Performance since start of current month
    - Year-to-date (YTD) changes: Performance since January 1st of current year
    - Optional statistics: Percentile rankings and z-scores since start_date

    Parameters:
    data (pd.DataFrame): DataFrame with DatetimeIndex containing metrics to analyze. Typically
                         a subset of the full dataset containing only analysis_columns from
                         data_definitions.py (optimized to ~28 columns instead of 400+).
    start_date (str): Start date for statistical calculations in 'YYYY-MM-DD' format.
                      Only used if include_statistics=True. Typically '2012-11-28' (first halving).
    periods (Optional[list]): List of day periods for fixed-window changes. Default: [7, 90].
                              Custom periods can be specified (e.g., [7, 30, 90, 365]).
    include_statistics (bool): If True, calculates percentile and z-score for each metric relative
                               to historical data since start_date. Default: False (not used in
                               current reports but available for advanced analysis).

    Returns:
    pd.DataFrame: Original data with added change columns:
        - {column}_7_change: 7-day percentage change for each metric
        - {column}_90_change: 90-day percentage change for each metric
        - {column}_MTD_change: Month-to-date percentage change
        - {column}_YTD_change: Year-to-date percentage change
        - {column}_percentile: Percentile rank (0-1) if include_statistics=True
        - {column}_zscore: Standard score if include_statistics=True

    """
    # Calculate time-based changes for the data
    changes = calculate_all_changes(data, periods)

    # Merge the changes with the original data
    data = pd.concat([data, changes], axis=1)

    # Optionally include percentiles and z-scores (not used in current reports)
    if include_statistics:
        percentiles, z_scores = calculate_statistics(data, start_date)
        data = pd.concat([data, percentiles, z_scores], axis=1)

    return data


# Create Market Statistics


def calculate_rolling_correlations(data, periods):
    """
    Calculates rolling return correlations for specified periods.

    Parameters:
    data (pd.DataFrame): DataFrame containing historical daily price data for assets as columns.
    periods (list): List of integers representing rolling window sizes in days.

    Returns:
    dict: Dictionary where keys are periods and values are DataFrames of rolling correlations.
    """
    # Never let pct_change silently pad NaNs left by the bounded ingestion fills. Doing
    # so would recreate a zero return for stale assets and contaminate correlations.
    returns = data.pct_change(fill_method=None)

    # Initialize a dictionary to store rolling correlations for each period
    correlations = {}
    for period in periods:
        # Calculate rolling correlation of returns
        correlations[period] = returns.rolling(window=period).corr()

    return correlations


# Calculate Custom Datasets


def create_btc_correlation_data(report_date, tickers, correlations_data):
    """
    Calculate Bitcoin's rolling correlation coefficients with all tracked assets for a specific date.

    This function computes Bitcoin's price correlation with stocks, ETFs, commodities, forex, and
    altcoins across four rolling windows (7, 30, 90, 365 days). Correlations are used in performance
    tables to show which assets move together with Bitcoin. Values range from -1 (perfect negative
    correlation) to +1 (perfect positive correlation).

    The function handles missing data gracefully by using the nearest available date if the exact
    report_date is not in the dataset (useful for weekends/holidays).

    Parameters:
    report_date (str or pd.Timestamp): Target date for correlation snapshot in 'YYYY-MM-DD' format.
                                       If date not available, uses nearest prior date.
    tickers (dict): Asset ticker dictionary from data_definitions.py with structure:
                    {"stocks": [...], "etfs": [...], "indices": [...], "commodities": [...],
                     "forex": [...], "crypto": [...]}.
    correlations_data (pd.DataFrame): DataFrame with DatetimeIndex containing price_close (Bitcoin)
                                      and {ticker}_close columns for all assets. Typically filtered
                                      to correlation_data columns from data_definitions.py.

    Returns:
    dict: Dictionary with keys: "price_close_7_days", "price_close_30_days", "price_close_90_days",
          "price_close_365_days". Each value is a pandas Series with:
          - Index: Asset column names ({ticker}_close)
          - Values: Correlation coefficient with Bitcoin (-1 to +1)
          Missing data returns NaN. Bitcoin's correlation with itself is always 1.0.
    """
    report_date = pd.to_datetime(report_date)
    all_tickers = [ticker for ticker_list in tickers.values() for ticker in ticker_list]
    ticker_list_with_suffix = ["price_close"] + [
        f"{ticker}_close" for ticker in all_tickers
    ]

    filtered_data = correlations_data[ticker_list_with_suffix].dropna(
        subset=["price_close"]
    )

    if filtered_data.empty:
        empty_corr = pd.Series(
            index=[f"{ticker}_close" for ticker in all_tickers], dtype=float
        )
        return {f"price_close_{p}_days": empty_corr for p in [7, 30, 90, 365]}

    correlations = calculate_rolling_correlations(
        filtered_data, periods=[7, 30, 90, 365]
    )
    if report_date in filtered_data.index:
        closest_date = report_date
    else:
        prior_dates = filtered_data.index[filtered_data.index <= report_date]
        closest_date = (
            prior_dates.max() if len(prior_dates) else filtered_data.index.min()
        )

    btc_correlations = {}
    for period in [7, 30, 90, 365]:
        corr_df = correlations[period]
        try:
            if report_date in corr_df.index:
                btc_correlations[f"price_close_{period}_days"] = corr_df.loc[
                    report_date
                ].loc[["price_close"]]
            else:
                btc_correlations[f"price_close_{period}_days"] = corr_df.loc[
                    closest_date
                ].loc[["price_close"]]
        except KeyError:
            btc_correlations[f"price_close_{period}_days"] = pd.Series(
                index=[f"{ticker}_close" for ticker in all_tickers], dtype=float
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
