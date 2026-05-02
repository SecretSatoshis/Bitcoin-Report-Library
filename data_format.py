"""
Data Format Module - Bitcoin Analytics Data Pipeline

This module handles all data fetching, transformation, and metric calculation for Bitcoin
market and on-chain analytics. It integrates multiple data sources and computes derived
metrics used throughout the reporting pipeline.

Data Sources:
    - BRK (Bitview): On-chain metrics, difficulty, supply data
    - Yahoo Finance: Equities, ETFs, indices, commodities, forex
    - CoinGecko: Altcoin prices, market caps, dominance
    - Kraken: Bitcoin OHLC price data
    - Alternative.me: Fear & Greed Index
    - Google Sheets: Miner efficiency data
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO
import time
import csv, io
from typing import Optional
from data_definitions import (
    BRK_BULK_URL,
    BRK_METRICS,
    ELECTRICITY_COST,
    PUE,
    ELEC_TO_TOTAL_COST_RATIO,
    MINER_DATA_SHEET_URL,
    API_TIMEOUT,
    SATS_PER_BTC,
    STOCK_TRADING_DAYS,
    CRYPTO_TRADING_DAYS,
)
import os


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


def get_kraken_ohlc(pair: str, since: int) -> pd.DataFrame:
    """
    Fetches historical OHLC data from the Kraken API with retry logic.

    Parameters:
    pair (str): The Kraken asset pair (e.g., 'BTCUSD').
    since (int): Timestamp from which to fetch data.

    Returns:
    pd.DataFrame: DataFrame with OHLC data resampled to weekly intervals.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    interval = 1440  # Daily interval
    data_frames = []  # Use a list to collect data chunks for a single concat outside the loop
    max_retries = 3
    retry_delay = 5

    while True:
        retries = 0
        success = False

        while not success and retries < max_retries:
            try:
                params = {"pair": pair, "interval": interval, "since": since}
                response = requests.get(url, params=params, timeout=API_TIMEOUT)
                response.raise_for_status()

                data = response.json()
                if data.get("error"):
                    print(f"Error in Kraken data: {data['error']}")
                    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "VWAP", "Volume", "Count"])

                ohlc_data = data["result"][list(data["result"].keys())[0]]
                since = data["result"]["last"]

                temp_df = pd.DataFrame(
                    ohlc_data,
                    columns=[
                        "Time",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "VWAP",
                        "Volume",
                        "Count",
                    ],
                )
                temp_df["Time"] = pd.to_datetime(temp_df["Time"], unit="s", utc=True)
                data_frames.append(temp_df)

                success = True

                if len(ohlc_data) < 720:
                    # No more data to fetch
                    break

                time.sleep(1)

            except requests.RequestException as e:
                retries += 1
                if retries < max_retries:
                    print(f"Error fetching Kraken OHLC data (attempt {retries}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to fetch Kraken OHLC data after {max_retries} retries: {e}")
                    break

        if not success:
            # Failed after all retries, stop trying
            break

        # Break out of outer loop if we got all data
        if success and len(ohlc_data) < 720:
            break

    if not data_frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "VWAP", "Volume", "Count"])

    df = pd.concat(data_frames)
    df.set_index("Time", inplace=True)
    float_cols = ["Open", "High", "Low", "Close", "VWAP", "Volume"]
    df[float_cols] = df[float_cols].astype(float)
    df = df.resample("W-SUN").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "VWAP": "mean",
            "Volume": "sum",
            "Count": "sum",
        }
    )

    return df


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
        data = data.resample("D").ffill().reset_index()
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
            col = close_series.rename(f"{ticker}_close").to_frame()
            # Index is already tz-naive with auto_adjust=True; reindex to fill gaps
            col = col.reindex(date_range).ffill()
            data_frames.append(col)
        except KeyError:
            print(f"[yfinance] Could not extract {ticker} from batch result — skipping")

    if not data_frames:
        return pd.DataFrame(columns=["time"])

    data = pd.concat(data_frames, axis=1)
    data.reset_index(inplace=True)
    data.rename(columns={"index": "time"}, inplace=True)
    data["time"] = pd.to_datetime(data["time"]).dt.tz_localize(None)
    return data


def get_marketcap(tickers: dict, start_date: str) -> pd.DataFrame:
    """
    Fetches historical market capitalization data for the given stock tickers.

    Parameters:
    tickers (dict): A dictionary containing stock tickers.
    start_date (str): The start date for fetching historical data.

    Returns:
    pd.DataFrame: DataFrame containing the historical market cap data.
    """
    data_frames = []  # Collect DataFrames to avoid repeated merges
    # Set the end date to 'yesterday' to ensure data availability
    end_date = pd.to_datetime("today") - pd.Timedelta(days=1)

    for ticker in tickers["stocks"]:
        stock = yf.Ticker(ticker)
        try:
            # Fetch historical data from Yahoo Finance
            market_cap = stock.info.get("marketCap", None)
            if market_cap is not None:
                # Create a DataFrame for the market cap data
                mc_data = pd.DataFrame(
                    {
                        "time": pd.date_range(start=start_date, end=end_date),
                        f"{ticker}_MarketCap": [market_cap]
                        * len(pd.date_range(start=start_date, end=end_date)),
                    }
                )
                mc_data.set_index(
                    "time", inplace=True
                )  # Set time as the index for easier concatenation

                # Add DataFrame to the list
                data_frames.append(mc_data)
        except Exception as e:
            # Print error message for failed data fetches
            print(f"Could not fetch market cap data for {ticker}. Reason: {str(e)}")

    # Concatenate all DataFrames along the column axis to keep each ticker's market cap as separate columns
    if data_frames:
        data = pd.concat(data_frames, axis=1).reset_index()
    else:
        data = pd.DataFrame()
    return data


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

    Returns:
    pd.DataFrame: Daily DataFrame with columns `time` and `cm_efficiency_j_gh`.
                  Returns an empty DataFrame with those columns on error.
    """
    if not google_sheet_url:
        google_sheet_url = MINER_DATA_SHEET_URL

    try:
        # Convert Google Sheets sharing URL to CSV export URL.
        csv_export_url = google_sheet_url.replace("/edit?usp=sharing", "/export?format=csv")
        csv_export_url = csv_export_url.split("#")[0]
        if "/edit?" in csv_export_url:
            csv_export_url = csv_export_url.split("/edit?")[0] + "/export?format=csv"

        df = pd.read_csv(csv_export_url)
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
        df = df[["time", "cm_efficiency_j_gh"]]
        df = df.drop_duplicates(subset=["time"], keep="last")

        # Monthly Coin Metrics efficiency is used as the best available estimate
        # for each day until the next monthly observation is available.
        df = df.set_index("time").resample("D").ffill().reset_index()

        return df
    except Exception as e:
        print(f"Failed to fetch Coin Metrics network efficiency data from Google Sheets: {e}")
        return pd.DataFrame(columns=["time", "cm_efficiency_j_gh"])


def _bitcoin_block_subsidy_from_time(time_index) -> pd.Series:
    """
    Infer Bitcoin's protocol block subsidy in BTC per block from the date.

    Returns:
    pd.Series: Block subsidy in BTC per block, indexed like `time_index`.
    """
    dates = pd.to_datetime(time_index)
    reward = pd.Series(3.125, index=dates)
    reward.loc[dates < pd.Timestamp("2024-04-20")] = 6.25
    reward.loc[dates < pd.Timestamp("2020-05-11")] = 12.5
    reward.loc[dates < pd.Timestamp("2016-07-09")] = 25.0
    reward.loc[dates < pd.Timestamp("2012-11-28")] = 50.0
    return pd.Series(reward.values, index=time_index)


def _brk_error_code(response: requests.Response) -> Optional[str]:
    """Extract a BRK error code from a non-2xx response when available."""
    try:
        payload = response.json()
    except ValueError:
        return None

    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("code")

    return None


def _brk_fetch_csv(metrics, index="dateindex", start=0, timeout=120, verbose=False):
    """
    Fetch series from BRK bulk API as CSV.

    Parameters:
    metrics (list): List of series names to fetch.
    index (str): Index type for the API request.
    start (int | str): Starting range bound for data retrieval.
    timeout (int): Request timeout in seconds.
    verbose (bool): If True, print debug information.

    Returns:
    tuple: (header, data_rows, raw_text) - CSV header, data rows, and raw response text.
    """
    if verbose:
        print(f"[BRK] fetching {len(metrics)} metrics: {metrics}")

    r = requests.get(
        BRK_BULK_URL,
        params={
            "series": ",".join(metrics),
            "index": index,
            "start": start,
            "format": "csv",
        },
        timeout=timeout,
    )

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


def _brk_fetch_csv_resilient(metrics, index="dateindex", start=0, timeout=120, verbose=False):
    """
    Fetch a BRK bulk CSV request, recursively splitting oversized or invalid chunks.

    Returns:
    list[tuple]: One or more (header, rows, raw_text) responses.
    """
    try:
        return [_brk_fetch_csv(metrics, index=index, start=start, timeout=timeout, verbose=verbose)]
    except requests.HTTPError as exc:
        response = exc.response
        code = _brk_error_code(response) if response is not None else None
        non_ts = [metric for metric in metrics if metric != "timestamp"]

        if code in {"weight_exceeded", "series_not_found", "metric_not_found"} and len(non_ts) > 1:
            midpoint = len(non_ts) // 2
            left = ["timestamp"] + non_ts[:midpoint]
            right = ["timestamp"] + non_ts[midpoint:]

            if verbose:
                print(f"[BRK] splitting chunk ({code}): {non_ts}")

            return (
                _brk_fetch_csv_resilient(left, index=index, start=start, timeout=timeout, verbose=verbose)
                + _brk_fetch_csv_resilient(right, index=index, start=start, timeout=timeout, verbose=verbose)
            )

        if code in {"series_not_found", "metric_not_found"} and len(non_ts) == 1:
            if verbose:
                print(f"[BRK] skipping missing series: {non_ts[0]}")
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

    for chunk in chunks:
        responses = _brk_fetch_csv_resilient(
            chunk, index=index, start=query_start, verbose=verbose
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

    # Keep downstream code stable even if BRK temporarily removes or renames a series.
    for metric in non_ts:
        if metric not in df.columns:
            df[metric] = np.nan

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
            bitcoin_dominance = pd.DataFrame(
                {
                    "bitcoin_dominance": [dominance_value, dominance_value],
                    "time": [latest_data_date - pd.Timedelta(days=1), latest_data_date],
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

    processed_datasets = []
    for name, dataset in datasets:
        if not dataset.empty and "time" in dataset.columns:
            dataset["time"] = pd.to_datetime(dataset["time"]).dt.tz_localize(None)
            dataset.set_index("time", inplace=True)
            processed_datasets.append(dataset)

    if not processed_datasets:
        return pd.DataFrame()

    # Start with coindata
    data = processed_datasets[0]  # coindata
    for i, dataset in enumerate(processed_datasets[1:], 1):
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
    data["RevAllTimeUSD"] = data["coinbase_sum_24h_usd"].fillna(0).cumsum()
    data["NVTAdj"] = data["market_cap"] / data["transfer_volume_sum_24h_usd"]
    data["NVTAdj90"] = data["market_cap"] / data["transfer_volume_sum_24h_usd"].rolling(90).mean()

    # Calculate the number of satoshis per dollar
    data["sat_per_dollar"] = 1 / (data["price_close"] / SATS_PER_BTC)

    # Calculate the Market Value to Realized Value (MVRV) ratio
    data["mvrv_ratio"] = data["market_cap"] / data["realized_cap"]
    data["CapMVRVCur"] = data["mvrv_ratio"]

    # Calculate the realized price (the value at which each coin was last moved).
    calculated_realized_price = data["realized_cap"] / data["supply"]
    if "realized_price" in data.columns:
        data["realized_price"] = data["realized_price"].fillna(calculated_realized_price)
    else:
        data["realized_price"] = calculated_realized_price

    # Calculate the Net Unrealized Profit/Loss (NUPL)
    data["nupl"] = (data["market_cap"] - data["realized_cap"]) / data["market_cap"]

    # Calculate NVT price based on adjusted NVT ratio, with a rolling median to smooth data
    data["nvt_price"] = (
        data["NVTAdj"].rolling(window=365 * 2).median() * data["transfer_volume_sum_24h_usd"]
    ) / data["supply"]

    # Calculate adjusted NVT price using a 365-day rolling median for smoothing
    data["nvt_price_adj"] = (
        data["NVTAdj90"].rolling(window=365).median() * data["transfer_volume_sum_24h_usd"]
    ) / data["supply"]

    # Calculate NVT price multiple (current price compared to NVT price)
    data["nvt_price_multiple"] = data["price_close"] / data["nvt_price"]

    # Calculate 14-day moving average of NVT price multiple for trend analysis
    data["nvt_price_multiple_ma"] = data["nvt_price_multiple"].rolling(window=14).mean()

    # Calculate price moving averages for different time windows to analyze price trends
    data["7_day_ma_price_close"] = data["price_close"].rolling(window=7).mean()
    data["50_day_ma_price_close"] = data["price_close"].rolling(window=50).mean()
    data["100_day_ma_price_close"] = data["price_close"].rolling(window=100).mean()
    data["200_day_ma_price_close"] = data["price_close"].rolling(window=200).mean()
    data["200_week_ma_price_close"] = data["price_close"].rolling(window=200 * 7).mean()

    # Calculate the price multiple relative to the 200-day moving average
    data["200_day_multiple"] = data["price_close"] / data["200_day_ma_price_close"]

    # Calculate Thermocap multiples and associated pricing metrics
    data["thermocap_multiple"] = data["market_cap"] / data["RevAllTimeUSD"]
    data["thermocap_price"] = data["RevAllTimeUSD"] / data["supply"]
    data["thermocap_price_multiple_4"] = (4 * data["RevAllTimeUSD"]) / data["supply"]
    data["thermocap_price_multiple_8"] = (8 * data["RevAllTimeUSD"]) / data["supply"]
    data["thermocap_price_multiple_16"] = (16 * data["RevAllTimeUSD"]) / data["supply"]
    data["thermocap_price_multiple_32"] = (32 * data["RevAllTimeUSD"]) / data["supply"]

    data["miner_revenue_1_Year"] = data["coinbase_sum_24h_usd"].rolling(window=365).sum()
    data["miner_revenue_4_Year"] = data["coinbase_sum_24h_usd"].rolling(window=4 * 365).sum()

    data["ss_multiple_1"] = data["market_cap"] / data["miner_revenue_1_Year"]
    data["ss_price_1"] = data["miner_revenue_1_Year"] / data["supply"]

    data["ss_multiple_4"] = data["market_cap"] / data["miner_revenue_4_Year"]
    data["ss_price_4"] = data["miner_revenue_4_Year"] / data["supply"]

    # Calculate Realized Cap multiples for different factors (2x, 3x, 5x, 7x)
    data["realizedcap_multiple_2"] = (2 * data["realized_cap"]) / data["supply"]
    data["realizedcap_multiple_3"] = (3 * data["realized_cap"]) / data["supply"]
    data["realizedcap_multiple_5"] = (5 * data["realized_cap"]) / data["supply"]
    data["realizedcap_multiple_7"] = (7 * data["realized_cap"]) / data["supply"]

    # Calculate the percentage of supply held for more than 1 year
    # BRK provides utxos_over_1y_old_supply in BTC; divide by circulating supply for percentage
    data["supply_pct_1_year_plus"] = (data["utxos_over_1y_old_supply"] / data["supply"]) * 100
    data["pct_supply_issued"] = data["supply"] / 21000000
    data["pct_fee_of_reward"] = (data["fees_sum_24h"] / data["coinbase_sum_24h"]) * 100

    # Calculate illiquid and liquid supply based on the 1+ year held supply
    data["illiquid_supply"] = (data["supply_pct_1_year_plus"] / 100) * data["supply"]
    data["liquid_supply"] = data["supply"] - data["illiquid_supply"]

    # BRK provides active_addrs_average_24h: rolling 24h average of unique active addresses.
    # This is already a daily total — no block-count multiplication needed.
    data["daily_active_addresses_sending"] = data["active_addrs_average_24h"]

    # =========================================================================
    # Reserve Risk Pipeline (calculated from raw BDD, Price, Supply)
    # =========================================================================

    # Step 1: Adjusted BDD = BDD / Circulating Supply
    data["adjusted_bdd"] = data["coindays_destroyed_sum_24h"] / data["supply"]

    # Step 2: Mean Adjusted BDD (binary indicator: above/below average)
    data["adjusted_bdd_mean"] = data["adjusted_bdd"].expanding().mean()
    data["adjusted_bdd_above_avg"] = data["adjusted_bdd"] > data["adjusted_bdd_mean"]

    # Step 3: VOCD = Price x Adjusted BDD
    data["vocd"] = data["price_close"] * data["adjusted_bdd"]

    # Step 4: MVOCD = 30-day rolling median of VOCD
    data["mvocd"] = data["vocd"].rolling(window=30).median()

    # Step 5: HODL Bank = cumulative opportunity cost (when MVOCD < Price)
    data["daily_hodl_value"] = (data["price_close"] - data["mvocd"]).clip(lower=0)
    data["hodl_bank_calc"] = data["daily_hodl_value"].cumsum()

    # Step 6: Reserve Risk = Price / HODL Bank
    data["reserve_risk_calc"] = data["price_close"] / data["hodl_bank_calc"]

    # =========================================================================
    # Average Cap and Delta Cap
    # =========================================================================

    data["cumulative_market_cap"] = data["market_cap"].cumsum()
    data["days_since_start"] = range(1, len(data) + 1)
    data["average_cap"] = data["cumulative_market_cap"] / data["days_since_start"]
    data["delta_cap"] = data["realized_cap"] - data["average_cap"]
    data["average_cap_price"] = data["average_cap"] / data["supply"]
    data["delta_cap_price"] = data["delta_cap"] / data["supply"]

    # =========================================================================
    # Bitcoin Volatility Metrics
    # =========================================================================

    daily_returns = data["price_close"].pct_change()
    data["VtyDayRet30d"] = daily_returns.rolling(30).std() * np.sqrt(365)
    data["VtyDayRet180d"] = daily_returns.rolling(180).std() * np.sqrt(365)

    # =========================================================================
    # Supply in Profit/Loss Percentages
    # =========================================================================

    data["supply_in_profit_btc"] = np.where(
        data["supply_in_profit"] > data["supply"] * 10,
        data["supply_in_profit"] / SATS_PER_BTC,
        data["supply_in_profit"],
    )
    data["supply_in_loss_btc"] = np.where(
        data["supply_in_loss"] > data["supply"] * 10,
        data["supply_in_loss"] / SATS_PER_BTC,
        data["supply_in_loss"],
    )
    data["supply_in_profit_pct"] = (data["supply_in_profit_btc"] / data["supply"]) * 100
    data["supply_in_loss_pct"] = (data["supply_in_loss_btc"] / data["supply"]) * 100

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

    # Calculate S2F using yearly supply difference to align with PlanB's original model
    new_columns["SF"] = data["supply"] / data["supply"].diff(periods=365).fillna(0)

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
        - Electricity_Cost: Capriole electricity-only production cost per BTC.
        - Bitcoin_Production_Cost: Capriole all-in production cost per BTC.
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

    data["daily_electricity_consumption_kwh"] = hash_rate_th_s * efficiency_j_gh * 24

    total_electricity_cost = (
        data["daily_electricity_consumption_kwh"] * ELECTRICITY_COST * PUE
    )

    bitcoin_electricity_price = np.where(
        data["subsidy_sum_24h"] > 0,
        total_electricity_cost / data["subsidy_sum_24h"],
        np.nan,
    )

    data["Electricity_Cost"] = bitcoin_electricity_price
    data["Bitcoin_Production_Cost"] = (
        bitcoin_electricity_price / ELEC_TO_TOTAL_COST_RATIO
    )

    btc_per_day_network_expected = (
        data["hash_rate"]
        * SECONDS_PER_DAY
        * data["block_reward"]
        / (data["difficulty"] * SHA_256_CONSTANT)
    )

    e_day_network = ELECTRICITY_COST * 24 * efficiency_j_gh * hash_rate_th_s

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


def calculate_ytd_change(data):
    """
    Calculate the Year-to-Date (YTD) percentage change for each column in the DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.

    Returns:
    pd.DataFrame: DataFrame containing the YTD percentage change for each column.
    """
    # Calculate the first value of the year for each group and transform it to align with the original DataFrame
    start_of_year = data.groupby(data.index.year).transform("first")

    # Calculate YTD change as a percentage difference from the start of the year
    ytd_change = ((data / start_of_year) - 1) * 100  # Convert to percentage
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
    # Calculate the first value of the month for each group and transform it to align with the original DataFrame
    start_of_month = data.groupby([data.index.year, data.index.month]).transform(
        "first"
    )

    # Calculate MTD change as a percentage difference from the start of the month
    mtd_change = ((data / start_of_month) - 1) * 100  # Convert to percentage
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
    # Calculate the year-over-year change using 365-day lag (assuming daily data)
    yoy_change = data.pct_change(periods=365) * 100  # Convert to percentage
    yoy_change.columns = [f"{col}_YOY_change" for col in yoy_change.columns]

    return yoy_change


def calculate_trading_week_change(data):
    """
    Calculates the weekly change in trading data by comparing each day's value to the Monday of the same week.

    Parameters:
    data (pd.DataFrame): DataFrame containing daily trading data with datetime index.

    Returns:
    pd.DataFrame: DataFrame with columns indicating the weekly change for each numeric column.
    """
    # Determine the Monday of the week for each date
    start_of_week = data.index - pd.to_timedelta(data.index.dayofweek, unit="d")

    # Get numeric columns only
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    # VECTORIZED: Get Monday values for each week using groupby + transform
    monday_values = data[numeric_cols].groupby(start_of_week).transform("first")

    # VECTORIZED: Calculate weekly change (current - monday) / monday
    trading_week_change = (data[numeric_cols] - monday_values) / monday_values

    # Rename columns with _trading_week_change suffix
    trading_week_change.columns = [
        f"{col}_trading_week_change" for col in numeric_cols
    ]

    # Forward fill the NaN values in the trading week change DataFrame
    trading_week_change.ffill(inplace=True)

    return trading_week_change


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
            (data.pct_change(periods=period) * 100).add_suffix(f"_{period}_change")
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
    # Calculate daily returns for each asset
    returns = data.pct_change()

    # Initialize a dictionary to store rolling correlations for each period
    correlations = {}
    for period in periods:
        # Calculate rolling correlation of returns
        correlations[period] = returns.rolling(window=period).corr()

    return correlations


def calculate_volatility_tradfi(prices, windows):
    """
    Calculates rolling annualized volatility for traditional financial assets.

    Parameters:
    prices (pd.DataFrame): DataFrame containing daily price data for each asset as columns.
    windows (list): List of integers specifying rolling window sizes in days.

    Returns:
    pd.DataFrame: DataFrame of annualized volatilities for each specified window.
    """
    # Calculate daily returns based on price data
    returns = prices.pct_change()

    # Initialize a DataFrame to store the volatilities for each window
    volatilities = pd.DataFrame(index=prices.index)
    for window in windows:
        # Compute rolling standard deviation of returns for the given window
        volatility = returns.rolling(window).std()
        # Annualize volatility using trading days for traditional financial assets
        annualized_volatility = volatility * np.sqrt(STOCK_TRADING_DAYS)
        # Store the annualized volatility for the current window
        volatilities[f"{window}_day_volatility"] = annualized_volatility

    return volatilities


def calculate_volatility_crypto(prices, windows):
    """
    Calculates rolling annualized volatility for cryptocurrency assets.

    Parameters:
    prices (pd.DataFrame): DataFrame containing daily price data for each cryptocurrency as columns.
    windows (list): List of integers specifying rolling window sizes in days.

    Returns:
    pd.DataFrame: DataFrame of annualized volatilities for each specified window.
    """
    # Calculate daily returns based on price data
    returns = prices.pct_change()

    # Initialize a DataFrame to store the volatilities for each window
    volatilities = pd.DataFrame(index=prices.index)
    for window in windows:
        # Compute rolling standard deviation of returns for the given window
        volatility = returns.rolling(window).std()
        # Annualize volatility using trading days for cryptocurrency assets
        annualized_volatility = volatility * np.sqrt(CRYPTO_TRADING_DAYS)
        # Store the annualized volatility for the current window
        volatilities[f"{window}_day_volatility"] = annualized_volatility

    return volatilities


def calculate_daily_expected_return(price_series, time_frame, trading_days_in_year):
    """
    Calculates the rolling expected annualized return for an asset.

    Parameters:
    price_series (pd.Series): Series containing daily price data for the asset.
    time_frame (int): Rolling window size in days.
    trading_days_in_year (int): Number of trading days in a year (e.g., 252 for stocks, 365 for crypto).

    Returns:
    pd.Series: Series of the rolling expected annualized return.
    """
    # Calculate daily returns based on price data
    daily_returns = price_series.pct_change()
    # Compute rolling average of daily returns and annualize it
    rolling_avg_return = (
        daily_returns.rolling(window=time_frame).mean() * trading_days_in_year
    )

    return rolling_avg_return


def calculate_standard_deviation_of_returns(
    price_series, time_frame, trading_days_in_year
):
    """
    Calculates the rolling standard deviation of returns, annualized.

    Parameters:
    price_series (pd.Series): Series containing daily price data for the asset.
    time_frame (int): Rolling window size in days.
    trading_days_in_year (int): Number of trading days in a year (e.g., 252 for stocks, 365 for crypto).

    Returns:
    pd.Series: Series of the rolling annualized standard deviation of returns.
    """
    # Calculate daily returns based on price data
    daily_returns = price_series.pct_change()
    # Compute rolling standard deviation of returns and annualize it
    rolling_std_dev = daily_returns.rolling(window=time_frame).std() * (
        trading_days_in_year**0.5
    )
    return rolling_std_dev


def calculate_sharpe_ratio(
    expected_return_series, std_dev_series, risk_free_rate_series
):
    """
    Calculates the Sharpe ratio based on expected returns, standard deviation of returns, and risk-free rate.

    Parameters:
    expected_return_series (pd.Series): Series of expected returns (annualized).
    std_dev_series (pd.Series): Series of standard deviations of returns (annualized).
    risk_free_rate_series (pd.Series): Series of risk-free rate values, aligned with asset data.

    Returns:
    pd.Series: Series of Sharpe ratios.
    """
    # Calculate Sharpe ratio as the excess return over risk-free rate, divided by volatility
    sharpe_ratio_series = (
        expected_return_series - risk_free_rate_series
    ) / std_dev_series
    return sharpe_ratio_series


def calculate_daily_sharpe_ratios(data):
    """
    Calculates rolling Sharpe ratios for multiple assets over various time frames.

    Parameters:
    data (pd.DataFrame): DataFrame with daily price data for assets, including a risk-free rate column.

    Returns:
    pd.DataFrame: DataFrame containing rolling Sharpe ratios for each asset and time frame.
    """
    # Define time frames in trading days for stocks and cryptocurrencies
    time_frames = {
        "1_year": {"stock": STOCK_TRADING_DAYS, "crypto": CRYPTO_TRADING_DAYS},
        "2_year": {"stock": STOCK_TRADING_DAYS * 2, "crypto": CRYPTO_TRADING_DAYS * 2},
        "3_year": {"stock": STOCK_TRADING_DAYS * 3, "crypto": CRYPTO_TRADING_DAYS * 3},
        "4_year": {"stock": STOCK_TRADING_DAYS * 4, "crypto": CRYPTO_TRADING_DAYS * 4},
    }

    # Convert the risk-free rate to decimal form (from percentage)
    risk_free_rate_series = data["^IRX_close"] / 100

    # Initialize a dictionary to store Sharpe ratios for each asset
    sharpe_ratios = {}

    for column in data.columns:
        # Skip rate columns in calculations
        if column in {"^IRX_close", "^TNX_close"}:
            continue

        # Determine if asset is TradFi or crypto based on column naming convention
        asset_type = "crypto" if column == "price_close" else "stock"
        sharpe_ratios[column] = {}

        for time_frame_label, time_frame_days in time_frames.items():
            # Calculate rolling expected return (annualized)
            expected_return_series = calculate_daily_expected_return(
                data[column], time_frame_days[asset_type], time_frame_days[asset_type]
            )
            # Calculate rolling standard deviation (annualized)
            std_dev_series = calculate_standard_deviation_of_returns(
                data[column], time_frame_days[asset_type], time_frame_days[asset_type]
            )
            # Calculate Sharpe ratio
            sharpe_ratio_series = calculate_sharpe_ratio(
                expected_return_series, std_dev_series, risk_free_rate_series
            )
            # Store Sharpe ratio for the current time frame
            sharpe_ratios[column][time_frame_label] = sharpe_ratio_series

    # Convert the Sharpe ratios dictionary to a DataFrame and return it
    sharpe_ratios_df = pd.DataFrame.from_dict(
        {
            (asset, time_frame): sharpe_ratios[asset][time_frame]
            for asset in sharpe_ratios.keys()
            for time_frame in sharpe_ratios[asset].keys()
        },
        orient="columns",
    )

    return sharpe_ratios_df


def calculate_52_week_high_low(data, current_date):
    """
    Calculates the 52-week high and low for each column in the provided data.

    Parameters:
    data (pd.DataFrame): DataFrame with a datetime index containing time series data for various assets.
    current_date (str or datetime): The reference date for calculating the 52-week high and low.

    Returns:
    dict: A dictionary containing the 52-week high and low for each column in `data`.
    """
    high_low = {}  # Initialize a dictionary to store 52-week high and low values

    # Check if current_date is a string; if so, convert it to a datetime object
    if isinstance(current_date, str):
        current_date = datetime.strptime(current_date, "%Y-%m-%d")

    # Calculate the date 52 weeks (1 year) prior to the current date
    start_date = current_date - timedelta(weeks=52)

    # Filter the data to include only the last 52 weeks, from start_date to current_date
    data = data[(data.index >= start_date) & (data.index <= current_date)]

    # Iterate over each column in the data to calculate high and low values
    for column in data.columns:
        # Calculate the maximum value in the past 52 weeks for the column
        high = data[column].max()
        # Calculate the minimum value in the past 52 weeks for the column
        low = data[column].min()
        # Store the 52-week high and low in the dictionary for this column
        high_low[column] = {"52_week_high": high, "52_week_low": low}

    # Return the dictionary containing the 52-week high and low values for each column
    return high_low


# Difficulty Adjustment Data


def check_difficulty_change(data: pd.DataFrame) -> dict:
    """
    Gets the last two difficulty adjustments from BRK data.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'difficulty' and 'difficulty_adjustment' columns.

    Returns:
    dict: Report with the last and previous difficulty changes, and the difficulty change percentage.
    """
    # Find dates where difficulty_adjustment is non-zero (adjustment days)
    adjustment_mask = (data["difficulty_adjustment"] != 0) & (data["difficulty_adjustment"].notna())
    adjustment_dates = data.index[adjustment_mask].sort_values(ascending=False)

    if len(adjustment_dates) < 2:
        raise ValueError("Not enough difficulty adjustment dates found in BRK data")

    # Get the last two adjustment dates
    last_adjustment_date = adjustment_dates[0]
    previous_adjustment_date = adjustment_dates[1]

    # Extract difficulty values and adjustment percentage
    last_difficulty = data.loc[last_adjustment_date, "difficulty"]
    previous_difficulty = data.loc[previous_adjustment_date, "difficulty"]
    difficulty_change_percentage = data.loc[last_adjustment_date, "difficulty_adjustment"]

    # Construct report dict (maintaining same structure for compatibility)
    report = {
        "last_difficulty_change": {
            "timestamp": int(last_adjustment_date.timestamp()),
            "difficulty": last_difficulty,
            "date": last_adjustment_date,
        },
        "previous_difficulty_change": {
            "timestamp": int(previous_adjustment_date.timestamp()),
            "difficulty": previous_difficulty,
            "date": previous_adjustment_date,
        },
        "difficulty_change_percentage": difficulty_change_percentage,
    }

    return report


def calculate_difficulty_period_change(difficulty_report: dict, df: pd.DataFrame) -> pd.Series:
    """
    Calculates the percentage change in specified metrics between two Bitcoin difficulty adjustment intervals.

    Parameters:
    difficulty_report (dict): Report dictionary containing dates of the last two difficulty changes.
    df (pd.DataFrame): DataFrame with time-indexed metrics to calculate percentage changes over the interval.

    Returns:
    pd.Series: Percentage change for each numeric column in `df` over the specified time period.
    """
    # Select only numeric columns in the DataFrame
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]

    # Ensure data is sorted by date to maintain correct time sequence
    df = df.sort_index()

    # Get the dates from the report (support both new date format and legacy timestamp format)
    if "date" in difficulty_report["last_difficulty_change"]:
        last_difficulty_change_time = difficulty_report["last_difficulty_change"]["date"]
        previous_difficulty_change_time = difficulty_report["previous_difficulty_change"]["date"]
    else:
        # Legacy format with Unix timestamps
        last_difficulty_change_time = pd.to_datetime(
            difficulty_report["last_difficulty_change"]["timestamp"], unit="s"
        )
        previous_difficulty_change_time = pd.to_datetime(
            difficulty_report["previous_difficulty_change"]["timestamp"], unit="s"
        )

    # Filter the DataFrame for rows within the interval of the last two difficulty adjustments
    df_filtered = df.loc[previous_difficulty_change_time:last_difficulty_change_time]

    # Calculate percentage change between the start and end of the filtered period
    percentage_changes = (
        (df_filtered.iloc[-1] - df_filtered.iloc[0]) / df_filtered.iloc[0] * 100
    ).round(2)

    return percentage_changes


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


def compute_drawdowns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Long-form drawdown series:
      - days_since_ath
      - drawdown_pct (0 at ATH, negative when below ATH)
      - Cycle (label)

    Aligns each cycle to its start ATH date (your provided start_date).
    """
    drawdown_periods = [
        ("Drawdown Cycle 1", "2011-06-08", "2013-02-28"),
        ("Drawdown Cycle 2", "2013-11-29", "2017-03-03"),
        ("Drawdown Cycle 3", "2017-12-17", "2020-12-16"),
        ("Drawdown Cycle 4", "2021-11-10", pd.to_datetime("today").strftime("%Y-%m-%d")),
    ]

    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    out = []

    for cycle_name, start_date, end_date in drawdown_periods:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

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
    cycle_periods = [
        ("Market Cycle 1", "2010-07-25", "2011-11-18"),
        ("Market Cycle 2", "2011-11-18", "2015-01-14"),
        ("Market Cycle 3", "2015-01-14", "2018-12-16"),
        ("Market Cycle 4", "2018-12-16", "2022-11-20"),
        ("Market Cycle 5", "2022-11-20", pd.to_datetime("today").strftime("%Y-%m-%d")),
    ]

    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    out = []
    for cycle_name, start_date, end_date in cycle_periods:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        period = df.loc[(df.index >= start_dt) & (df.index <= end_dt)].copy()
        if period.empty:
            continue

        low_date = period["price_close"].idxmin()
        low_px = float(period.loc[low_date, "price_close"])

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
    bitcoin_halvings = [
        ("Genesis Era", "2009-01-03", "2012-11-28"),
        ("2nd Era",     "2012-11-28", "2016-07-09"),
        ("3rd Era",     "2016-07-09", "2020-05-11"),
        ("4th Era",     "2020-05-11", "2024-04-20"),
        ("5th Era",     "2024-04-20", pd.to_datetime("today").strftime("%Y-%m-%d")),
    ]

    # Ensure datetime index
    data = data.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Ensure sorted
    data = data.sort_index()

    out = []

    for era_name, start_date, end_date in bitcoin_halvings:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        period = data.loc[(data.index >= start_dt) & (data.index <= end_dt)].copy()
        if period.empty:
            continue

        # Pick normalization price:
        # If exact halving date is missing (common), use first available price AFTER start_dt
        if start_dt in period.index:
            start_px = float(period.loc[start_dt, "price_close"])
        else:
            start_px = float(period["price_close"].iloc[0])

        period["days_since_halving"] = (period.index - start_dt).days
        period["index_value"] = period["price_close"] / start_px  # 1.0 at halving
        period["Era"] = era_name

        out.append(period[["days_since_halving", "index_value", "Era"]])

    if not out:
        return pd.DataFrame(columns=["days_since_halving", "index_value", "Era"])

    return pd.concat(out, ignore_index=True)
