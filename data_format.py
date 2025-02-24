import requests
import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
from datetime import datetime, timedelta
from io import StringIO
import calendar
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.ticker as mticker
import mplfinance as mpf
import time
import datapane as dp
from pandas.tseries.offsets import MonthEnd


def get_coinmetrics_onchain(endpoint: str) -> pd.DataFrame:
    """
    Fetches on-chain data from CoinMetrics in CSV format.

    Parameters:
    endpoint (str): The endpoint to specify the CSV file from CoinMetrics repository.

    Returns:
    pd.DataFrame: DataFrame containing the on-chain data.
    """
    # Construct the URL to fetch the CSV data from CoinMetrics repository
    url = f"https://raw.githubusercontent.com/coinmetrics/data/master/csv/{endpoint}"
    # Send a GET request to the URL
    response = requests.get(url)
    # Read the response text as CSV into a DataFrame
    data = pd.read_csv(StringIO(response.text), low_memory=False)
    # Convert 'time' column to datetime
    data["time"] = pd.to_datetime(data["time"])
    return data


def get_fear_and_greed_index() -> pd.DataFrame:
    """
    Fetches the Fear and Greed Index data from the Alternative.me API.

    Returns:
    pd.DataFrame: DataFrame containing the Fear and Greed Index data.
    """
    # URL to fetch the Fear and Greed Index data
    url = "https://api.alternative.me/fng/?limit=0"

    try:
        # Attempt to send a GET request to the URL
        response = requests.get(
            url, timeout=10
        )  # Set a timeout to avoid indefinite waits
        response.raise_for_status()  # Raise an error for unsuccessful status codes

        # Convert the JSON response to a dictionary
        data = response.json()
        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(data["data"])
        # Convert 'timestamp' from unix to datetime format
        df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        # Select the required columns
        df = df[["value", "value_classification", "time"]]
        return df

    except (requests.exceptions.RequestException, KeyError) as e:
        # If an error occurs, return an empty DataFrame and print the error
        print(f"Failed to fetch Fear and Greed Index data. Reason: {e}")
        return pd.DataFrame(columns=["value", "value_classification", "time"])


def get_bitcoin_dominance() -> pd.DataFrame:
    """
    Fetches the current Bitcoin dominance from the CoinGecko API.

    Returns:
    pd.DataFrame: DataFrame containing Bitcoin dominance and timestamp.
    """
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url)

        data = response.json()
        bitcoin_dominance = data["data"]["market_cap_percentage"]["btc"]
        timestamp = pd.to_datetime(data["data"]["updated_at"], unit="s")

        df = pd.DataFrame(
            {"bitcoin_dominance": [bitcoin_dominance], "time": [timestamp]}
        )
        # Ensure timestamp is in the correct format
        df["time"] = pd.to_datetime(df["time"])

        return df

    except requests.RequestException as e:
        print(f"Failed to fetch Bitcoin dominance. Reason: {e}")
        return pd.DataFrame(columns=["bitcoin_dominance", "time"])


def get_kraken_ohlc(pair: str, since: int) -> pd.DataFrame:
    """
    Fetches historical OHLC data from the Kraken API.

    Parameters:
    pair (str): The Kraken asset pair (e.g., 'BTCUSD').
    since (int): Timestamp from which to fetch data.

    Returns:
    pd.DataFrame: DataFrame with OHLC data resampled to weekly intervals.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    interval = 1440  # Daily interval
    data_frames = []  # Use a list to collect data chunks for a single concat outside the loop

    while True:
        try:
            params = {"pair": pair, "interval": interval, "since": since}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get("error"):
                print(f"Error in Kraken data: {data['error']}")
                break

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

            if len(ohlc_data) < 720:
                break
            time.sleep(1)

        except requests.RequestException as e:
            print(f"Error fetching Kraken OHLC data: {e}")
            break

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
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        volume_data = response.json()["total_volumes"]
        df = pd.DataFrame(volume_data, columns=["time", "btc_trading_volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        return df

    except requests.RequestException as e:
        print(f"Failed to fetch Bitcoin trading volume. Reason: {e}")
        return pd.DataFrame(columns=["time", "btc_trading_volume"])


def get_crypto_data(ticker_list):
    """
    Fetches historical daily data for a list of cryptocurrencies from the CoinGecko API.

    Parameters:
    ticker_list (list): List of CoinGecko-compatible cryptocurrency tickers.

    Returns:
    pd.DataFrame: DataFrame containing merged close prices, volumes, and market caps.
    """
    data = pd.DataFrame()  # Initialize an empty DataFrame to store all fetched data
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
                response = requests.get(url, params=params)
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

                # Append merged data to the main DataFrame
                data = (
                    merged_data
                    if data.empty
                    else pd.merge(data, merged_data, on="time", how="outer")
                )

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

    # Set 'time' as the index and resample to fill any missing daily data
    data.set_index("time", inplace=True)
    data = data.resample("D").ffill().reset_index()

    return data


def get_price(tickers: dict, start_date: str) -> pd.DataFrame:
    data_frames = []
    end_date = datetime.today().strftime("%Y-%m-%d")
    excluded_crypto_tickers = {
        "ethereum",
        "ripple",
        "dogecoin",
        "binancecoin",
        "tether",
    }

    # Create a continuous daily index (timezone-naive)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    for category, ticker_list in tickers.items():
        for ticker in ticker_list:
            if category == "crypto" and ticker.lower() in excluded_crypto_tickers:
                continue
            for attempt in range(3):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    if hist.empty:
                        raise ValueError(f"No data returned for {ticker}.")
                    stock_data = hist[["Close"]].rename(
                        columns={"Close": f"{ticker}_close"}
                    )
                    # Strip timezone from index before reindexing
                    stock_data.index = pd.to_datetime(stock_data.index).tz_localize(
                        None
                    )
                    # Reindex to daily and forward-fill
                    stock_data = stock_data.reindex(date_range, method="ffill").fillna(
                        method="ffill"
                    )
                    data_frames.append(stock_data)
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"Retrying {ticker}: {e}")
                        time.sleep(2)
                    else:
                        print(f"Failed {ticker} in {category}: {e}")

    if data_frames:
        data = pd.concat(data_frames, axis=1)
        data.reset_index(inplace=True)
        data.rename(columns={"index": "time"}, inplace=True)
        data["time"] = pd.to_datetime(data["time"]).dt.tz_localize(
            None
        )  # Ensure timezone-naive
    else:
        data = pd.DataFrame(columns=["time"])
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
            hist = stock.history(start=start_date, end=end_date)
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


def get_miner_data(google_sheet_url: str) -> pd.DataFrame:
    """
    Fetches miner data from a Google Sheets URL and returns it as a pandas DataFrame.

    Parameters:
    google_sheet_url (str): The Google Sheets URL to extract data from.

    Returns:
    pd.DataFrame: DataFrame containing the miner data.
    """
    # Construct the CSV export URL from the Google Sheets URL
    csv_export_url = google_sheet_url.replace("/edit?usp=sharing", "/export?format=csv")

    # Load the data into a pandas DataFrame
    df = pd.read_csv(csv_export_url)
    # Ensure 'time' is in datetime format
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    # Drop rows with invalid datetime values
    df.dropna(subset=["time"], inplace=True)
    return df


def get_data(tickers, start_date):
    """
    Fetch and consolidate multiple financial and on-chain datasets for analysis.

    Parameters:
    tickers (dict): Dictionary of stock or cryptocurrency tickers to fetch data for.
    start_date (str): The start date from which to fetch data.

    Returns:
    pd.DataFrame: A consolidated DataFrame containing all relevant datasets, merged on coindata.
    """
    # Fetch data
    coindata = get_coinmetrics_onchain("btc.csv")
    prices = get_price(tickers, start_date)
    marketcaps = get_marketcap(tickers, start_date)
    fear_greed_index = get_fear_and_greed_index()
    miner_data = get_miner_data(
        "https://docs.google.com/spreadsheets/d/1GXaY6XE2mx5jnCu5uJFejwV95a0gYDJYHtDE0lmkGeA/edit?usp=sharing"
    )
    bitcoin_dominance = get_bitcoin_dominance()
    bitcoin_dominance["time"] = bitcoin_dominance["time"].dt.normalize()
    btc_trade_volume_14d = get_btc_trade_volume_14d()
    crypto_data = get_crypto_data(tickers["crypto"])

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

    return data


def calculate_custom_on_chain_metrics(data):
    """
    Calculate custom on-chain metrics for cryptocurrency data.

    Parameters:
    data (pd.DataFrame): A DataFrame containing on-chain data.

    Returns:
    pd.DataFrame: DataFrame with new metrics calculated and added.
    """
    # Calculate the number of satoshis per dollar, using the PriceUSD to determine the exchange rate
    data["sat_per_dollar"] = 1 / (data["PriceUSD"] / 100000000)

    # Calculate the Market Value to Realized Value (MVRV) ratio
    data["mvrv_ratio"] = data["CapMrktCurUSD"] / data["CapRealUSD"]

    # Calculate the realized price (the value at which each coin was last moved)
    data["realised_price"] = data["CapRealUSD"] / data["SplyCur"]

    # Calculate the Net Unrealized Profit/Loss (NUPL)
    data["nupl"] = (data["CapMrktCurUSD"] - data["CapRealUSD"]) / data["CapMrktCurUSD"]

    # Calculate a price based on adjusted NVT ratio, with a rolling median to smooth data
    data["nvt_price"] = (
        data["NVTAdj"].rolling(window=365 * 2).median() * data["TxTfrValAdjUSD"]
    ) / data["SplyCur"]

    # Calculate adjusted NVT price using a 365-day rolling median for smoothing
    data["nvt_price_adj"] = (
        data["NVTAdj90"].rolling(window=365).median() * data["TxTfrValAdjUSD"]
    ) / data["SplyCur"]

    # Calculate NVT price multiple (current price compared to NVT price)
    data["nvt_price_multiple"] = data["PriceUSD"] / data["nvt_price"]

    # Calculate 14-day moving average of NVT price multiple for trend analysis
    data["nvt_price_multiple_ma"] = data["nvt_price_multiple"].rolling(window=14).mean()

    # Calculate price moving averages for different time windows to analyze price trends
    data["7_day_ma_priceUSD"] = data["PriceUSD"].rolling(window=7).mean()  # 7-day MA
    data["50_day_ma_priceUSD"] = data["PriceUSD"].rolling(window=50).mean()  # 50-day MA
    data["100_day_ma_priceUSD"] = (
        data["PriceUSD"].rolling(window=100).mean()
    )  # 100-day MA
    data["200_day_ma_priceUSD"] = (
        data["PriceUSD"].rolling(window=200).mean()
    )  # 200-day MA
    data["200_week_ma_priceUSD"] = (
        data["PriceUSD"].rolling(window=200 * 7).mean()
    )  # 200-week MA

    # Calculate the price multiple relative to the 200-day moving average
    data["200_day_multiple"] = data["PriceUSD"] / data["200_day_ma_priceUSD"]

    # Calculate Thermocap multiples and associated pricing metrics
    data["thermocap_multiple"] = data["CapMrktCurUSD"] / data["RevAllTimeUSD"]
    data["thermocap_price"] = data["RevAllTimeUSD"] / data["SplyCur"]
    data["thermocap_price_multiple_4"] = (4 * data["RevAllTimeUSD"]) / data["SplyCur"]
    data["thermocap_price_multiple_8"] = (8 * data["RevAllTimeUSD"]) / data["SplyCur"]
    data["thermocap_price_multiple_16"] = (16 * data["RevAllTimeUSD"]) / data["SplyCur"]
    data["thermocap_price_multiple_32"] = (32 * data["RevAllTimeUSD"]) / data["SplyCur"]

    # Calculate Realized Cap multiples for different factors (3x, 5x, 7x)
    data["realizedcap_multiple_3"] = (3 * data["CapRealUSD"]) / data["SplyCur"]
    data["realizedcap_multiple_5"] = (5 * data["CapRealUSD"]) / data["SplyCur"]
    data["realizedcap_multiple_7"] = (7 * data["CapRealUSD"]) / data["SplyCur"]

    # Calculate the percentage of supply held for more than 1 year
    data["supply_pct_1_year_plus"] = 100 - data["SplyActPct1yr"]
    data["pct_supply_issued"] = data["SplyCur"] / 21000000
    data["pct_fee_of_reward"] = (data["FeeTotUSD"] / (data["RevUSD"])) * 100

    # Calculate illiquid and liquid supply based on the 1+ year held supply
    data["illiquid_supply"] = (data["supply_pct_1_year_plus"] / 100) * data["SplyCur"]
    data["liquid_supply"] = data["SplyCur"] - data["illiquid_supply"]

    return data


def calculate_moving_averages(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Calculate moving averages for each specified metric and add them to the DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing the metrics to calculate moving averages for.
    metrics (list): List of metric names to calculate moving averages for.

    Returns:
    pd.DataFrame: DataFrame with added columns for moving averages of the specified metrics.
    """
    for metric in metrics:
        # Calculate 7-day, 30-day, and 365-day moving averages for each metric
        data[f"7_day_ma_{metric}"] = data[metric].rolling(window=7).mean()
        data[f"30_day_ma_{metric}"] = data[metric].rolling(window=30).mean()
        data[f"365_day_ma_{metric}"] = data[metric].rolling(window=365).mean()
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
    data (pd.DataFrame): DataFrame containing existing financial data.
    gold_supply_breakdown (pd.DataFrame): DataFrame containing breakdown percentages for gold supply.

    Returns:
    pd.DataFrame: DataFrame with added columns for BTC prices needed to surpass metal categories.
    """
    # Ensure 'SplyCur' is forward filled to avoid NaN values
    data["SplyCur"].ffill(inplace=True)

    # Early return if 'SplyCur' for the latest row is zero or NaN to avoid division by zero
    if data["SplyCur"].iloc[-1] == 0 or pd.isna(data["SplyCur"].iloc[-1]):
        print(
            "Warning: 'SplyCur' is zero or NaN for the latest row. Skipping calculations."
        )
        return data

    new_columns = {}  # Use a dictionary to store new columns

    # Calculating BTC prices required to match or surpass gold market cap
    gold_marketcap_billion_usd = data["gold_marketcap_billion_usd"].iloc[-1]
    new_columns["gold_marketcap_btc_price"] = (
        gold_marketcap_billion_usd / data["SplyCur"]
    )

    # Iterating through gold supply breakdown to calculate BTC prices for specific categories
    for _, row in gold_supply_breakdown.iterrows():
        category = row["Gold Supply Breakdown"].replace(" ", "_").lower()
        percentage_of_market = row["Percentage Of Market"] / 100.0
        new_columns[f"gold_{category}_marketcap_btc_price"] = (
            gold_marketcap_billion_usd * percentage_of_market
        ) / data["SplyCur"]

    # Silver market cap calculations
    silver_marketcap_billion_usd = data["silver_marketcap_billion_usd"].iloc[-1]
    new_columns["silver_marketcap_btc_price"] = (
        silver_marketcap_billion_usd / data["SplyCur"]
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
    data (pd.DataFrame): DataFrame containing existing financial data.
    fiat_money_data (pd.DataFrame): DataFrame containing fiat supply data for different countries.

    Returns:
    pd.DataFrame: DataFrame with added columns for BTC prices needed to surpass fiat supplies.
    """
    for _, row in fiat_money_data.iterrows():
        country = row["Country"]
        fiat_supply_usd_trillion = row["US Dollar Trillion"]

        # Convert the fiat supply from trillions to units
        fiat_supply_usd = fiat_supply_usd_trillion * 1e12

        # Compute the price of Bitcoin needed to surpass this country's M0 money supply
        metric_name_price = country.replace(" ", "_") + "_btc_price"
        metric_name_cap = country.replace(" ", "_") + "_cap"
        data[metric_name_price] = fiat_supply_usd / data["SplyCur"]
        data[metric_name_cap] = fiat_supply_usd
    return data


def calculate_btc_price_for_stock_mkt_caps(
    data: pd.DataFrame, stock_tickers: list
) -> pd.DataFrame:
    """
    Calculate the BTC price needed to surpass market caps of different stocks.

    Parameters:
    data (pd.DataFrame): DataFrame containing existing financial data.
    stock_tickers (list): List of stock tickers to calculate market cap-based BTC prices for.

    Returns:
    pd.DataFrame: DataFrame with added columns for BTC prices needed to surpass stock market caps.
    """
    new_columns = {}
    for ticker in stock_tickers:
        # Calculate the BTC price needed to match each stock's market cap
        new_columns[ticker + "_mc_btc_price"] = (
            data[ticker + "_MarketCap"] / data["SplyCur"]
        )
    # Concatenate the new columns with the original DataFrame
    data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
    return data


def calculate_stock_to_flow_metrics(data):
    """
    Calculate stock-to-flow metrics for the given data using the PlanB model curve.

    Parameters:
    data (pd.DataFrame): DataFrame containing the current supply (SplyCur) and price (PriceUSD) data.

    Returns:
    pd.DataFrame: DataFrame with new stock-to-flow metrics added.
    """
    # Initialize a dictionary to hold new columns
    new_columns = {}

    # Use PlanB's Stock-to-Flow model directly
    # PlanB model parameters: intercept and power coefficient are pre-determined
    intercept = 14.6
    power = 3.3

    # Calculate S2F using yearly supply difference to align with PlanB's original model
    new_columns["SF"] = data["SplyCur"] / data["SplyCur"].diff(periods=365).fillna(0)

    # Applying the PlanB linear regression formula
    new_columns["SF_Predicted_Market_Value"] = (
        np.exp(intercept) * new_columns["SF"] ** power
    )

    # Calculating the predicted market price using supply
    new_columns["SF_Predicted_Price"] = (
        new_columns["SF_Predicted_Market_Value"] / data["SplyCur"]
    )

    # Apply a 365-day moving average to the predicted S2F price to smooth the curve
    new_columns["SF_Predicted_Price_MA365"] = (
        new_columns["SF_Predicted_Price"].rolling(window=365).mean()
    )

    # Calculating the S/F multiple using the actual price and the predicted price
    new_columns["SF_Multiple"] = data["PriceUSD"] / new_columns["SF_Predicted_Price"]

    # Concatenate all new columns to the DataFrame at once
    data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)

    return data


def calculate_hayes_production_cost(
    electricity_cost, miner_efficiency_j_gh, network_hashrate_th_s
):
    """
    Calculate the daily electricity consumption cost for Bitcoin mining using the Hayes model.

    Parameters:
    electricity_cost (float): Cost of electricity per kWh.
    miner_efficiency_j_gh (float): Miner efficiency in Joules per GH.
    network_hashrate_th_s (float): Network hashrate in TH/s.

    Returns:
    float: Daily electricity consumption cost in USD.
    """
    # Calculate the daily energy consumption cost based on the electricity cost, miner efficiency, and network hashrate.
    e_day = electricity_cost * 24 * miner_efficiency_j_gh * network_hashrate_th_s
    return e_day


def calculate_hayes_network_price_per_btc(
    electricity_cost_per_kwh,
    miner_efficiency_j_gh,
    total_network_hashrate_hs,
    block_reward,
    mining_difficulty,
):
    """
    Calculate the network price per BTC using the Hayes model, based on electricity cost and network parameters.

    Parameters:
    electricity_cost_per_kwh (float): Cost of electricity per kWh.
    miner_efficiency_j_gh (float): Miner efficiency in Joules per GH.
    total_network_hashrate_hs (float): Total network hashrate in H/s.
    block_reward (float): Reward for mining a block in BTC.
    mining_difficulty (float): Current mining difficulty.

    Returns:
    float: Network price per BTC in USD.
    """
    # Define constants for conversion and time calculations.
    SECONDS_PER_HOUR = 3600
    HOURS_PER_DAY = 24
    SHA_256_CONSTANT = 2**32  # Constant for SHA-256 hashing calculations.

    # Calculate the total number of hashes performed by the network per day.
    THETA = HOURS_PER_DAY * SHA_256_CONSTANT / SECONDS_PER_HOUR

    # Convert network hashrate from H/s to TH/s for consistent units.
    total_network_hashrate_th_s = total_network_hashrate_hs / 1e12

    # Calculate the expected number of BTC mined per day across the entire network.
    btc_per_day_network_expected = (
        THETA * (block_reward * total_network_hashrate_th_s) / mining_difficulty
    )

    # Calculate the daily electricity cost for the entire network.
    e_day_network = calculate_hayes_production_cost(
        electricity_cost_per_kwh, miner_efficiency_j_gh, total_network_hashrate_th_s
    )

    # Calculate the network price per BTC by dividing the energy cost by the expected BTC mined per day.
    price_per_btc_network_objective = (
        e_day_network / btc_per_day_network_expected
        if btc_per_day_network_expected != 0
        else None
    )
    return price_per_btc_network_objective


def calculate_energy_value(
    network_hashrate_hs, miner_efficiency_j_gh, annual_growth_rate_percent
):
    """
    Calculate the energy value of the network, representing the value of energy input into mining operations.

    Parameters:
    network_hashrate_hs (float): Network hashrate in H/s.
    miner_efficiency_j_gh (float): Miner efficiency in Joules per GH.
    annual_growth_rate_percent (float): Annual growth rate percentage of Bitcoin supply.

    Returns:
    float: Energy value in BTC.
    """
    # Constants for the energy value calculation.
    FIAT_FACTOR = 2.0e-15  # Conversion factor from energy input to USD.
    SECONDS_PER_YEAR = (
        365.25 * 24 * 60 * 60
    )  # Total seconds in a year, accounting for leap years.

    # Convert network hashrate from H/s to TH/s for consistent units.
    network_hashrate_th_s = network_hashrate_hs

    # Calculate the supply growth rate per second.
    supply_growth_rate_s = annual_growth_rate_percent / 100 / SECONDS_PER_YEAR

    # Convert miner efficiency from J/GH to W/TH for calculation consistency.
    miner_efficiency_w_th = miner_efficiency_j_gh * 1000

    # Calculate the total energy input in Watts.
    energy_input_watts = network_hashrate_th_s * miner_efficiency_w_th

    # Calculate the energy value in BTC based on energy input and growth rate.
    energy_value_btc = (energy_input_watts / supply_growth_rate_s) * FIAT_FACTOR
    return energy_value_btc


def calculate_daily_electricity_consumption_kwh_from_hashrate(
    network_hashrate_th_s, efficiency_j_gh
):
    """
    Calculate the daily electricity consumption in kWh for mining operations based on hashrate and miner efficiency.

    Parameters:
    network_hashrate_th_s (float): Network hashrate in TH/s.
    efficiency_j_gh (float): Miner efficiency in Joules per GH.

    Returns:
    float: Daily electricity consumption in kWh.
    """
    # Convert miner efficiency from J/GH to J/TH (1 TH = 1000 GH).
    efficiency_j_th = efficiency_j_gh * 1000

    # Calculate total energy consumption in Joules for one day (24 hours).
    total_energy_consumption_joules = (
        network_hashrate_th_s * efficiency_j_th * 3600 * 24
    )

    # Convert energy consumption from Joules to kWh (1 kWh = 3.6 million Joules).
    daily_electricity_consumption_kwh = total_energy_consumption_joules / (1000 * 3600)

    return daily_electricity_consumption_kwh


def calculate_bitcoin_production_cost(
    daily_electricity_consumption_kwh,
    electricity_cost_per_kwh,
    PUE,
    coinbase_issuance,
    elec_to_total_cost_ratio,
):
    """
    Calculate the total production cost of Bitcoin based on electricity consumption and other factors.

    Parameters:
    daily_electricity_consumption_kwh (float): Daily electricity consumption in kWh.
    electricity_cost_per_kwh (float): Cost of electricity per kWh.
    PUE (float): Power Usage Effectiveness, accounting for energy overhead.
    coinbase_issuance (float): Daily issuance of Bitcoin from mining (in BTC).
    elec_to_total_cost_ratio (float): Ratio of electricity cost to the total production cost.

    Returns:
    float: Total production cost of Bitcoin in USD.
    """
    # Calculate the total electricity cost including overhead, represented by PUE.
    total_electricity_cost = (
        daily_electricity_consumption_kwh * electricity_cost_per_kwh * PUE
    )

    # Calculate the cost per Bitcoin mined based on the coinbase issuance.
    bitcoin_electricity_price = total_electricity_cost / coinbase_issuance

    # Adjust the production cost by considering the electricity to total cost ratio.
    return bitcoin_electricity_price / elec_to_total_cost_ratio


def electric_price_models(data):
    """
    Calculate various electricity-based Bitcoin production cost models for each row in the provided dataset.

    Parameters:
    data (pd.DataFrame): DataFrame containing relevant mining and network data.

    Returns:
    pd.DataFrame: Updated DataFrame with additional metrics related to electricity cost models.
    """
    # Constants for energy value calculation.
    FIAT_FACTOR = 2.0e-15  # Conversion factor from energy to USD.
    SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60  # Total seconds in a year.
    ELECTRICITY_COST = 0.05  # Cost of electricity per kWh.
    PUE = 1.1  # Power Usage Effectiveness factor.
    elec_to_total_cost_ratio = (
        0.6  # Ratio of electricity cost to total production cost.
    )

    # Iterate over each row in the DataFrame to calculate metrics.
    for index, row in data.iterrows():
        # Calculate daily electricity consumption in kWh based on network hashrate and miner efficiency.
        daily_electricity_consumption_kwh = (
            calculate_daily_electricity_consumption_kwh_from_hashrate(
                row["HashRate"], row["lagged_efficiency_j_gh"]
            )
        )

        # Calculate Bitcoin production cost for the day.
        production_cost = calculate_bitcoin_production_cost(
            daily_electricity_consumption_kwh,
            ELECTRICITY_COST,
            PUE,
            row["IssContNtv"],  # Coinbase Issuance.
            elec_to_total_cost_ratio,
        )

        # Update the DataFrame with calculated Bitcoin production cost and electricity cost.
        data.at[index, "Bitcoin_Production_Cost"] = production_cost
        data.at[index, "Electricity_Cost"] = production_cost * elec_to_total_cost_ratio

        # Calculate the network price per BTC using the Hayes model.
        network_price = calculate_hayes_network_price_per_btc(
            ELECTRICITY_COST,  # Assumed electricity cost per kWh.
            row["lagged_efficiency_j_gh"],
            row["HashRate"],
            row["block_reward"],
            row["DiffLast"],
        )
        data.at[index, "Hayes_Network_Price_Per_BTC"] = network_price

        # Calculate the energy value for the lagged efficiency.
        energy_value = calculate_energy_value(
            row["HashRate"], row["lagged_efficiency_j_gh"], row["IssContPctAnn"]
        )
        data.at[index, "Lagged_Energy_Value"] = energy_value

        # Calculate the energy value multiple as the ratio of actual price to energy value.
        if energy_value != 0:
            energy_value_multiple = row["PriceUSD"] / energy_value
        else:
            energy_value_multiple = None

        data.at[index, "Energy_Value_Multiple"] = energy_value_multiple

        # Calculate the energy value using current miner efficiency (CM Efficiency).
        energy_value = calculate_energy_value(
            row["HashRate"], row["cm_efficiency_j_gh"], row["IssContPctAnn"]
        )
        data.at[index, "CM_Energy_Value"] = energy_value
    return data


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

    # Handle missing values by filling with 0 to avoid issues during CAGR calculation
    data.fillna(0, inplace=True)

    # Calculate the start value for CAGR by shifting data backward by the number of years in days
    days_per_year = 365
    start_value = data.shift(int(years * days_per_year))

    # Calculate CAGR using the formula: ((End Value / Start Value)^(1/years)) - 1
    cagr = ((data / start_value) ** (1 / years) - 1) * 100  # Convert to percentage
    cagr.columns = [f"{col}_{years}_Year_CAGR" for col in cagr.columns]

    return cagr


def calculate_rolling_cagr_for_all_metrics(data):
    """
    Calculate rolling CAGR for all columns in the DataFrame for both 4-year and 2-year periods.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.

    Returns:
    pd.DataFrame: DataFrame containing the calculated 4-year and 2-year CAGR for each column.
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

    # Collect weekly changes
    trading_week_change_data = []

    # Calculate the trading week change for each date in the DataFrame
    for date, monday_of_week in zip(data.index, start_of_week):
        row = {}

        # Get Monday's data safely
        monday_data = data.loc[monday_of_week] if monday_of_week in data.index else None

        if monday_data is not None:  # Check if Monday's data is available
            for col in numeric_cols:
                monday_value = np.nan_to_num(monday_data.get(col, np.nan), nan=np.nan)
                current_value = np.nan_to_num(data.at[date, col], nan=np.nan)

                # Ensure values are not NaN or inf before performing calculations
                if (
                    np.isfinite(monday_value)
                    and np.isfinite(current_value)
                    and monday_value != 0
                ):
                    row[f"{col}_trading_week_change"] = (
                        current_value - monday_value
                    ) / monday_value
                else:
                    row[f"{col}_trading_week_change"] = np.nan

        trading_week_change_data.append(row)

    # Construct the DataFrame after collecting all rows
    trading_week_change = pd.DataFrame(trading_week_change_data, index=data.index)

    # Forward fill the NaN values in the trading week change DataFrame
    trading_week_change.ffill(inplace=True)

    return trading_week_change


def calculate_all_changes(data):
    """
    Calculate various time-based changes for each column in the DataFrame, including daily, weekly, monthly, quarterly, yearly, and multi-year changes.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.

    Returns:
    pd.DataFrame: DataFrame containing all calculated changes.
    """

    # Define the periods (in days) for which we want to calculate changes
    periods = [1, 7, 30, 90, 365, 2 * 365, 3 * 365, 4 * 365, 5 * 365]

    # Calculate changes for the specified periods
    changes = calculate_time_changes(data, periods)

    # Calculate YTD, MTD, YoY and trading week changes
    ytd_change = calculate_ytd_change(data)
    mtd_change = calculate_mtd_change(data)
    yoy_change = calculate_yoy_change(data)
    tw_change = calculate_trading_week_change(data)

    # Concatenate all changes into a single DataFrame to avoid fragmentation
    changes = pd.concat(
        [changes, ytd_change, mtd_change, yoy_change, tw_change], axis=1
    )

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
    # Calculate percentage changes for each specified period and concatenate the results
    changes = pd.concat(
        [
            (data.pct_change(periods=period)).add_suffix(f"_{period}_change")
            for period in periods
        ],
        axis=1,
    )

    return changes


def run_data_analysis(data, start_date):
    """
    Run a comprehensive data analysis by calculating changes, percentiles, and z-scores for each column in the DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical data.
    start_date (str): The start date from which to begin analysis.

    Returns:
    pd.DataFrame: DataFrame containing the original data along with calculated changes, percentiles, and z-scores.
    """
    # Calculate various time-based changes for the data
    changes = calculate_all_changes(data)
    # Calculate statistical metrics (percentiles and z-scores) for the data
    percentiles, z_scores = calculate_statistics(data, start_date)

    # Merge the changes, percentiles, and z-scores with the original data
    data = pd.concat([data, changes, percentiles, z_scores], axis=1)

    return data


def compute_drawdowns(data):
    """
    Compute drawdown metrics for each major Bitcoin drawdown cycle.

    Parameters:
    data (pd.DataFrame): DataFrame containing the historical price data with a DateTime index.

    Returns:
    pd.DataFrame: DataFrame containing drawdown percentages and days since all-time high for each cycle.
    """
    drawdown_periods = [
        # Define major drawdown periods with start and end dates
        ("2011-06-08", "2013-02-28"),
        ("2013-11-29", "2017-03-03"),
        ("2017-12-17", "2020-12-16"),
        ("2021-11-10", pd.to_datetime("today")),
    ]

    drawdown_data = (
        pd.DataFrame()
    )  # Initialize an empty DataFrame to store drawdown metrics

    # Loop through each drawdown period to calculate metrics
    for i, period in enumerate(drawdown_periods, 1):
        start_date, end_date = pd.to_datetime(period[0]), pd.to_datetime(period[1])
        # Filter data for the specific drawdown period
        period_data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
        # Calculate the all-time high (ATH) within the drawdown period
        period_data[f"ath_cycle_{i}"] = period_data["PriceUSD"].cummax()
        # Calculate drawdown percentage from ATH
        period_data[f"drawdown_cycle_{i}"] = (
            period_data["PriceUSD"] / period_data[f"ath_cycle_{i}"] - 1
        ) * 100
        # Calculate days since the start of the drawdown period
        period_data["index_as_date"] = pd.to_datetime(period_data.index)
        period_data[f"days_since_ath_cycle_{i}"] = (
            period_data["index_as_date"] - start_date
        ).dt.days

        # Select relevant columns for the current drawdown cycle
        selected_columns = [f"days_since_ath_cycle_{i}", f"drawdown_cycle_{i}"]
        # Append the results to drawdown_data DataFrame
        if drawdown_data.empty:
            drawdown_data = period_data[selected_columns].rename(
                columns={
                    f"days_since_ath_cycle_{i}": "days_since_ath",
                    f"drawdown_cycle_{i}": f"drawdown_cycle_{i}",
                }
            )
        else:
            drawdown_data = pd.concat(
                [
                    drawdown_data,
                    period_data[selected_columns].rename(
                        columns={
                            f"days_since_ath_cycle_{i}": "days_since_ath",
                            f"drawdown_cycle_{i}": f"drawdown_cycle_{i}",
                        }
                    ),
                ]
            )

    return drawdown_data


def compute_cycle_lows(data):
    """
    Compute metrics related to cycle lows for each Bitcoin cycle.

    Parameters:
    data (pd.DataFrame): DataFrame containing the historical price data with a DateTime index.

    Returns:
    pd.DataFrame: DataFrame containing days since cycle low and return since cycle low for each cycle.
    """
    cycle_periods = [
        # Define cycle periods with start and end dates to identify cycle lows
        ("2010-07-25", "2011-11-18"),
        ("2011-11-18", "2015-01-14"),
        ("2015-01-14", "2018-12-16"),
        ("2018-12-16", "2022-11-20"),
        ("2022-11-20", pd.to_datetime("today")),
    ]

    cycle_low_data = (
        pd.DataFrame()
    )  # Initialize an empty DataFrame to store cycle low metrics

    # Loop through each cycle period and calculate metrics
    for i, period in enumerate(cycle_periods, 1):
        start_date, end_date = pd.to_datetime(period[0]), pd.to_datetime(period[1])
        # Filter data for the specific cycle period
        period_data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
        # Calculate the lowest price in the cycle
        cycle_low_price = period_data["PriceUSD"].min()
        # Identify the date of the cycle low
        cycle_low_date = period_data["PriceUSD"].idxmin()
        # Calculate days since the cycle low
        period_data["index_as_date"] = pd.to_datetime(period_data.index)
        period_data[f"days_since_cycle_low_{i}"] = (
            period_data["index_as_date"] - cycle_low_date
        ).dt.days
        # Calculate return since the cycle low
        period_data[f"return_since_cycle_low_{i}"] = (
            period_data["PriceUSD"] / cycle_low_price - 1
        ) * 100

        # Select relevant columns for the current cycle low period
        selected_columns = [f"days_since_cycle_low_{i}", f"return_since_cycle_low_{i}"]
        # Append the results to cycle_low_data DataFrame
        if cycle_low_data.empty:
            cycle_low_data = period_data[selected_columns].rename(
                columns={
                    f"days_since_cycle_low_{i}": "days_since_cycle_low",
                    f"return_since_cycle_low_{i}": f"return_since_cycle_low_{i}",
                }
            )
        else:
            cycle_low_data = pd.concat(
                [
                    cycle_low_data,
                    period_data[selected_columns].rename(
                        columns={
                            f"days_since_cycle_low_{i}": "days_since_cycle_low",
                            f"return_since_cycle_low_{i}": f"return_since_cycle_low_{i}",
                        }
                    ),
                ]
            )

    return cycle_low_data


def compute_halving_days(data):
    """
    Compute metrics related to Bitcoin halving events.

    Parameters:
    data (pd.DataFrame): DataFrame containing the historical price data with a DateTime index.

    Returns:
    pd.DataFrame: DataFrame containing days since halving and return since halving for each halving period.
    """
    bitcoin_halvings = [
        # Define Bitcoin halving periods with start and end dates
        ("Genesis Era", "2009-01-03", "2012-11-28"),
        ("2nd Era", "2012-11-28", "2016-07-09"),
        ("3rd Era", "2016-07-09", "2020-05-11"),
        ("4th Era", "2020-05-11", "2024-04-20"),
        ("5th Era", "2024-04-20", pd.to_datetime("today").strftime("%Y-%m-%d")),
    ]

    # Initialize an empty DataFrame to store halving metrics
    halving_data = pd.DataFrame()

    # Loop through each halving period and calculate metrics
    for i, (era_name, start_date, end_date) in enumerate(bitcoin_halvings, 1):
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        # Filter data for the specific halving period
        period_data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
        # Calculate days since the halving event
        period_data["index_as_date"] = pd.to_datetime(period_data.index)
        period_data["days_since_halving"] = (
            period_data["index_as_date"] - start_date
        ).dt.days
        # Calculate return since the halving date
        period_data[f"return_since_halving_{i}"] = (
            period_data["PriceUSD"] / period_data.loc[start_date, "PriceUSD"] - 1
        ) * 100

        # Add era identifier column
        period_data["Era"] = era_name

        # Select relevant columns for the current halving period
        selected_columns = ["days_since_halving", f"return_since_halving_{i}", "Era"]
        period_data = period_data[selected_columns]

        # Append to main DataFrame
        halving_data = pd.concat([halving_data, period_data], ignore_index=True)

    return halving_data


## Custom Reports


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
        # Annualize volatility using 252 trading days for traditional financial assets
        annualized_volatility = volatility * np.sqrt(252)
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
        # Annualize volatility using 365 days for cryptocurrency assets
        annualized_volatility = volatility * np.sqrt(365)
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
    # Define time frames in trading days for stocks (252) and cryptocurrencies (365)
    time_frames = {
        "1_year": {"stock": 252, "crypto": 365},
        "2_year": {"stock": 252 * 2, "crypto": 365 * 2},
        "3_year": {"stock": 252 * 3, "crypto": 365 * 3},
        "4_year": {"stock": 252 * 4, "crypto": 365 * 4},
    }

    # Convert the risk-free rate to decimal form (from percentage)
    risk_free_rate_series = data["^IRX_close"] / 100

    # Initialize a dictionary to store Sharpe ratios for each asset
    sharpe_ratios = {}

    for column in data.columns:
        # Skip the risk-free rate column in calculations
        if column == "^TNX_close":
            continue

        # Determine if asset is TradFi or crypto based on column naming convention
        asset_type = "crypto" if column == "PriceUSD" else "stock"
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


def get_current_block(retries=3, delay=1):
    """
    Retrieves the current block height from the Blockstream API with retries in case of request failure.

    Parameters:
    retries (int): Number of retry attempts if the request fails.
    delay (int): Initial delay in seconds before making the request.

    Returns:
    int: Current block height.
    """
    for _ in range(retries):
        try:
            time.sleep(delay)  # Optional delay before request
            response = requests.get("https://blockstream.info/api/blocks/tip/height")
            response.raise_for_status()  # Check for successful response
            return response.json()  # Return the block height
        except requests.exceptions.RequestException:
            time.sleep(5)  # Wait before retrying in case of failure
    raise Exception(
        "Failed to retrieve the current block height after multiple attempts."
    )


def get_block_info(block_height):
    """
    Retrieves information for a specified block using its block height, with retries in case of server or rate-limit issues.

    Parameters:
    block_height (int): The height of the block to retrieve information for.

    Returns:
    dict: JSON data containing block information.
    """
    time.sleep(1)  # Initial delay to avoid rapid requests
    for _ in range(10):  # Attempt up to 10 retries
        response = requests.get(
            f"https://blockstream.info/api/block-height/{block_height}"
        )
        if response.status_code in [429, 502]:  # Check for rate limit or server error
            time.sleep(10)  # Wait 10 seconds before retrying
            continue
        response.raise_for_status()  # Ensure successful response
        block_hash = response.text.strip()  # Extract block hash

        # Request detailed information for the block using its hash
        response = requests.get(f"https://blockstream.info/api/block/{block_hash}")
        if response.status_code in [429, 502]:
            time.sleep(10)  # Retry if rate-limited or server error occurs
            continue
        response.raise_for_status()  # Confirm success
        return response.json()  # Return block details as JSON
    else:
        raise Exception("Too many retries")


def get_last_difficulty_change():
    """
    Identifies the most recent Bitcoin difficulty adjustment block based on a known reference block.

    Returns:
    dict: JSON data of the last difficulty adjustment block, with a modified timestamp.
    """
    # Reference block at a known difficulty adjustment point
    KNOWN_DIFFICULTY_ADJUSTMENT_BLOCK = 800352
    current_block_height = get_current_block()  # Get the latest block height

    # Calculate the number of blocks since the last known difficulty adjustment
    blocks_since_last_known = current_block_height - KNOWN_DIFFICULTY_ADJUSTMENT_BLOCK
    completed_difficulty_periods = (
        blocks_since_last_known // 2016
    )  # Calculate complete difficulty periods

    # Calculate the block height of the last difficulty adjustment
    last_difficulty_adjustment_block_height = (
        completed_difficulty_periods * 2016
    ) + KNOWN_DIFFICULTY_ADJUSTMENT_BLOCK

    # Retrieve block info for the last difficulty adjustment block
    last_difficulty_adjustment_block = get_block_info(
        last_difficulty_adjustment_block_height
    )

    # Adjust the timestamp by subtracting 10 minutes to approximate the end of the previous interval
    last_difficulty_adjustment_block["timestamp"] -= 10 * 60

    return last_difficulty_adjustment_block


def check_difficulty_change():
    """
    Compares the most recent difficulty adjustment with the previous adjustment to calculate the percentage change in difficulty.

    Returns:
    dict: Report with the last and previous difficulty changes, and the calculated difficulty change percentage.
    """
    last_difficulty_change_block = get_last_difficulty_change()
    if last_difficulty_change_block is not None:
        # Retrieve details for the latest difficulty change
        last_difficulty_change_block_height = last_difficulty_change_block["height"]
        last_difficulty_change_timestamp = last_difficulty_change_block["timestamp"]
        last_difficulty_change_difficulty = last_difficulty_change_block["difficulty"]

        # Retrieve details for the previous difficulty change (2016 blocks before)
        previous_difficulty_change_block = get_block_info(
            last_difficulty_change_block_height - 2016
        )
        previous_difficulty_change_block_height = previous_difficulty_change_block[
            "height"
        ]
        previous_difficulty_change_timestamp = previous_difficulty_change_block[
            "timestamp"
        ]
        previous_difficulty_change_difficulty = previous_difficulty_change_block[
            "difficulty"
        ]

        # Calculate the difference and percentage change in difficulty
        difficulty_change = (
            last_difficulty_change_difficulty - previous_difficulty_change_difficulty
        )
        difficulty_change_percentage = (
            difficulty_change / previous_difficulty_change_difficulty
        ) * 100

        # Construct and return the report dictionary
        report = {
            "last_difficulty_change": {
                "block_height": last_difficulty_change_block_height,
                "timestamp": last_difficulty_change_timestamp,
                "difficulty": last_difficulty_change_difficulty,
            },
            "previous_difficulty_change": {
                "block_height": previous_difficulty_change_block_height,
                "timestamp": previous_difficulty_change_timestamp,
                "difficulty": previous_difficulty_change_difficulty,
            },
            "difficulty_change_percentage": difficulty_change_percentage,
        }

        return report


def calculate_difficulty_period_change(difficulty_report, df):
    """
    Calculates the percentage change in specified metrics between two Bitcoin difficulty adjustment intervals.

    Parameters:
    difficulty_report (dict): Report dictionary containing timestamps of the last two difficulty changes.
    df (pd.DataFrame): DataFrame with time-indexed metrics to calculate percentage changes over the interval.

    Returns:
    pd.Series: Percentage change for each numeric column in `df` over the specified time period.
    """
    # Select only numeric columns in the DataFrame
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]

    # Ensure data is sorted by date to maintain correct time sequence
    df = df.sort_index()

    # Convert Unix timestamps in the report to datetime for indexing
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


def create_valuation_data(report_data, valuation_metrics, report_date):
    """
    Creates valuation data based on the report metrics, discount rate, and targets.

    Parameters:
    report_data (pd.DataFrame): DataFrame containing report data.
    valuation_metrics (dict): Dictionary of valuation metrics and their target values.
    report_date (str): The date for which the valuation data is generated.

    Returns:
    dict: Dictionary containing the calculated valuation data.
    """
    valuation_data = {}
    number_of_years = 10

    # Retrieve discount rate and future Bitcoin supply
    discount_rate = (
        report_data.loc[report_date, "^TNX_close"] / 100
    )  # Assumes percentage
    total_bitcoins_in_circulation = report_data.loc[report_date, "SplyExpFut10yr"]
    current_btc_price = report_data.loc[report_date, "PriceUSD"]

    for metric, targets in valuation_metrics.items():
        if metric != "market_cap_metrics":
            # For non-market cap metrics, calculate buy and sell targets
            current_multiplier = report_data.loc[report_date, metric]
            underlying_metric_value = current_btc_price / current_multiplier

            buy_target = targets["buy_target"][0]
            sell_target = targets["sell_target"][0]

            valuation_data[f"{metric}_buy_target"] = (
                buy_target * underlying_metric_value
            )
            valuation_data[f"{metric}_sell_target"] = (
                sell_target * underlying_metric_value
            )
        else:
            # For market cap metrics, calculate expected and present values for each scenario
            for market_cap_metric, details in targets.items():
                market_cap = report_data.loc[report_date, market_cap_metric]
                probabilities = details["probabilities"]

                for case, prob in probabilities.items():
                    future_value_per_case = market_cap / total_bitcoins_in_circulation
                    present_value_per_case = (
                        future_value_per_case
                        / ((1 + discount_rate) ** number_of_years)
                        * prob
                    )
                    valuation_data[f"{market_cap_metric}_{case}_future_value"] = (
                        future_value_per_case
                    )
                    valuation_data[f"{market_cap_metric}_{case}_present_value"] = (
                        present_value_per_case
                    )

    return valuation_data


def calculate_price_buckets(data, bucket_size):
    """
    Calculates the number of days the price spent in each bucket range and saves
    the current price data to a CSV file.

    Parameters:
    data (pd.DataFrame): DataFrame containing time-indexed price data.
    bucket_size (int): The size of each price bucket.

    Returns:
    pd.DataFrame: DataFrame containing counts of entries in each price bucket.
    """
    # Ensure the data index is sorted by time
    data = data.sort_index(ascending=True)

    # Define the bucket ranges for price intervals
    bucket_ranges = pd.interval_range(
        start=0,
        end=(data["PriceUSD"].max() // bucket_size + 1) * bucket_size,
        freq=bucket_size,
    )

    # Assign each price to a bucket
    data["PriceBucket"] = pd.cut(data["PriceUSD"], bins=bucket_ranges)

    # Calculate the number of unique index values (days) in each bucket
    bucket_days_count = data.groupby("PriceBucket").apply(lambda x: x.index.nunique())

    # Get the current price and its bucket
    current_price = data["PriceUSD"].iloc[-1]
    current_bucket = pd.cut([current_price], bins=bucket_ranges)[0]

    # Extract the count of days for the current price bucket
    current_bucket_days = bucket_days_count[current_bucket]

    # Create DataFrame for current price data with bucket and days count
    current_price_data = pd.DataFrame(
        {
            "Current Bitcoin Price (USD)": [current_price],
            f"Current {bucket_size} Price Bucket": [
                f"${int(current_bucket.left / 1000)}K-${int(current_bucket.right / 1000)}K"
            ],
            "Days in Current Bucket": [current_bucket_days],
        }
    )

    # Save current price data to a CSV file
    current_price_data.to_csv(f"csv/trading_range_data_{bucket_size}.csv", index=False)

    # Count the number of entries in each bucket (not days)
    bucket_counts = data["PriceBucket"].value_counts().sort_index()

    # Create DataFrame for bucket counts with formatted price range
    bucket_counts_df = bucket_counts.reset_index()
    bucket_counts_df.columns = ["Price Range ($)", "Count"]
    bucket_counts_df["Price Range ($)"] = bucket_counts_df["Price Range ($)"].apply(
        lambda x: f"${int(x.left / 1000)}K-${int(x.right / 1000)}K"
    )

    return bucket_counts_df


def style_bucket_counts_table(bucket_counts_df):
    # Define the style for the table: smaller font size
    table_style = [
        {
            "selector": "th",
            "props": "font-size: 12px;",  # Adjust header font size
        },
        {
            "selector": "td",
            "props": "font-size: 12px;",  # Adjust cell font size
        },
    ]

    # Apply the style to the table and hide the index
    styled_table = bucket_counts_df.style.set_table_styles(table_style).hide_index()

    return styled_table


def monthly_heatmap(data, export_csv=True):
    """
    Creates a monthly and yearly returns heatmap for Bitcoin price data.
    """
    # Filter data to start from January 2011
    data = data[data.index >= pd.to_datetime("2012-01-01")]

    # Calculate monthly returns
    monthly_returns = data["PriceUSD"].resample("M").last().pct_change()

    # Calculate YTD returns based on the first price of the year
    start_of_year = data["PriceUSD"].groupby(data.index.year).transform("first")
    ytd_returns = (data["PriceUSD"] / start_of_year) - 1

    # Aggregate YTD returns by year
    yearly_returns = ytd_returns.groupby(data.index.year).last()

    # Prepare the data for the heatmap: years as rows, months as columns
    heatmap_data = (
        monthly_returns.groupby(
            [monthly_returns.index.year, monthly_returns.index.month]
        )
        .mean()
        .unstack()
    )

    # Add the YTD returns as a separate column
    heatmap_data[13] = yearly_returns

    # Get the last date in the data to check if the current month is complete
    last_date = data.index[-1]
    current_year, current_month = last_date.year, last_date.month

    # Check if the current month is incomplete
    is_incomplete_month = last_date.day != (last_date + MonthEnd(0)).day

    if is_incomplete_month:
        # Get the first price of the current month
        start_of_month = data["PriceUSD"].loc[last_date.strftime("%Y-%m")].iloc[0]
        # Calculate the MTD return for the incomplete month
        current_month_return = (data["PriceUSD"].iloc[-1] / start_of_month) - 1
        # Add the MTD return to the heatmap for display
        if current_year in heatmap_data.index:
            heatmap_data.loc[current_year, current_month] = current_month_return

    # Create a copy of the data excluding the incomplete month for additional calculations
    heatmap_data_excluded = heatmap_data.copy()
    if is_incomplete_month and current_year in heatmap_data.index:
        heatmap_data_excluded.loc[current_year, current_month] = pd.NA

    # Add the "4-Year Average" row (last 4 years for each month)
    heatmap_data.loc["4-Year Average"] = heatmap_data_excluded.iloc[-4:].apply(
        lambda col: col[~col.isna()].mean(), axis=0
    )

    # Add the "Median" row, excluding the incomplete month
    heatmap_data.loc["Median"] = heatmap_data_excluded.apply(
        lambda col: col[~col.isna()].median(), axis=0
    )

    # Add the "Average" row, excluding the incomplete month
    heatmap_data.loc["Average"] = heatmap_data_excluded.apply(
        lambda col: col[~col.isna()].mean(), axis=0
    )

    # Rename columns to month names
    month_names = [calendar.month_abbr[i] for i in range(1, 13)] + ["Yearly"]
    heatmap_data.columns = month_names

    # Optionally export the heatmap data to CSV
    if export_csv:
        heatmap_data.to_csv("csv/monthly_heatmap_data.csv")

    # Create text values for annotations in the heatmap
    text_values = heatmap_data.applymap(
        lambda x: f"{x:.2%}" if pd.notnull(x) else ""
    ).values

    # Define a custom colorscale: shades of red (-1 to 0) to white (0) to green (0 to 3)
    custom_colorscale = [
        [0.0, "red"],  # -1 mapped to red
        [0.5, "white"],  # 0 mapped to white
        [1.0, "green"],  # 3 mapped to green
    ]

    # Create the Plotly heatmap figure
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index.astype(str),
            colorscale=custom_colorscale,
            zmin=-1,  # Minimum value for the color scale
            zmax=1,  # Maximum value for the color scale
            text=text_values,
            hoverinfo="text",
            texttemplate="%{text}",
        )
    )

    # Update the layout of the figure
    fig.update_layout(
        title="Monthly Bitcoin Price Return Heatmap",
        xaxis_nticks=13,
        yaxis_nticks=25,
        autosize=False,
        width=1200,
        height=600,
    )

    # Create a Datapane plot
    dp_chart = dp.Plot(fig)

    return dp_chart


def weekly_heatmap(data, last_n_years=5, export_csv=False):
    """
    Creates a weekly returns heatmap for Bitcoin data over the specified number of years.

    Parameters:
    data (pd.DataFrame): DataFrame containing Bitcoin price data with a DateTime index.
    last_n_years (int): The number of years to include in the heatmap.
    export_csv (bool): Whether to export the heatmap data to a CSV file.

    Returns:
    dp.Plot: A Datapane Plot object with the weekly returns heatmap.
    """
    # Filter data to include only the last `last_n_years` years
    start_date = datetime.now() - pd.DateOffset(years=last_n_years)
    data_last_n_years = data.loc[start_date:]

    # Calculate weekly returns, filling forward for partial weeks
    weekly_returns = data_last_n_years["PriceUSD"].resample("W").ffill().pct_change()

    # Exclude the current incomplete week if necessary
    current_year, current_week, _ = datetime.now().isocalendar()
    last_date = data.index[-1]
    if last_date.isocalendar().week == current_week and last_date.year == current_year:
        weekly_returns = weekly_returns.drop(
            weekly_returns[
                (weekly_returns.index.isocalendar().week == current_week)
                & (weekly_returns.index.isocalendar().year == current_year)
            ].index,
            errors="ignore",
        )

    # Prepare the heatmap data with weeks as rows and years as columns
    heatmap_data = (
        weekly_returns.groupby(
            [
                weekly_returns.index.isocalendar().week,
                weekly_returns.index.isocalendar().year,
            ]
        )
        .mean()
        .unstack()
    )

    # Add an "Average" column showing the average return for each week
    heatmap_data["Average"] = heatmap_data.mean(axis=1)

    # Convert indices and columns to strings for Plotly compatibility
    heatmap_data.columns = heatmap_data.columns.astype(str)
    heatmap_data.index = heatmap_data.index.astype(str)

    # Optionally export heatmap data to a CSV file
    if export_csv:
        heatmap_data.to_csv("csv/weekly_heatmap_data.csv")

    # Create text annotations for the heatmap, formatted as percentages
    text_values = heatmap_data.applymap(
        lambda x: f"{x:.2%}" if pd.notnull(x) else ""
    ).values

    # Create the Plotly heatmap figure
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            y=heatmap_data.index,
            x=heatmap_data.columns,
            colorscale="RdYlGn",
            zmin=-0.1,  # Minimum value for color scale
            zmax=0.1,  # Maximum value for color scale
            zmid=0,  # Midpoint for the color scale
            text=text_values,
            hoverinfo="text",
            texttemplate="%{text}",
        )
    )

    # Update the layout for readability and styling
    fig.update_layout(
        title=f"Bitcoin Weekly Returns Heatmap (Last {last_n_years} Years)",
        xaxis_title="Years",
        yaxis_title="Weeks",
        xaxis_nticks=10,
        yaxis_nticks=52,
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=10, r=10, t=50, b=50),
    )

    # Create and return a Datapane plot
    dp_chart = dp.Plot(fig)
    return dp_chart


def plot_yoy_change(data, column_name):
    """
    Plots the Year-Over-Year (YOY) percentage change of a specified column and the Bitcoin price on dual y-axes with log scaling.

    Parameters:
    data (pd.DataFrame): DataFrame containing historical data with a DateTime index.
    column_name (str): The name of the column in the data for YOY change calculations.

    Returns:
    matplotlib.figure.Figure: A Matplotlib figure with the YOY change and Bitcoin price plot.
    """
    # Filter the data from 2012 onwards
    data_since_2012 = data[data.index.year >= 2012]

    # Create the figure and first axis
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Get the most recent YOY change and Bitcoin price
    latest_yoy_change = data_since_2012[column_name].iloc[-1]
    latest_bitcoin_price = data_since_2012["PriceUSD"].iloc[-1]

    # Format the latest YOY change and Bitcoin price for the legend
    yoy_legend_label = f"YOY Change (latest: {latest_yoy_change:.0%})"
    btc_price_legend_label = f"Bitcoin Price (latest: ${latest_bitcoin_price:,.2f})"

    # Plot the YOY data on ax1
    ax1.plot(
        data_since_2012.index,
        data_since_2012[column_name],
        label=yoy_legend_label,
        color="tab:blue",
    )
    ax1.set_yscale("symlog", linthresh=1)  # Set the y-axis to symlog scale
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: "{:.0%}".format(y))
    )  # Format as percentages
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Percent Change", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", ls="--", linewidth=0.5)
    ax1.set_title("Bitcoin Year-Over-Year Change and Price on Log Scale")

    # Create a second y-axis for the bitcoin price with log scale
    ax2 = ax1.twinx()
    ax2.plot(
        data_since_2012.index,
        data_since_2012["PriceUSD"],
        label=btc_price_legend_label,
        color="tab:orange",
    )
    ax2.set_yscale("log")  # Set the y-axis to log scale

    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    dp.Plot(fig)

    return fig


def create_price_buckets_chart(bucket_counts_df):
    """
    Creates a horizontal bar chart of Bitcoin price ranges and the number of days spent in each range.

    Parameters:
    bucket_counts_df (pd.DataFrame): DataFrame containing price range data with 'Price Range ($)' and 'Count' columns.

    Returns:
    dp.Plot: A Datapane Plot object containing the price buckets bar chart.
    """
    # Exclude the 0-1K range from the plotting data
    plot_data = bucket_counts_df[
        bucket_counts_df["Price Range ($)"] != "$0K-$1K"
    ].copy()

    # Convert the 'Price Range ($)' to a sortable numeric value
    plot_data["Sort Key"] = plot_data["Price Range ($)"].apply(
        lambda x: int(x.split("-")[0][1:-1])
    )

    # Sort the DataFrame by 'Sort Key' in descending order
    plot_data = plot_data.sort_values(by="Sort Key", ascending=False)

    # Create the bar chart using Plotly
    fig = px.bar(
        plot_data,
        y="Price Range ($)",
        x="Count",  # Change 'Days Count' to 'Count'
        orientation="h",  # Makes the bars horizontal
        color="Count",  # Use 'Count' as the color scale
        color_continuous_scale="Viridis",  # Choose a color scale
        title="Number of Days Bitcoin Traded within 1K Price Ranges",
    )

    # Update figure layout
    fig.update_layout(height=500, width=800, margin=dict(l=5, r=5, t=50, b=5))

    # Create a Datapane Plot object
    dp_chart = dp.Plot(fig)

    return dp_chart


def calculate_roi_table(data, report_date, price_column="PriceUSD"):
    """
    Calculates the return on investment (ROI) for Bitcoin over various time frames from the report date.

    Parameters:
    data (pd.DataFrame): DataFrame containing price data with a DateTime index.
    report_date (str or datetime): The date for which to calculate ROI.
    price_column (str): The column name for Bitcoin price data.

    Returns:
    pd.DataFrame: DataFrame containing ROI, Start Date, and BTC Price for each time frame.
    """
    if price_column not in data.columns:
        raise ValueError(
            f"The price column '{price_column}' does not exist in the data."
        )

    if data.empty:
        raise ValueError("The input data is empty.")

    periods = {
        "1 day": 1,
        "3 day": 3,
        "7 day": 7,
        "30 day": 30,
        "90 day": 90,
        "1 Year": 365,
        "2 Year": 730,
        "4 Year": 1460,
        "5 Year": 1825,
        "10 Year": 3650,
    }

    today = pd.to_datetime(report_date).normalize()

    # Pre-compute the 'Start Date' and 'BTC Price' for each period
    start_dates = {
        period: today - pd.Timedelta(days=days) for period, days in periods.items()
    }
    btc_prices = {
        period: data.loc[start_dates[period], price_column]
        if start_dates[period] in data.index
        else None
        for period in periods
    }

    roi_data = {
        period: data[price_column].pct_change(days).iloc[-1] * 100
        for period, days in periods.items()
    }

    # Combine the ROI, Start Dates, and BTC Prices into a DataFrame
    roi_table = pd.DataFrame(
        {
            "Time Frame": periods.keys(),
            "ROI": roi_data.values(),
            "Start Date": start_dates.values(),
            "BTC Price": btc_prices.values(),
        }
    )
    roi_table.to_csv("csv/roi_table.csv")
    return roi_table.set_index("Time Frame")


def style_roi_table(roi_table):
    """
    Styles the ROI table by formatting the 'ROI' column with colors and the 'BTC Price' column as currency.

    Parameters:
    roi_table (pd.DataFrame): DataFrame containing the ROI data.

    Returns:
    pd.io.formats.style.Styler: Styled DataFrame for display.
    """

    # Function to color ROI values
    def color_roi(val):
        color = "green" if val > 0 else ("red" if val < 0 else "black")
        return "color: %s" % color

    # Function to format BTC Price as currency
    def format_currency(val):
        return "${:,.2f}".format(val)

    return (
        roi_table.style.applymap(color_roi, subset=["ROI"])
        .format(
            {
                "ROI": "{:.2f}%",  # Format ROI as percentage with 2 decimal places
                "BTC Price": format_currency,
            }
        )  # Format BTC Price as currency
        .set_properties(**{"font-size": "10pt"})
    )


def calculate_ma_table(data, price_column="PriceUSD"):
    """
    Calculates various moving averages and their respective 7-day percentage changes for Bitcoin prices.

    Parameters:
    data (pd.DataFrame): DataFrame containing price data with a DateTime index.
    price_column (str): The column name for Bitcoin price data.

    Returns:
    pd.DataFrame: DataFrame containing moving averages and 7-day percentage changes.
    """
    periods = {
        "3 Day MA": 3,
        "7 Day MA": 7,
        "30 Day MA": 30,
        "50 Day MA": 50,
        "100 Day MA": 100,
        "200 Day MA": 200,
        "50 Week MA": 50 * 7,
        "200 Week MA": 200 * 7,
        "50 Month MA": 50 * 30,
    }

    ma_data = {
        period: {
            "Price": data[price_column].rolling(window=days).mean().iloc[-1],
            "7 Day % Change": (
                (
                    data[price_column].rolling(window=days).mean().iloc[-1]
                    / data[price_column].rolling(window=days).mean().iloc[-8]
                )
                - 1
            )
            * 100,
        }
        for period, days in periods.items()
    }
    ma_data["Current Price"] = {
        "Price": data[price_column].iloc[-1],
        "7 Day % Change": (
            (data[price_column].iloc[-1] / data[price_column].iloc[-8]) - 1
        )
        * 100,
    }
    ma_table = pd.DataFrame.from_dict(ma_data, orient="index")
    ma_table.index.name = "Moving Average"
    ma_table.to_csv("csv/ma_table.csv")
    return ma_table[["Price", "7 Day % Change"]]


def style_ma_table(ma_table):
    """
    Styles the MA table by formatting the 'Price' as currency and the '7 Day % Change' as a percentage with colors.

    Parameters:
    ma_table (pd.DataFrame): DataFrame containing moving average data.

    Returns:
    pd.io.formats.style.Styler: Styled DataFrame for display.
    """

    # Function to color percentage values
    def color_percentage(val):
        color = "green" if val > 0 else ("red" if val < 0 else "black")
        return "color: %s" % color

    return (
        ma_table.style.format(
            {
                "Price": "${:,.2f}",  # Format 'Bitcoin Price' as currency with two decimal places
                "7 Day % Change": "{:.2f}%",  # Format '7 Day Avg % Change' as percentage with two decimal places
            }
        )
        .applymap(
            color_percentage, subset=["7 Day % Change"]
        )  # Color the '7 Day Avg % Change' based on value
        .set_properties(**{"font-size": "10pt"})
    )


def calculate_weekly_values(df, column_name, start_of_week, latest_date):
    """
    Calculates daily values for each weekday (Monday to Friday) and the weekly total
    based on the provided start date (Monday) and latest date.

    Parameters:
    df (pd.DataFrame): DataFrame with time-indexed data containing the specified column.
    column_name (str): The column name for which weekly values are calculated.
    start_of_week (datetime): The starting Monday of the week.
    latest_date (datetime): The latest date available in the data.

    Returns:
    tuple: A tuple containing:
        - weekly_values (list): List of daily values from Monday to Friday.
        - week_total (float): The total sum of values from Monday to the latest available date.
    """
    # Generate dates from Monday to Friday for the given week
    week_days = pd.date_range(start=start_of_week, periods=5, freq="D").date
    # Select values for each day; if missing, fill with None
    weekly_values = (
        df.loc[df.index.date.isin(week_days), column_name]
        .reindex(week_days, fill_value=None)
        .values
    )

    # Sum values for the week, capped at the latest available date
    week_total = df.loc[
        (df.index.date >= start_of_week.date()) & (df.index.date <= latest_date.date()),
        column_name,
    ].sum()
    return weekly_values, week_total


def create_metrics_table(df, metrics_template):
    """
    Generates a weekly metrics table with custom formatting based on a provided template.

    Parameters:
    df (pd.DataFrame): DataFrame with time-indexed data containing columns specified in the template.
    metrics_template (dict): Dictionary where keys are section headers and values are dictionaries
                             with metric names as keys and tuples (column_name, format_type) as values.

    Returns:
    pd.io.formats.style.Styler: A styled DataFrame ready for display with customized formatting.
    """
    # Define formatting functions
    formatters = {
        "number": lambda val: f"{val:,.0f}" if pd.notnull(val) else "",
        "number2": lambda val: f"{val:,.2f}" if pd.notnull(val) else "",
        "currency": lambda val: f"${val:,.2f}" if pd.notnull(val) else "",
        "percent": lambda val: f"{val:.2f}%" if pd.notnull(val) else "",
        "percent_change": lambda val: (
            f"<span style='color: {'green' if val > 0 else 'red'};'>{val:.2f}%</span>"
            if pd.notnull(val)
            else ""
        ),
    }

    def apply_format(column, format_type):
        """Apply the correct formatting function based on format_type."""
        return column.apply(formatters[format_type])

    # Initialize a list to hold the calculated values
    table_data = []

    # Get the most recent date in the DataFrame for current week calculations
    latest_date = df.index.max()
    start_of_week = latest_date - timedelta(days=latest_date.weekday())  # Monday

    # Helper function to append header and metric rows
    def append_row(data, row, header=False):
        """Append a row with specific formatting and track headers for styling."""
        if header:
            header_style_indices.append(len(data))  # Track header row index
        data.append(row)

    header_style_indices = []  # Store header row indices

    # Loop through each section and metric in the template
    for section, metrics in metrics_template.items():
        # Add section header row
        append_row(
            table_data,
            {
                "Metric": f"<strong>{section}</strong>",
                "7 Day Avg": "-",
                "7 Day Avg % Change": "-",
                "Monday": "-",
                "Tuesday": "-",
                "Wednesday": "-",
                "Thursday": "-",
                "Friday": "-",
            },
            header=True,
        )

        # Process each metric
        for metric_display_name, (column_name, format_type) in metrics.items():
            # Calculate the 7-day average and percentage change from the previous 7-day average
            week_avg = df[column_name].tail(7).mean()
            prev_week_avg = df[column_name].tail(14).head(7).mean()
            pct_change = (
                ((week_avg - prev_week_avg) / prev_week_avg) * 100
                if prev_week_avg != 0
                else np.nan
            )

            # Get weekly values (Mon-Fri) up to the latest available date
            weekly_values = (
                df[column_name][start_of_week:latest_date]
                .reindex(
                    pd.date_range(start=start_of_week, periods=5, freq="D").date,
                    fill_value=None,
                )
                .tolist()
            )

            # Add the metric data with formatted values
            append_row(
                table_data,
                {
                    "Metric": metric_display_name,
                    "7 Day Avg": apply_format(pd.Series([week_avg]), format_type)[0],
                    "7 Day Avg % Change": apply_format(
                        pd.Series([pct_change]), "percent_change"
                    )[0],
                    "Monday": apply_format(pd.Series([weekly_values[0]]), format_type)[
                        0
                    ],
                    "Tuesday": apply_format(pd.Series([weekly_values[1]]), format_type)[
                        0
                    ],
                    "Wednesday": apply_format(
                        pd.Series([weekly_values[2]]), format_type
                    )[0],
                    "Thursday": apply_format(
                        pd.Series([weekly_values[3]]), format_type
                    )[0],
                    "Friday": apply_format(pd.Series([weekly_values[4]]), format_type)[
                        0
                    ],
                },
            )

    # Create the final DataFrame and style it
    table_df = pd.DataFrame(table_data)

    # Apply styling to hide index and adjust text properties
    styled_table = table_df.style.hide_index().set_properties(
        **{"font-size": "10pt", "white-space": "nowrap"}
    )

    # Define specific styles for the header rows
    header_styles = [
        {
            "selector": f".row{row_index}",
            "props": [
                ("border-top", "1px solid black"),
                ("border-bottom", "1px solid black"),
            ],
        }
        for row_index in header_style_indices
    ]

    # Apply header row styles to the styled table
    styled_table = styled_table.set_table_styles(header_styles, overwrite=False)

    return styled_table


def create_ohlc_chart(ohlc_data, report_data, chart_template):
    """
    Creates an interactive OHLC (candlestick) chart with Plotly, including additional metrics and optional event markers.

    Parameters:
    - ohlc_data (pd.DataFrame): Time-indexed DataFrame with OHLC data, containing 'Open', 'High', 'Low', and 'Close' columns.
    - report_data (pd.DataFrame): DataFrame with additional metrics for overlaying on the OHLC chart (e.g., moving averages).
    - chart_template (dict): Dictionary defining chart settings, including title, labels, filename, and event markers.

    Returns:
    - fig (plotly.graph_objs.Figure): Plotly figure object containing the candlestick chart with overlays and event annotations.
    """
    # Ensure the index is timezone-naive for compatibility

    ohlc_data.index = ohlc_data.index.tz_localize(None)
    report_data.index = report_data.index.tz_localize(None)

    # Extract chart attributes from the template
    title = chart_template["title"]
    x_label = chart_template["x_label"]
    y_label = chart_template["y1_label"]
    filename = chart_template["filename"]

    # Calculate the extended x-axis range
    start_date = ohlc_data.index.min() - timedelta(days=30)  # Extend 30 days back
    end_date = ohlc_data.index.max() + timedelta(days=30)  # Extend 30 days forward

    # Filter `report_data` to align with OHLC data range
    report_data_filtered = report_data[report_data.index >= ohlc_data.index.min()]

    # Initialize the candlestick chart with OHLC data
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ohlc_data.index,
                open=ohlc_data["Open"],
                high=ohlc_data["High"],
                low=ohlc_data["Low"],
                close=ohlc_data["Close"],
                name="Weekly Price",
            )
        ]
    )

    # Overlay additional metrics on the chart
    for metric in [
        "200_week_ma_priceUSD",
        "realised_price",
        "realizedcap_multiple_3",
        "realizedcap_multiple_5",
        "thermocap_price_multiple_8",
        "thermocap_price_multiple_16",
        "thermocap_price_multiple_32",
    ]:
        if metric in report_data_filtered.columns:
            fig.add_trace(
                go.Scatter(
                    x=report_data_filtered.index,
                    y=report_data_filtered[metric],
                    mode="lines",
                    name=metric,
                )
            )

    # Update the layout, including axis settings, legend, and custom buttons for y-axis scaling
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.95),
        xaxis=dict(
            title=x_label,
            showgrid=False,
            tickformat="%B-%d-%Y",
            rangeslider_visible=False,
            range=[start_date, end_date],  # Use extended range
        ),
        yaxis=dict(title=y_label, showgrid=False, type="log", autorange=True),
        plot_bgcolor="rgba(255, 255, 255, 1)",
        hovermode="x",
        autosize=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.23, xanchor="center", x=0.5
        ),
        template="plotly_white",
        updatemenus=[
            dict(
                buttons=list(
                    [
                        dict(
                            label="Y1-axis: Linear",
                            method="relayout",
                            args=["yaxis.type", "linear"],
                        ),
                        dict(
                            label="Y1-axis: Log",
                            method="relayout",
                            args=["yaxis.type", "log"],
                        ),
                    ]
                ),
                direction="right",
                x=-0.1,
                xanchor="left",
                y=-0.25,
                yanchor="top",
            )
        ],
        width=1400,
        height=700,
        margin=dict(l=50, r=50, b=50, t=50, pad=50),
        font=dict(family="PT Sans Narrow", size=14, color="black"),
    )

    # Add event markers (vertical lines and annotations) from the template
    if "events" in chart_template:
        for event in chart_template["events"]:
            event_dates = pd.to_datetime(event["dates"])
            for date in event_dates:
                if ohlc_data.index.min() <= date <= ohlc_data.index.max():
                    fig.add_vline(
                        x=date.timestamp() * 1000,
                        line=dict(color="black", width=1, dash="dash"),
                    )
                    fig.add_annotation(
                        x=date, y=0.5, text=event["name"], showarrow=False, yref="paper"
                    )

    # Add a watermark to the chart
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        text="SecretSatoshis.com",
        showarrow=False,
        font=dict(size=50, color="rgba(128, 128, 128, 0.5)"),
        align="center",
    )
    return fig


def calculate_weekly_ohlc(ohlc_data, output_file="csv/weekly_ohlc.csv"):
    """
    Calculates the latest weekly OHLC (Open, High, Low, Close) values from daily data and saves them to a CSV file.

    Parameters:
    - ohlc_data (pd.DataFrame): DataFrame with daily OHLC data containing 'Open', 'High', 'Low', and 'Close' columns.
    - output_file (str): Filename for saving the weekly OHLC values as a CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the latest weekly OHLC values for the most recent week.
    """
    # Ensure the index is a datetime index for resampling
    ohlc_data.index = pd.to_datetime(ohlc_data.index)

    # Resample daily data to get weekly OHLC values
    weekly_ohlc = ohlc_data.resample("W").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )

    # Extract the most recent weekly OHLC values
    latest_weekly_ohlc = weekly_ohlc.iloc[-1]

    # Convert to a DataFrame for export and save to CSV
    latest_weekly_ohlc_df = pd.DataFrame(latest_weekly_ohlc).T
    latest_weekly_ohlc_df.to_csv(output_file)

    return latest_weekly_ohlc_df


def create_btc_correlation_tables(report_date, tickers, correlations_data):
    """
    Creates tables of Bitcoin correlations for specified assets at a given date.

    Parameters:
    report_date (str or pd.Timestamp): Date for correlation snapshot.
    tickers (dict): Dictionary of asset categories and ticker lists.
    correlations_data (pd.DataFrame): DataFrame with historical price data.

    Returns:
    dict: Rolling correlations for specified periods with PriceUSD.
    """
    report_date = pd.to_datetime(report_date)
    all_tickers = [ticker for ticker_list in tickers.values() for ticker in ticker_list]
    ticker_list_with_suffix = ["PriceUSD"] + [
        f"{ticker}_close" for ticker in all_tickers
    ]

    filtered_data = correlations_data[ticker_list_with_suffix].dropna(
        subset=["PriceUSD"]
    )

    if filtered_data.empty:
        empty_corr = pd.Series(
            index=[f"{ticker}_close" for ticker in all_tickers], dtype=float
        )
        return {f"priceusd_{p}_days": empty_corr for p in [7, 30, 90, 365]}

    correlations = calculate_rolling_correlations(
        filtered_data, periods=[7, 30, 90, 365]
    )
    closest_date = (
        filtered_data.index[
            filtered_data.index.get_indexer([report_date], method="nearest")[0]
        ]
        if report_date not in filtered_data.index
        else report_date
    )

    btc_correlations = {}
    for period in [7, 30, 90, 365]:
        corr_df = correlations[period]
        try:
            if report_date in corr_df.index:
                btc_correlations[f"priceusd_{period}_days"] = corr_df.loc[
                    report_date
                ].loc[["PriceUSD"]]
            else:
                btc_correlations[f"priceusd_{period}_days"] = corr_df.loc[
                    closest_date
                ].loc[["PriceUSD"]]
        except KeyError:
            btc_correlations[f"priceusd_{period}_days"] = pd.Series(
                index=[f"{ticker}_close" for ticker in all_tickers], dtype=float
            )

    return btc_correlations


def create_monthly_returns_table(selected_metrics):
    today = datetime.today().date()
    current_year = today.year
    current_month = today.month
    current_day = today.day

    # Ensure data is filtered to entries from January 1, 2014, onwards
    selected_metrics = selected_metrics[selected_metrics.index >= "2014-01-01"].copy()

    monthly_returns = {}
    report_date_returns = {}

    # Get the starting price for the current month of the current year
    current_month_data = selected_metrics[
        (selected_metrics.index.year == current_year)
        & (selected_metrics.index.month == current_month)
    ]
    if current_month_data.empty:
        return None  # No data for current month

    current_start_price = current_month_data["PriceUSD"].iloc[0]

    # Calculate monthly returns for each year
    for year in selected_metrics.index.year.unique():
        monthly_data = selected_metrics[
            (selected_metrics.index.year == year)
            & (selected_metrics.index.month == current_month)
        ]

        if not monthly_data.empty:
            start_price = monthly_data["PriceUSD"].iloc[0]
            end_price = monthly_data["PriceUSD"].iloc[-1]
            return_pct = (end_price / start_price - 1) * 100
            monthly_returns[year] = (start_price, end_price, return_pct)

            # Report Date Return Calculation
            report_date_data = monthly_data[(monthly_data.index.day == current_day)]
            if not report_date_data.empty:
                report_date_price = report_date_data["PriceUSD"].iloc[-1]
                report_date_return = (report_date_price / start_price - 1) * 100
                report_date_returns[year] = report_date_return
            else:
                report_date_returns[year] = None

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(
        monthly_returns,
        orient="index",
        columns=["Start Price ($)", "End Price ($)", "Return (%)"],
    )
    df.index.name = "Year"

    # Add report date return column
    df["Report Date Return (%)"] = pd.Series(report_date_returns)

    # Extract the current year's data
    current_year_row = df.loc[[current_year]].reset_index()

    # Calculate the historical median return
    median_return = df["Return (%)"].median()
    median_end_price = current_start_price * (1 + median_return / 100)

    # Calculate the median return at the current date (not full period)
    median_report_date_return = df["Report Date Return (%)"].median()

    # Create the projected median row
    median_row = pd.DataFrame(
        {
            "Year": ["Median Projection"],
            "Start Price ($)": [current_start_price],
            "End Price ($)": [median_end_price],
            "Return (%)": [median_return],
            "Report Date Return (%)": [median_report_date_return],
        }
    )

    # Concatenate current year and median projection rows
    df_filtered = pd.concat([current_year_row, median_row], ignore_index=True)

    return df_filtered


def create_yearly_returns_table(selected_metrics):
    today = datetime.today().date()
    current_year = today.year
    current_day_of_year = today.timetuple().tm_yday

    # Ensure data is filtered correctly
    selected_metrics = selected_metrics[selected_metrics.index >= "2014-01-01"].copy()

    yearly_returns = {}
    report_date_returns = {}

    # Get the starting price for the current year
    current_year_data = selected_metrics[selected_metrics.index.year == current_year]
    if current_year_data.empty:
        return None  # No data for current year

    current_start_price = current_year_data["PriceUSD"].iloc[0]

    # Calculate yearly returns for each year
    for year in selected_metrics.index.year.unique():
        yearly_data = selected_metrics[selected_metrics.index.year == year]

        if not yearly_data.empty:
            start_price = yearly_data["PriceUSD"].iloc[0]
            end_price = yearly_data["PriceUSD"].iloc[-1]
            return_pct = (end_price / start_price - 1) * 100
            yearly_returns[year] = (start_price, end_price, return_pct)

            # Report Date Return Calculation
            report_date_data = yearly_data[
                yearly_data.index.dayofyear == current_day_of_year
            ]
            if not report_date_data.empty:
                report_date_price = report_date_data["PriceUSD"].iloc[-1]
                report_date_return = (report_date_price / start_price - 1) * 100
                report_date_returns[year] = report_date_return
            else:
                report_date_returns[year] = None

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(
        yearly_returns,
        orient="index",
        columns=["Start Price ($)", "End Price ($)", "Return (%)"],
    )
    df.index.name = "Year"

    # Add report date return column
    df["Report Date Return (%)"] = pd.Series(report_date_returns)

    # Extract the current year's data
    current_year_row = df.loc[[current_year]].reset_index()

    # Calculate the historical median return
    median_return = df["Return (%)"].median()
    median_end_price = current_start_price * (1 + median_return / 100)

    # **Fix the Median Report Date Return (%)**
    median_report_date_return_pct = df["Report Date Return (%)"].dropna().median()

    # Create the projected median row
    median_row = pd.DataFrame(
        {
            "Year": ["Median Projection"],
            "Start Price ($)": [current_start_price],
            "End Price ($)": [median_end_price],
            "Return (%)": [median_return],
            "Report Date Return (%)": [median_report_date_return_pct],
        }
    )

    # Concatenate current year and median projection rows
    df_filtered = pd.concat([current_year_row, median_row], ignore_index=True)

    return df_filtered


def create_asset_valuation_table(report_data):
    """
    Generates a valuation table for various assets compared to Bitcoin.

    Parameters:
    - report_data (pd.DataFrame): DataFrame containing asset market cap and BTC price data.

    Returns:
    - pd.DataFrame: DataFrame summarizing asset valuations and Bitcoin price targets.
    """
    assets = [
        {"name": "Bitcoin", "data": "PriceUSD", "marketcap": "CapMrktCurUSD"},
        {
            "name": "Total Silver Market",
            "data": "silver_marketcap_btc_price",
            "marketcap": "silver_marketcap_billion_usd",
        },
        {
            "name": "UK M0",
            "data": "United_Kingdom_btc_price",
            "marketcap": "United_Kingdom_cap",
        },
        {"name": "Meta", "data": "META_mc_btc_price", "marketcap": "META_MarketCap"},
        {"name": "Amazon", "data": "AMZN_mc_btc_price", "marketcap": "AMZN_MarketCap"},
        {
            "name": "Gold Country Holdings",
            "data": "gold_official_country_holdings_marketcap_btc_price",
            "marketcap": "gold_marketcap_official_country_holdings_billion_usd",
        },
        {"name": "NVIDIA", "data": "NVDA_mc_btc_price", "marketcap": "NVDA_MarketCap"},
        {
            "name": "Gold Private Investment",
            "data": "gold_private_investment_marketcap_btc_price",
            "marketcap": "gold_marketcap_private_investment_billion_usd",
        },
        {"name": "Apple", "data": "AAPL_mc_btc_price", "marketcap": "AAPL_MarketCap"},
        {
            "name": "US M0",
            "data": "United_States_btc_price",
            "marketcap": "United_States_cap",
        },
        {
            "name": "Total Gold Market",
            "data": "gold_marketcap_btc_price",
            "marketcap": "gold_marketcap_billion_usd",
        },
    ]

    # Get the latest values (last row)
    latest_data = report_data.iloc[-1]
    bitcoin_price = latest_data.get("PriceUSD", float("nan"))

    valuation_data = []
    for asset in assets:
        marketcap_btc_price = latest_data.get(asset["data"], float("nan"))
        marketcap_value = latest_data.get(asset["marketcap"], float("nan"))

        # Avoid division by zero or invalid values
        if (
            pd.notna(bitcoin_price)
            and pd.notna(marketcap_btc_price)
            and bitcoin_price > 0
        ):
            percent_move = ((marketcap_btc_price - bitcoin_price) / bitcoin_price) * 100
        else:
            percent_move = "N/A"

        valuation_data.append(
            {
                "Asset": asset["name"],
                "Market Cap (USD)": f"${marketcap_value:,.0f}"
                if pd.notna(marketcap_value)
                else "N/A",
                "Market Cap BTC Price": f"${marketcap_btc_price:,.0f}"
                if pd.notna(marketcap_btc_price)
                else "N/A",
                "BTC % Move to Marketcap BTC Price": f"{percent_move:.0f}%"
                if percent_move != "N/A"
                else "N/A",
            }
        )

    valuation_df = pd.DataFrame(valuation_data)

    return valuation_df
