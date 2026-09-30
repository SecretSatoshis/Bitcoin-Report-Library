"""Every calculated metric: on-chain models, relative value, changes, CAGRs and correlations.

Inputs are the merged daily frame from sources.get_data after the freshness checks.
Each function returns the frame with its new columns added.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from data_definitions import (
    BITCOIN_GENESIS_DATE,
    ELECTRICITY_BASE_TARIFF_USD_PER_KWH,
    ELECTRICITY_TARIFFS_USD_PER_KWH,
    HASH_RIBBON_FAST_WINDOW,
    HASH_RIBBON_SLOW_WINDOW,
    METCALFE_ADDRESS_COLUMNS,
    SATS_PER_BTC,
)
from sources import MARKET_DATA_MAX_FFILL_DAYS, _normalized_index, _source_observation_column


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

    return data


def calculate_moving_averages(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Add 30-day and 365-day moving averages for each metric in `metrics`
    (data_definitions.MOVING_AVERAGE_METRICS), as `30_day_ma_{metric}` and `365_day_ma_{metric}`.
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
            warnings.warn(f"Supply data for {metal} is missing", RuntimeWarning, stacklevel=2)
            continue

        # Determine the correct price column based on the metal type
        if metal == "Gold":
            if "GC=F_close" not in data:
                warnings.warn("Gold price data column is missing", RuntimeWarning, stacklevel=2)
                continue
            # Use the last available price, forward filling missing values
            price_usd_per_ounce = data["GC=F_close"].ffill()
        elif metal == "Silver":
            if "SI=F_close" not in data:
                warnings.warn("Silver price data column is missing", RuntimeWarning, stacklevel=2)
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
