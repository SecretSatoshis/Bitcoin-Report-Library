"""Every calculated metric: on-chain models, relative value, changes, CAGRs and correlations.

Inputs are the merged daily frame from sources.get_data after the freshness checks.
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
    METCALFE_ADDRESS_COLUMN,
    POWER_LAW_VALUATION_BANDS,
    SATS_PER_BTC,
)
from sources import MARKET_DATA_MAX_FFILL_DAYS, _normalized_index, _source_observation_column


# Genesis followed by every observed halving. Drives block subsidy and halving eras.
BITCOIN_HALVING_DATES = [
    "2009-01-03",  # Genesis — 50 BTC/block
    "2012-11-28",
    "2016-07-09",
    "2020-05-11",
    "2024-04-20",
]


# Projected days to the next halving: observed intervals have run 1,319-1,440 days, and
# 1,435 lands the next one in late March 2028, in line with block-height projections.
# Add the observed date to BITCOIN_HALVING_DATES once it happens.
HALVING_INTERVAL_DAYS = 1435


GENESIS_BLOCK_SUBSIDY = 50.0


def bitcoin_halving_dates(through=None) -> list:
    """Halving dates, genesis first, projected every HALVING_INTERVAL_DAYS past `through`."""
    dates = [pd.Timestamp(d) for d in BITCOIN_HALVING_DATES]
    if through is None:
        return dates

    through = pd.Timestamp(through)
    while dates[-1] <= through:
        dates.append(dates[-1] + pd.Timedelta(days=HALVING_INTERVAL_DAYS))
    return dates


def _bitcoin_block_subsidy_from_time(time_index) -> pd.Series:
    """Protocol block subsidy (BTC per block) on each date, from the halving schedule."""
    dates = pd.to_datetime(time_index)
    if len(dates) == 0:
        return pd.Series(dtype=float, index=time_index)

    halvings = bitcoin_halving_dates(through=dates.max())

    # Each halving at position i (i >= 1) halves the genesis subsidy i times.
    reward = np.full(len(dates), GENESIS_BLOCK_SUBSIDY, dtype=float)
    for i, halving_date in enumerate(halvings[1:], start=1):
        reward[np.asarray(dates >= halving_date)] = GENESIS_BLOCK_SUBSIDY / (2**i)

    return pd.Series(reward, index=time_index)


def calculate_nvt_price_models(data: pd.DataFrame) -> dict:
    """NVT Price models: trailing 730-day median NVT x transfer volume / supply.

    `nvt_price` uses daily volume; `nvt_price_{30,90,365}d` use the median volume over
    that window, each with a price multiple. Windows must be complete, so early rows are NaN.
    """
    volume = data["transfer_volume_sum_24h_usd"]
    supply = data["supply"]
    reference = (data["market_cap"] / volume).rolling(730).median()
    models = {
        "nvt_price": reference * volume / supply,
        **{
            f"nvt_price_{window}d": reference * volume.rolling(window).median() / supply
            for window in (30, 90, 365)
        },
    }
    models.update({
        f"nvt_price_multiple_{window}d": data["price_close"].where(data["price_close"] > 0)
        / models[f"nvt_price_{window}d"].where(models[f"nvt_price_{window}d"] > 0)
        for window in (30, 90, 365)
    })
    return models


def calculate_custom_on_chain_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Add the on-chain valuation and network metrics built from BRK series.

    Covers MVRV, NUPL, NVT, thermocap and realized-cap multiples, price moving averages,
    supply profitability, reserve risk, average/delta cap and volatility. Intermediates
    (all-time miner revenue, adjusted BDD, HODL bank) are not published.
    """
    market_cap = data["market_cap"]
    supply = data["supply"]
    realized_cap = data["realized_cap"]
    price_close = data["price_close"]
    miner_revenue_usd = data["coinbase_sum_24h_usd"]

    # Blank before the first price; assert_no_internal_onchain_gaps rules out later gaps.
    rev_all_time = miner_revenue_usd.cumsum()

    mvrv_ratio = market_cap / realized_cap
    ma_200_day = price_close.rolling(window=200).mean()

    supply_pct_1_year_plus = (data["utxos_over_1y_old_supply"] / supply) * 100
    illiquid_supply = (supply_pct_1_year_plus / 100) * supply

    # Reserve risk: adjusted BDD -> VOCD -> MVOCD -> HODL bank -> reserve risk
    adjusted_bdd = data["coindays_destroyed_sum_24h"] / supply
    vocd = price_close * adjusted_bdd
    mvocd = vocd.rolling(window=30).median()
    daily_hodl_value = (price_close - mvocd).clip(lower=0)
    hodl_bank = daily_hodl_value.cumsum()

    # Average cap divides by the network's age since genesis, not the row count: the
    # fetched history starts 2010-01-01, 363 days after genesis.
    cumulative_market_cap = market_cap.cumsum()
    days_since_start = pd.Series(
        (_normalized_index(data) - BITCOIN_GENESIS_DATE).days + 1,
        index=data.index,
    ).clip(lower=1)
    average_cap = cumulative_market_cap / days_since_start
    delta_cap = realized_cap - average_cap

    daily_returns = price_close.pct_change(fill_method=None)

    new_columns = {
        "sat_per_dollar": SATS_PER_BTC / price_close,
        "mvrv": mvrv_ratio,
        "nupl": (market_cap - realized_cap) / market_cap,
        **calculate_nvt_price_models(data),
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
        "vocd": vocd,
        "mvocd": mvocd,
        "reserve_risk_calc": price_close / hodl_bank,
        "average_cap_price": average_cap / supply,
        "delta_cap_price": delta_cap / supply,
        # Annualized volatility in percentage points (37.0 means 37%), like the
        # release's other percent columns.
        "volatility_30d": daily_returns.rolling(30).std() * np.sqrt(365) * 100,
        "volatility_180d": daily_returns.rolling(180).std() * np.sqrt(365) * 100,
        "supply_in_profit_pct": (data["supply_in_profit"] / supply) * 100,
        "supply_in_loss_pct": (data["supply_in_loss"] / supply) * 100,
    }

    # BRK supplies realized_price; the calculated value only fills its gaps.
    calculated_realized_price = realized_cap / supply
    if "realized_price" in data.columns:
        data["realized_price"] = data["realized_price"].fillna(calculated_realized_price)
    else:
        new_columns["realized_price"] = calculated_realized_price

    # One concat avoids pandas' fragmentation warning from dozens of single assignments.
    data = pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)
    data = data.rename(columns={"active_addrs_average_24h": "daily_active_addresses_sending"})

    return data


def calculate_moving_averages(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Add `30_day_ma_{metric}` and `365_day_ma_{metric}` for each metric."""
    moving_averages = {
        f"{window}_day_ma_{metric}": data[metric].rolling(window=window).mean()
        for window in (30, 365)
        for metric in metrics
    }

    data = pd.concat([data, pd.DataFrame(moving_averages)], axis=1)
    return data


def _positive_supply(data: pd.DataFrame) -> pd.Series:
    """Bitcoin supply with non-positive values as NaN, so a per-coin price is never inf."""
    return data["supply"].where(data["supply"] > 0)


def calculate_metal_market_caps(
    data: pd.DataFrame, gold_silver_supply: pd.DataFrame
) -> pd.DataFrame:
    """Add `gold_market_cap_usd` and `silver_market_cap_usd`: supply x each day's futures close."""
    new_columns = {}
    for _, row in gold_silver_supply.iterrows():
        metal = row["Metal"]
        supply_troy_ounces = row["Supply Troy Ounces"]

        if pd.isna(supply_troy_ounces):
            warnings.warn(f"Supply data for {metal} is missing", RuntimeWarning, stacklevel=2)
            continue

        if metal == "Gold":
            if "GC=F_close" not in data:
                warnings.warn("Gold price data column is missing", RuntimeWarning, stacklevel=2)
                continue
            price_usd_per_ounce = data["GC=F_close"]
        elif metal == "Silver":
            if "SI=F_close" not in data:
                warnings.warn("Silver price data column is missing", RuntimeWarning, stacklevel=2)
                continue
            price_usd_per_ounce = data["SI=F_close"]

        new_columns[f"{metal.lower()}_market_cap_usd"] = supply_troy_ounces * price_usd_per_ounce

    data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
    return data


def calculate_btc_price_to_surpass_metal_categories(
    data: pd.DataFrame, gold_supply_breakdown: pd.DataFrame
) -> pd.DataFrame:
    """Add the BTC price at which Bitcoin's market cap equals gold, each gold use and silver."""
    supply = _positive_supply(data)

    gold_market_cap = data["gold_market_cap_usd"]
    new_columns = {"gold_market_cap_btc_price": gold_market_cap / supply}

    for _, row in gold_supply_breakdown.iterrows():
        category = row["Gold Supply Breakdown"].replace(" ", "_").lower()
        percentage_of_market = row["Percentage Of Market"] / 100.0
        new_columns[f"gold_{category}_market_cap_btc_price"] = (
            gold_market_cap * percentage_of_market
        ) / supply

    new_columns["silver_market_cap_btc_price"] = data["silver_market_cap_usd"] / supply

    new_columns_df = pd.DataFrame(new_columns, index=data.index)
    data = pd.concat([data, new_columns_df], axis=1)

    return data


def calculate_btc_price_to_surpass_fiat(
    data: pd.DataFrame, fiat_money_data: pd.DataFrame
) -> pd.DataFrame:
    """Add `{country}_m0_btc_price`: the BTC price at which Bitcoin's market cap equals that M0."""
    supply = _positive_supply(data)
    fiat_prices = {}

    for _, row in fiat_money_data.iterrows():
        country = row["Country"].replace(" ", "_").lower()
        fiat_supply_usd = row["US Dollar Trillion"] * 1e12
        fiat_prices[f"{country}_m0_btc_price"] = fiat_supply_usd / supply

    data = pd.concat([data, pd.DataFrame(fiat_prices)], axis=1)
    return data


def calculate_btc_price_to_surpass_stocks(
    data: pd.DataFrame, stock_tickers: list
) -> pd.DataFrame:
    """Add `{ticker}_market_cap_btc_price`: the BTC price at which Bitcoin's market cap equals the stock's."""
    supply = _positive_supply(data)
    stock_prices = {
        f"{ticker}_market_cap_btc_price": data[f"{ticker}_market_cap"] / supply
        for ticker in stock_tickers
    }

    data = pd.concat([data, pd.DataFrame(stock_prices)], axis=1)
    return data


def calculate_network_model_metrics(data, model_end_date=None):
    """Add the Metcalfe, power-law and hash-ribbon series.

    Coefficients are fitted on positive observations through ``model_end_date`` (normally
    the report date), then evaluated across the whole frame. Metcalfe fixes the exponent
    at 2 and fits the scale; the power law fits ``price = scale * age**exponent`` in
    log-log space. Hash ribbons compare the 30- and 60-day hash-rate averages.

    Returns (frame, parameters), where parameters holds the fitted coefficients.
    """
    required = {
        "price_close",
        "market_cap",
        "supply",
        "hash_rate",
        METCALFE_ADDRESS_COLUMN,
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
    # Market cap as price x supply, the same two inputs as every per-coin model.
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
        }
    )

    # USD curves at the valuation label's band boundaries, for the charts.
    new_columns.update(calculate_power_law_price_bands(power_law_price))

    addresses = pd.to_numeric(result[METCALFE_ADDRESS_COLUMN], errors="coerce")
    metcalfe_fit = fit_mask & model_market_cap.gt(0) & supply.gt(0) & addresses.gt(0)
    if not metcalfe_fit.any():
        raise ValueError(
            f"Metcalfe model requires positive observations for {METCALFE_ADDRESS_COLUMN}"
        )
    metcalfe_scale = np.exp(
        (
            np.log(model_market_cap.loc[metcalfe_fit])
            - 2 * np.log(addresses.loc[metcalfe_fit])
        ).mean()
    )
    new_columns["metcalfe_value"] = (metcalfe_scale * addresses.pow(2)).div(
        supply.where(supply > 0)
    )
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

    # Replace columns that already exist (calculate_moving_averages makes the same
    # 30-day hash-rate average) rather than duplicate them.
    existing = [column for column in new_columns if column in result.columns]
    if existing:
        result = result.drop(columns=existing)
    parameters = {
        "power_law_exponent": float(exponent),
        "power_law_scale": float(np.exp(log_scale)),
        "metcalfe_scale": float(metcalfe_scale),
    }
    return pd.concat([result, pd.DataFrame(new_columns, index=result.index)], axis=1), parameters


def calculate_power_law_price_bands(power_law_price):
    """Power-law price x each finite valuation band boundary other than fair value (1.0)."""
    return {
        f"power_law_price_band_{round(upper * 100):03d}": power_law_price * upper
        for upper, _ in POWER_LAW_VALUATION_BANDS
        if np.isfinite(upper) and upper != 1.0
    }


def electric_price_models(data):
    """Add the electricity-cost price models, using Coin Metrics network efficiency.

    - electricity_cost_{3..7}c: power expense per BTC earned (subsidy plus fees) at each
      tariff.
    - hayes_network_price: Hayes cost-of-production price, using the protocol block
      subsidy; hayes_network_price_multiple is price over it.
    """
    SECONDS_PER_DAY = 24 * 60 * 60
    SHA_256_CONSTANT = 2**32

    hash_rate_th_s = data["hash_rate"] / 1e12

    efficiency_j_gh = data["cm_efficiency_j_gh"]
    block_reward = _bitcoin_block_subsidy_from_time(data.index)

    # H/s ÷ 1e9 gives GH/s; multiplying by J/GH gives J/s (watts), then kWh per day.
    daily_electricity_consumption_kwh = data["hash_rate"] / 1e9 * efficiency_j_gh * 24 / 1000
    miner_revenue_btc = data["subsidy_sum_24h"] + data["fees_sum_24h"]

    for tariff in ELECTRICITY_TARIFFS_USD_PER_KWH:
        cents = int(round(tariff * 100))
        data[f"electricity_cost_{cents}c"] = (
            daily_electricity_consumption_kwh * tariff
        ).div(miner_revenue_btc.where(miner_revenue_btc > 0))

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

    data["hayes_network_price"] = np.where(
        btc_per_day_network_expected > 0,
        e_day_network / btc_per_day_network_expected,
        np.nan,
    )

    data["hayes_network_price_multiple"] = np.where(
        data["hayes_network_price"] != 0,
        data["price_close"] / data["hayes_network_price"],
        np.nan,
    )

    return data


def calculate_rolling_cagr_for_all_columns(data, years):
    """Rolling `years`-year CAGR of every column, in percentage points (`{col}_{years}y_cagr`)."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")

    data = data.apply(pd.to_numeric, errors="coerce")

    # Same calendar date `years` earlier; a row shift would drift across leap days.
    start_value = data.reindex(data.index - pd.DateOffset(years=years))
    start_value.index = data.index

    start_value = start_value.replace(0, np.nan)  # CAGR from zero is undefined

    cagr = ((data / start_value) ** (1 / years) - 1) * 100
    cagr = cagr.replace([np.inf, -np.inf], np.nan)

    cagr.columns = [f"{col}_{years}y_cagr" for col in cagr.columns]

    return cagr


def _safe_pct_change(numerator, denominator):
    """Percentage change in percentage points; NaN where the denominator is 0 or missing."""
    denominator = denominator.where(denominator != 0)
    return ((numerator / denominator) - 1) * 100


def _previous_period_positive_close(data, period):
    """Each row's last positive value before its month or year began, per column.

    A missing or zero value at the boundary falls back to the column's earlier close.
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

    sorted_data = data.sort_index()
    positive = sorted_data.where(sorted_data > 0).ffill()
    previous_close = positive.reindex(lookup_dates, method="ffill")
    previous_close.index = data.index
    return previous_close


def calculate_ytd_change(data):
    """YTD change of every column from the last close before January 1 (`{col}_ytd_change`)."""
    prior_year_close = _previous_period_positive_close(data, "year")
    ytd_change = _safe_pct_change(data, prior_year_close)
    ytd_change.columns = [f"{col}_ytd_change" for col in ytd_change.columns]

    return ytd_change


def calculate_mtd_change(data):
    """MTD change of every column from the last close before the 1st (`{col}_mtd_change`)."""
    prior_month_close = _previous_period_positive_close(data, "month")
    mtd_change = _safe_pct_change(data, prior_month_close)
    mtd_change.columns = [f"{col}_mtd_change" for col in mtd_change.columns]

    return mtd_change


def calculate_yoy_change(data):
    """Change of every column from the same calendar date a year earlier (`{col}_yoy_change`)."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")

    # A 365-row shift would drift across leap days; February 29 maps to February 28.
    prior_year_dates = data.index - pd.DateOffset(years=1)
    prior_year = data.reindex(prior_year_dates)
    prior_year.index = data.index
    yoy_change = _safe_pct_change(data, prior_year)
    yoy_change.columns = [f"{col}_yoy_change" for col in yoy_change.columns]

    return yoy_change


def calculate_all_changes(data: pd.DataFrame, yoy_columns: list, periods: Optional[list] = None) -> pd.DataFrame:
    """Fixed-window (default 7 and 90 day), MTD and YTD changes for every column, plus
    YoY for `yoy_columns`. Returns only the change columns, in percentage points."""
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
    """Change of every column over each row-count period (`{col}_{period}d_change`)."""
    changes = pd.concat(
        [
            _safe_pct_change(data, data.shift(period)).add_suffix(f"_{period}d_change")
            for period in periods
        ],
        axis=1,
    )

    return changes


# Fewest paired returns a window may hold and still publish a correlation.
MIN_CORRELATION_RETURNS = 3


def observed_market_values(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Return `columns` with carried-forward market values masked to NaN.

    A value is kept only where its observation-date marker equals the row date, so a
    Friday close does not pose as a flat Saturday. Columns without a marker (on-chain
    series) are unchanged. Must run before ``forward_fill_market_data`` drops the markers.
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


def _paired_return_correlation(first, second, as_of, period):
    """Pearson return correlation, measured between the two series' shared observations.

    A Friday-to-Monday equity return is paired with BTC's Friday-to-Monday return. NaN
    unless both series cover the whole window and traded recently.
    """
    pair = pd.concat([first, second], axis=1).loc[:as_of].dropna()
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
    if len(returns) < MIN_CORRELATION_RETURNS or returns.nunique().lt(2).any():
        return np.nan
    return returns.iloc[:, 0].corr(returns.iloc[:, 1])


def create_correlation_matrix_data(report_date, columns, correlations_data, period=90):
    """Symmetric return matrix over a calendar-day window ending on the report date.

    Input must contain real observations only (``observed_market_values``). Each pair
    uses its shared observation dates, including a prior close at the window boundary.
    Missing, stale, short or constant histories produce NaN, including on the diagonal.
    """
    if period <= 0 or len(columns) != len(set(columns)):
        raise ValueError("Correlation window must be positive and columns unique")
    as_of = pd.to_datetime(report_date).normalize()
    prices = correlations_data.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    ).sort_index()
    matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
    for i, left in enumerate(columns):
        for right in columns[i:]:
            value = _paired_return_correlation(prices[left], prices[right], as_of, period)
            matrix.loc[left, right] = matrix.loc[right, left] = value
    return matrix

