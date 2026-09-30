"""Checks that decide whether a run may publish.

Market data may bridge weekends and short holidays; on-chain data is never filled. These
functions bound the market fill, warn on stale market or miner inputs, and refuse to
publish when on-chain data is missing, gapped or stale, or when a hand-maintained
reference figure is past its review date.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from data_definitions import (
    BRK_METRICS,
    PRICE_OUTLOOK_YEAR,
    REFERENCE_DATA_MAX_AGE_DAYS,
    REFERENCE_DATA_VINTAGES,
)
from data_validation import validate_calendar
from sources import (
    MARKET_DATA_MAX_FFILL_DAYS,
    MINER_EFFICIENCY_COLUMNS,
    MINER_EFFICIENCY_SOURCE_DATE_COLUMN,
    MINER_EFFICIENCY_SOURCE_URL_COLUMN,
    MINER_EFFICIENCY_VALUE_COLUMN,
    _SOURCE_OBSERVATION_DATE_PREFIX,
    _normalized_index,
    _source_observation_column,
)


# Coin Metrics miner efficiency is a monthly observation. Allow at most two monthly
# publication intervals before refusing to carry it further; freshness is validated from
# the retained source observation date, never inferred from a repeated daily value.
MINER_EFFICIENCY_MAX_AGE_DAYS = 62


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
            "PRICE_OUTLOOK_YEAR and PRICE_OUTLOOK_LEVELS in data_definitions.py."
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
