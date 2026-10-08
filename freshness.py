"""Checks that decide whether a run may publish.

Stale market or miner data only warns. Missing, gapped or stale on-chain data, an
out-of-date price outlook or an expired reference figure fails the run.
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
    BRK_DERIVED_SERIES,
    MARKET_DATA_MAX_FFILL_DAYS,
    MINER_EFFICIENCY_COLUMNS,
    MINER_EFFICIENCY_SOURCE_DATE_COLUMN,
    MINER_EFFICIENCY_SOURCE_URL_COLUMN,
    MINER_EFFICIENCY_VALUE_COLUMN,
    _SOURCE_OBSERVATION_DATE_PREFIX,
    _normalized_index,
    _source_observation_column,
)


# Miner efficiency is monthly; warn once it is two publication intervals old.
MINER_EFFICIENCY_MAX_AGE_DAYS = 62


# On-chain series that must have a value on the report date. BRK can return a partial
# payload with a 200 status, which would otherwise surface only as NaN cells.
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
    # Investor sentiment (NUPL comes from market_cap and realized_cap)
    "supply_in_profit",
    # Daily flows in the summary table; like every on-chain series, they are never filled
    "coinbase_sum_24h_usd",
    "transfer_volume_sum_24h_usd",
]

# Miner revenue feeds the all-time total behind thermocap, so a hole in it shifts every later
# total without looking wrong; supply is the divisor of every per-coin price. Neither may have
# an interior gap.
CUMULATIVE_ONCHAIN_INPUTS = ["coinbase_sum_24h_usd"]
GAP_CHECKED_ONCHAIN_INPUTS = CUMULATIVE_ONCHAIN_INPUTS + ["supply"]


def _ordinary_market_columns(data: pd.DataFrame) -> list:
    """Market columns under the ordinary fill budget: not on-chain (BRK series and the daily
    flows derived from them), miner or marker columns."""
    onchain_columns = {metric for metric in BRK_METRICS if metric != "timestamp"}
    onchain_columns.update(BRK_DERIVED_SERIES)
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
    """Warn about market series whose last real observation is older than `max_age_days`.

    Must run before `forward_fill_market_data`, which drops the observation-date markers.
    A stale ticker does not stop the run; its values stay NaN. Returns the stale and
    missing series, empty when all are fresh.
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
    """Return a copy with market gaps filled and the observation-date markers dropped.

    Market data (market caps included) is filled up to `market_max_age_days` from its
    real observation. Miner efficiency carries its last estimate, capped only when
    `miner_max_age_days` is given. On-chain series are never filled.
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
    """Check miner efficiency reaches the report date, and warn when its source is old.

    Raises when the value, its source date or its source URL is missing, or the source
    date is after the report date. An observation older than `max_age_days` only warns.
    Returns the warnings, empty when the observation is current.
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
    """Raise unless the data reaches the report date with every required on-chain value."""
    metrics = metrics or REQUIRED_ONCHAIN_METRICS
    report_date = pd.to_datetime(report_date).normalize()

    available = data.index[data.index <= report_date]
    if len(available) == 0:
        raise RuntimeError(
            f"No data on or before the report date ({report_date.date()}). "
            "Upstream fetch returned nothing usable."
        )

    # Every table reads the report-date row; failing here names the real cause.
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



def assert_price_outlook_current(report_date, outlook_year: int = PRICE_OUTLOOK_YEAR) -> None:
    """Raise when the price outlook is not for the report date's year.

    Otherwise the first run of a new year would publish last year's levels as current.
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
    """Raise when a hand-maintained reference figure's recorded check date is too old."""
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
    """Raise when a gap-checked input has a hole between its first and last observation.

    Leading nulls before a series starts are allowed.
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
