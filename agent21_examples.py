"""Facts behind two example answers on the Agent 21 landing page.

The page shows how Agent 21 analyzes the market (price against its on-chain cost basis)
and reviews the Secret Satoshis outlook. Its wording is fixed; the figures, dates and
chart series come from this file, rebuilt with each release so the examples stay
current. It sits beside the release rather than in its manifest because it is a view
of released values, not a dataset.
"""

import json
from pathlib import Path

import pandas as pd

AGENT21_EXAMPLES_FILE = "agent21_examples.json"
SCHEMA_VERSION = 1

COST_BASIS_COLUMNS = ["price_close", "realized_price", "sth_realized_price", "realizedcap_multiple_3"]
CHART_YEARS = 5
CHART_STEP_DAYS = 7
# A stretch below realized price is counted over the year up to its last close below.
STRETCH_WINDOW = pd.Timedelta(days=365)
CANDLE_WEEKS = 52


def _day(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _price(value) -> float:
    return round(float(value), 2)


def _side_since(price: pd.Series, level: pd.Series) -> tuple[str, str]:
    """Which side of `level` price closed on last, and the first close of that run."""
    above = price >= level
    current = bool(above.iloc[-1])
    switched = above[above != current]
    start = above.index[0] if switched.empty else above.index[above.index > switched.index[-1]][0]
    return ("above" if current else "below"), _day(start)


def cost_basis_facts(report_data: pd.DataFrame, report_date) -> dict:
    """Price against realized, short-term-holder realized and 3x realized price."""
    end = pd.Timestamp(report_date)
    frame = report_data.loc[:end, COST_BASIS_COLUMNS].dropna()
    if frame.empty or frame.index[-1] != end:
        raise ValueError("Cost-basis series must reach the report date")
    price = frame["price_close"]
    last = frame.iloc[-1]

    realized_side, realized_since = _side_since(price, frame["realized_price"])
    sth_side, sth_since = _side_since(price, frame["sth_realized_price"])
    above_3x = price[price > frame["realizedcap_multiple_3"]]

    below_realized = price < frame["realized_price"]
    stretch = None
    if below_realized.any():
        last_below = below_realized[below_realized].index[-1]
        window = below_realized.loc[last_below - STRETCH_WINDOW : last_below]
        stretch = {
            "start": _day(window[window].index[0]),
            "end": _day(last_below),
            "days": int(window.sum()),
        }

    start = end - pd.DateOffset(years=CHART_YEARS) + pd.Timedelta(days=1)
    chart = frame.loc[start:]
    # Sampled back from the report date so the latest close is always the last point.
    chart = chart.iloc[::-1].iloc[::CHART_STEP_DAYS].iloc[::-1]

    return {
        "close": _price(last["price_close"]),
        "realized_price": _price(last["realized_price"]),
        "sth_realized_price": _price(last["sth_realized_price"]),
        "realized_price_3x": _price(last["realizedcap_multiple_3"]),
        "realized_side": realized_side,
        "realized_side_since": realized_since,
        "sth_side": sth_side,
        "sth_side_since": sth_since,
        "last_close_above_3x": _day(above_3x.index[-1]) if not above_3x.empty else None,
        "last_stretch_below_realized": stretch,
        "chart": {
            "dates": [_day(d) for d in chart.index],
            "price": [_price(v) for v in chart["price_close"]],
            "realized_price": [_price(v) for v in chart["realized_price"]],
            "sth_realized_price": [_price(v) for v in chart["sth_realized_price"]],
            "realized_price_3x": [_price(v) for v in chart["realizedcap_multiple_3"]],
        },
    }


def _level_name(label: str) -> str:
    """'Support $73,757 - 2024 Prior ATH' -> '2024 Prior ATH'."""
    return label.split(" - ", 1)[1].strip() if " - " in label else label


def _nearest(levels: pd.DataFrame, kind: str, close: float, side: str):
    rows = levels[levels["type"] == kind]
    rows = rows[rows["price"] < close] if side == "below" else rows[rows["price"] > close]
    if rows.empty:
        return None
    row = rows.loc[rows["price"].idxmax() if side == "below" else rows["price"].idxmin()]
    return {"price": _price(row["price"]), "name": _level_name(row["label"])}


def outlook_facts(report_data: pd.DataFrame, candles: pd.DataFrame, levels: pd.DataFrame, year: int, report_date) -> dict:
    """The year's outlook cases and levels against the latest close and weekly candles."""
    end = pd.Timestamp(report_date)
    price = report_data["price_close"].loc[:end].dropna()
    if price.empty or price.index[-1] != end:
        raise ValueError("Price must reach the report date")
    close = float(price.iloc[-1])
    prior_year = price.loc[: f"{end.year - 1}-12-31"]
    if prior_year.empty:
        raise ValueError("Price history must include the previous year's close")
    past_year = price[price.index > end - pd.Timedelta(days=365)]

    cases = levels[levels["type"] == "case"]
    weekly = candles[candles["interval"] == "weekly"]
    weekly = weekly[pd.to_datetime(weekly["period_end"]) > end - pd.Timedelta(weeks=CANDLE_WEEKS)]

    return {
        "year": int(year),
        "close": _price(close),
        "previous_year_close": _price(prior_year.iloc[-1]),
        "days_left_in_year": int((pd.Timestamp(f"{end.year}-12-31") - end).days),
        "cases": [{"name": str(row["label"]), "price": _price(row["price"])} for _, row in cases.iterrows()],
        "nearest_support": _nearest(levels, "support", close, "below"),
        "nearest_resistance": _nearest(levels, "resistance", close, "above"),
        "year_high": {"date": _day(past_year.idxmax()), "close": _price(past_year.max())},
        "year_low": {"date": _day(past_year.idxmin()), "close": _price(past_year.min())},
        "weekly_candles": [
            {
                "start": _day(row["period_start"]),
                "open": _price(row["Open"]),
                "high": _price(row["High"]),
                "low": _price(row["Low"]),
                "close": _price(row["Close"]),
                "complete": bool(row["complete"]),
            }
            for _, row in weekly.iterrows()
        ],
    }


def build_agent21_examples(report_data, candles, levels, year, report_date) -> dict:
    """The landing page's example facts for this release."""
    return {
        "schema_version": SCHEMA_VERSION,
        "report_date": _day(report_date),
        "cost_basis": cost_basis_facts(report_data, report_date),
        "outlook": outlook_facts(report_data, candles, levels, year, report_date),
    }


def write_agent21_examples(examples: dict, output_dir="csv") -> Path:
    """Write the facts as compact JSON and return the path."""
    path = Path(output_dir) / AGENT21_EXAMPLES_FILE
    path.write_text(json.dumps(examples, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    return path
