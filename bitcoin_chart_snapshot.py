"""Collect the recent BRK Bitcoin chart data that extends the daily release."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import argparse
import json
import math
from pathlib import Path

from sources import _http_get_with_retries


BRK_SERIES = "https://bitview.space/api/series"
HOUR = 3600
SERIES = {
    "dates": ("date", "day1", 4),
    "ohlc": ("price_ohlc", "day1", 4),
    "realized": ("realized_price", "day1", 4),
    "sth": ("sth_realized_price", "day1", 4),
    "hours": ("timestamp", "hour1", 26),
    "closes": ("price_close", "hour1", 26),
}


def read_series(metric, index, count):
    reply = _http_get_with_retries(
        f"{BRK_SERIES}/{metric}/{index}", params={"start": -count}, timeout=15
    ).json()
    start, end, data = reply.get("start"), reply.get("end"), reply.get("data")
    if (reply.get("index") != index or type(start) is not int or type(end) is not int
            or not isinstance(data, list) or not 0 < len(data) <= count
            or end - start != len(data)):
        raise ValueError(f"BRK {metric}/{index} returned invalid series metadata")
    return dict(enumerate(data, start))


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def build_snapshot(series, as_of):
    now = int(as_of.timestamp())
    daily = []
    for index in sorted(series["dates"].keys() & series["ohlc"].keys()):
        day = date.fromisoformat(series["dates"][index])
        row = series["ohlc"][index]
        if (day > as_of.date() or not isinstance(row, list) or len(row) != 4
                or not all(positive(value) for value in row)):
            raise ValueError(f"Invalid BRK daily price for {day}")
        open_, high, low, close = row
        if low > min(open_, close) or high < max(open_, close):
            raise ValueError(f"Invalid BRK OHLC bounds for {day}")
        realized, sth = series["realized"].get(index), series["sth"].get(index)
        daily.append({
            "date": day.isoformat(), "open": open_, "high": high, "low": low, "close": close,
            "realized": realized if positive(realized) else None,
            "sth": sth if positive(sth) else None,
        })
    dates = [date.fromisoformat(row["date"]) for row in daily]
    if (not dates or (as_of.date() - dates[-1]).days > 1
            or any((after - before).days != 1 for before, after in zip(dates, dates[1:]))):
        raise ValueError("BRK daily observations are stale or discontinuous")

    # The timestamp and close calls can cross an hourly boundary. Match their
    # common absolute indexes to keep each price paired with its own hour.
    hourly = []
    for index in sorted(series["hours"].keys() & series["closes"].keys()):
        time, price = series["hours"][index], series["closes"][index]
        if (type(time) is not int or time <= 0 or time > now or time % HOUR
                or not positive(price)):
            raise ValueError("Invalid BRK hourly price")
        hourly.append({"time": time, "price": price})
    if (len(hourly) < 25 or now - hourly[-1]["time"] >= 2 * HOUR
            or any(after["time"] - before["time"] != HOUR for before, after in zip(hourly, hourly[1:]))):
        raise ValueError("BRK hourly observations do not cover the last day")

    return {
        "schema_version": 1,
        "as_of": as_of.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "daily": daily,
        "hourly": hourly,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as executor:
        requests = {name: executor.submit(read_series, *query) for name, query in SERIES.items()}
        series = {name: future.result() for name, future in requests.items()}
    snapshot = build_snapshot(series, datetime.now(timezone.utc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(snapshot, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
