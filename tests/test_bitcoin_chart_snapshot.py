from datetime import datetime, timezone
from unittest import TestCase

from bitcoin_chart_snapshot import build_snapshot


NOW = datetime(2026, 11, 26, 13, 30, tzinfo=timezone.utc)
HOUR = 3600
LATEST = int(NOW.timestamp()) // HOUR * HOUR


def series():
    return {
        "dates": {1: "2026-11-25", 2: "2026-11-26"},
        "ohlc": {1: [100, 110, 95, 105], 2: [105, 110, 101, 106]},
        "realized": {1: 50},
        "sth": {1: 70, 2: 71},
        "hours": {i: LATEST - (25 - i) * HOUR for i in range(26)},
        "closes": {i: 100 + i for i in range(1, 26)},
    }


class ChartSnapshotTests(TestCase):
    def test_coherent_snapshot_aligns_source_indexes(self):
        snapshot = build_snapshot(series(), NOW)
        self.assertEqual(snapshot["as_of"], "2026-11-26T13:30:00Z")
        self.assertEqual(len(snapshot["hourly"]), 25)
        self.assertEqual(snapshot["hourly"][-1], {"time": LATEST, "price": 125})
        self.assertIsNone(snapshot["daily"][-1]["realized"])
        self.assertEqual(snapshot["daily"][-1]["sth"], 71)

    def test_invalid_sources_do_not_replace_the_previous_snapshot(self):
        data = series()
        data["closes"].pop(24)
        with self.assertRaisesRegex(ValueError, "cover the last day"):
            build_snapshot(data, NOW)
        data = series()
        data["ohlc"][2] = [105, 105, 101, 106]
        with self.assertRaisesRegex(ValueError, "OHLC bounds"):
            build_snapshot(data, NOW)
