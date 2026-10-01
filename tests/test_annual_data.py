"""Annual reference series (annual_data.py)."""
import unittest
import warnings
from unittest.mock import patch

import pandas as pd
import requests

import annual_data

REPORT_DATE = pd.Timestamp("2026-09-30")


def _series(first, last, value=100.0):
    return lambda: pd.DataFrame({"year": range(first, last + 1), "value": value})


def _outage():
    raise requests.ConnectionError("down")


def _live_sources(**overrides):
    sources = {series: _series(2000, 2025) for series in annual_data.FETCHED_SERIES}
    sources["world_internet_users_pct"] = _series(2005, 2025, 50.0)
    sources.update(overrides)
    return patch.dict(annual_data.FETCHED_SERIES, sources)


class AnnualReferenceTests(unittest.TestCase):
    def test_every_series_is_published_once_per_year_with_its_source(self):
        with _live_sources():
            table = annual_data.get_annual_reference_data(REPORT_DATE, previous_release=None)
        self.assertEqual(set(table["series"]), set(annual_data.ANNUAL_SERIES))
        self.assertFalse(table.duplicated(["series", "year"]).any())
        self.assertEqual(list(table.columns), annual_data.COLUMNS)
        owners = table[table["series"] == "bitcoin_owners_millions"]
        self.assertTrue(owners["source_url"].str.startswith("https://crypto.com/").all())
        early = table[table["series"] == "world_internet_users"]
        self.assertEqual((early["year"].min(), early["year"].max()), (1990, 2004))

    def test_an_outage_reuses_the_last_release_with_its_original_date(self):
        with _live_sources():
            previous = annual_data.get_annual_reference_data(REPORT_DATE, previous_release=None)
        previous["retrieved_date"] = "2026-09-01"
        with _live_sources(world_population=_outage), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table = annual_data.get_annual_reference_data(REPORT_DATE, previous_release=lambda: previous)
        population = table[table["series"] == "world_population"]
        self.assertEqual(set(population["retrieved_date"]), {"2026-09-01"})
        self.assertTrue(any("world_population" in str(w.message) for w in caught))

    def test_a_format_change_fails_instead_of_reusing_old_rows(self):
        def changed():
            raise annual_data.SourceFormatError("renamed column")

        with _live_sources(world_population=changed), self.assertRaises(annual_data.SourceFormatError):
            annual_data.get_annual_reference_data(REPORT_DATE, previous_release=None)

    def test_a_series_too_far_behind_the_report_date_fails(self):
        with (_live_sources(us_median_household_income_usd=_series(2000, 2021)),
              self.assertRaisesRegex(RuntimeError, "more than 3 years old")):
            annual_data.get_annual_reference_data(REPORT_DATE, previous_release=None)

    def test_bad_values_are_a_format_error(self):
        with (_live_sources(world_internet_users_pct=_series(2005, 2025, 140.0)),
              self.assertRaises(annual_data.SourceFormatError)):
            annual_data.get_annual_reference_data(REPORT_DATE, previous_release=None)

    def test_fred_maintenance_page_is_an_outage(self):
        page = requests.Response()
        page.status_code, page._content = 200, b"<html>maintenance</html>"
        with (patch.object(annual_data.requests, "get", return_value=page),
              self.assertRaises(requests.RequestException)):
            annual_data._fred_median_income()


if __name__ == "__main__":
    unittest.main()
