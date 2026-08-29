"""Regression tests for the generated data-release landing page.

The page, its sitemap and its structured data are produced by
build_release_page.py inside the daily workflow. These assertions exist so a
stale date or an invalid Dataset field fails the build rather than shipping.
"""

from __future__ import annotations

import csv
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"
SITEMAP = ROOT / "sitemap.xml"
BASE = "https://secretsatoshis.github.io/Bitcoin-Report-Library"
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def latest_csv_date() -> str:
    latest = ""
    for path in (ROOT / "csv").glob("*.csv"):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for row in csv.reader(handle):
                for cell in row:
                    value = cell.strip()
                    if DATE.fullmatch(value) and value > latest:
                        latest = value
    return latest


@unittest.skipUnless(INDEX.is_file(), "release page not generated yet")
class ReleasePageSeoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = INDEX.read_text(encoding="utf-8")
        self.data = json.loads(
            re.search(r'application/ld\+json">(.*?)</script>', self.document, re.S).group(1)
        )
        self.latest = latest_csv_date()

    def test_date_modified_tracks_the_release(self) -> None:
        """Catches the failure this replaced: hand-edited HTML drifting behind the data."""
        self.assertEqual(self.data["dateModified"], self.latest)

    def test_sitemap_lastmod_tracks_the_release(self) -> None:
        self.assertIn(f"<lastmod>{self.latest}</lastmod>", SITEMAP.read_text(encoding="utf-8"))

    def test_temporal_coverage_ends_at_the_latest_data(self) -> None:
        self.assertEqual(self.data["temporalCoverage"].split("/")[1], self.latest)

    def test_every_csv_is_listed_once(self) -> None:
        names = {p.name for p in (ROOT / "csv").glob("*.csv")}
        urls = [d["contentUrl"] for d in self.data["distribution"]]
        self.assertEqual(len(urls), len(set(urls)))
        self.assertEqual({u.rsplit("/", 1)[-1] for u in urls}, names)

    def test_per_file_coverage_finds_dates_in_any_column(self) -> None:
        """ROI dates live in column three and must not render as missing."""
        row = re.search(
            r'<a href="csv/roi_table\.csv">.*?</tr>', self.document, re.S
        )
        self.assertIsNotNone(row)
        self.assertRegex(row.group(0), r"\d{4}-\d{2}-\d{2} → \d{4}-\d{2}-\d{2}")

    def test_dataset_does_not_claim_a_licence_over_upstream_data(self) -> None:
        self.assertNotIn("license", self.data)
        self.assertNotIn("codeRepository", self.data)
        self.assertEqual(self.data["isBasedOn"]["@type"], "SoftwareSourceCode")
        self.assertTrue(self.data["isBasedOn"]["license"].endswith("gpl-3.0.html"))

    def test_social_and_canonical_tags_are_present(self) -> None:
        for tag in (
            'rel="canonical"',
            'name="description"',
            'property="og:title"',
            'property="og:description"',
            'property="og:image"',
            'property="og:image:alt"',
            'name="twitter:card"',
        ):
            self.assertIn(tag, self.document, f"missing {tag}")
        self.assertEqual(len(re.findall(r"<h1[ >]", self.document)), 1)

    def test_sitemap_has_no_fragments_and_stays_on_one_host(self) -> None:
        locs = re.findall(r"<loc>([^<]+)</loc>", SITEMAP.read_text(encoding="utf-8"))
        self.assertTrue(locs)
        self.assertTrue(all(u.startswith(BASE) for u in locs))
        self.assertEqual([u for u in locs if "#" in u], [])
