# Test fixtures

`release-2026-09-29.tar.gz` holds the twelve dashboard inputs and the release manifest from
Report Library commit `11dfde7`, so the tests run without fetching anything.

`reference-2026-09-29.json` holds the expected cards, table cells, price-chart fingerprints
and seasonal averages for that release. The unit tests check every chart value against it,
and the browser tests check every displayed cell when the built site shows this release.
