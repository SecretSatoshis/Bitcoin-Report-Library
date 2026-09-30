# Frozen migration reference

`release-2026-09-29.tar.gz` contains only the twelve dashboard inputs and original release manifest from Report Library commit `11dfde70eec2484c168c6fcef675a291b91c8c71`. No source fetching is needed to test it.

`reference-2026-09-29.json` records the old dashboard's rendered cards and tables, price payload fingerprints, and the old seasonal aggregator's complete output. Responsive duplicate tables were removed, retaining complete row coverage. Browser tests compare formatting and row order when displaying this release; adapter tests always use the frozen fixture and verify each chart observation.

Reference and replacement PNGs are kept locally in `dashboard-migration-review/`. Future daily reports do not overwrite this fixture.
