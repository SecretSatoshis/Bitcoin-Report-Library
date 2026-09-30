# Bitcoin Report Dashboard

An Evidence.dev dashboard for the Bitcoin Report Library outputs used by the Secret Satoshis research stack. It uses the same branded navigation, hero, typography, spacing, and footer structure as the Secret Satoshis Chart Library while rendering the dashboard natively through Evidence.

**Live:** [dashboard.secretsatoshis.com](https://dashboard.secretsatoshis.com)

The dashboard can read CSVs from local report outputs or from the published GitHub Pages CSV endpoint:

- Local outputs: `../csv/`
- Published CSV base path: `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/`

Source repo: `https://github.com/SecretSatoshis/Bitcoin-Report-Library`

## Requirements

- Node.js 24 — the verified runtime pinned in `.nvmrc`. `package.json` `engines`
  accepts Node 22.13 through 24, but only Node 24 is tested.
- npm 12 (pinned by `packageManager`)

## Setup

From inside `dashboard/`:

```bash
nvm use
npm ci
npm run sync:local
npm run sources
npm run dev
```

Run `nvm use` before Evidence commands when using nvm, or otherwise ensure
`node --version` reports Node 24. Older runtimes such as Node 16 can stall during CSV
source compilation. Use `npm install` only when intentionally changing dependencies and
updating `package-lock.json`; otherwise `npm ci` keeps local and production installs aligned.

All dashboard packages are build-time dependencies: the deployed output is static. The
current Evidence release still pins Svelte 4 and Vite 5, so `npm audit --omit=dev` is the
production-exposure check until Evidence publishes a compatible toolchain upgrade.
The advisories that remain open for those pinned packages concern the development server
and test tooling (Vite, esbuild, Vitest), or cross-site scripting through untrusted
content (Svelte and SvelteKit server rendering, ECharts). The deployed pages are
prerendered from this repository's own CSV release, with no user-supplied content. Patch
advisories that are fixable within the pinned majors with `npm update <package>`.

Use `npm run sync:remote` to pull published CSVs from GitHub Pages instead of local report outputs.

To refresh the complete dataset before launching the dashboard, run this from the
repository root in a separate terminal:

```bash
uv run --no-sync python main.py
uv run --no-sync python validate_outputs.py
```

Then return to `dashboard/` and run `sync:local`, `sources`, and `dev`. If you only
want to view the existing CSV outputs, the Python refresh can be skipped.

To verify a release build without starting the development server:

```bash
nvm use
npm run sync:local
npm run sources
npm run build
```

The static site is written to ignored directory `build/`. Use `npm run preview` to
serve that production build locally.

## Newsletter visual exports

The weekly Secret Satoshis newsletter uses exact, frozen exports from the same local
Dashboard build as the report data. Install the pinned Playwright browser once after
`npm ci`:

```bash
node node_modules/playwright-core/cli.js install chromium
```

After the completed Report Library run, create the production build and export the
five approved newsletter visuals:

```bash
npm run sync:local
npm run sources
npm run build
npm run export:newsletter -- \
  --report-date YYYY-MM-DD \
  --output-dir /absolute/path/to/run/visuals
```

The report date must equal the Dashboard's latest data date. The exporter refuses to
overwrite files, renders at a fixed 1440px dark viewport and 2x pixel density, checks
capture contents/dimensions, and writes SHA-256 provenance to `visual-manifest.json`.
Its immutable outputs are:

- the Bitcoin Snapshot Market Data card row only;
- the Bitcoin Price section with BTC, realized/STH/3x realized prices, 3-month/1-year/200-week
  moving averages and the bear/base/bull cases (electricity/power expense is excluded);
- the Monthly Bitcoin Price Return Heatmap;
- separate MTD and YTD seasonal-return charts for newsletter legibility.

The export-only layout is activated by the exporter and does not change the live
Dashboard layout. If the price outlook shows its "Price outlook unavailable" error, the
export fails with that message rather than capturing an empty chart.

## How It Works

1. `npm run sync:local` copies the dashboard CSV subset from `../csv/`.
2. `npm run sync:remote` downloads the same CSV subset from GitHub Pages.
3. Both modes stage the files in `.sync-staging/`, verify every file against the
   release's `release_manifest.json`, decode `bitcoin_candles.csv.gz`, and only then
   replace `sources/bitcoin_report_library/` in one step. A missing manifest, a hash
   mismatch, or any failed file leaves the existing sources untouched.
4. Evidence reads CSV files from `sources/bitcoin_report_library/`.
5. `pages/index.md` defines the dashboard and SQL queries.
6. `npm run build` writes the deployable static site to `build/`.

## Dashboard Data Scope

The sync script intentionally uses only the CSVs required by the dashboard:

- `summary_table.csv`
- `summary_history.csv`
- `fundamentals_table.csv`
- `performance_table.csv`
- `monthly_heatmap_data.csv`
- `relative_value_comparison.csv`
- `roi_table.csv`
- `onchain_price_models.csv`
- `mtd_returns_history.csv`
- `ytd_returns_history.csv`
- `price_outlook.csv` (case levels plus their `outlook_year`, which labels the outlook)
- `bitcoin_candles.csv.gz` (verified, then decoded to `bitcoin_candles.csv` for Evidence)

Wide files such as `master_metrics_data.csv.gz` are intentionally excluded because they can slow or hang Evidence CSV type inference.

## Production Deploy

The dashboard is published at [dashboard.secretsatoshis.com](https://dashboard.secretsatoshis.com). It is deployed by a Vercel project whose Git integration is configured outside this repository and builds every push to `main`; each commit shows a Vercel deployment status. The repository's daily data-refresh workflow is scheduled for 00:30 UTC, shortly after the completed UTC day, then tests, regenerates, and validates the report before committing refreshed CSVs; that commit triggers the dashboard rebuild. GitHub starts scheduled workflows late — recent runs began around 05:15–05:45 UTC (about 1–2 AM in New York) — so the dashboard usually updates overnight rather than the same evening.

The production build sequence is `npm ci → sync:remote → sources → build`, with the static `build/` folder served behind a CDN. The commit that triggers the build is the same one GitHub Pages is still deploying, and Pages serves files with a 10-minute CDN cache. `sync:remote` therefore reads the release in the checked-out `../csv/release_manifest.json` and waits (up to 12 minutes) until Pages serves that release, requesting each file with a release-keyed query so a cached copy of an older file cannot be used. A release counts as current only when its report date is newer, or the same date with a `generated_at` time at least as new: a manual rerun of a day that already had a scheduled release produces a second release for the same date, and the build must not accept the earlier copy. If Pages never catches up the build fails instead of deploying stale data. Switching the Vercel build command to `sync:local` would remove the wait entirely, because the checkout already holds the triggering release. The build finishes by replacing Evidence's hardcoded X publisher attribution with `@SecretSatoshis`; it fails if the upstream tag changes instead of silently publishing incorrect metadata. Because the hosting integration is external, verify those build settings in Vercel when changing the Node version or production command.

`sync:local` and `sync:remote` both require `csv/release_manifest.json` and verify every
dashboard input against its hashes before Evidence ingests them.

## Key Files

- `pages/index.md` — branded dashboard hero, report content, and SQL queries
- `pages/+layout.svelte` — shared Secret Satoshis navigation/footer around the Evidence layout
- `components/PriceOutlookChart.svelte` — adapts the price/model query and candles to the shared chart payload; shows a visible error when its inputs do not reach the report date
- `components/chart-colors.json` and `components/chart-events.json` — model line colors and historical events, written by the Chart Library's `sync-dashboard.py`
- `static/shared-chart/` — vendored Chart Library renderer, frame page, and `source-manifest.json` hashes
- `sources/bitcoin_report_library/connection.yaml` — CSV datasource config
- `scripts/download-data.mjs` — local/remote CSV sync script
- `scripts/export-newsletter-visuals.mjs` — deterministic Dashboard-to-newsletter PNG exporter
- `scripts/fix-social-attribution.mjs` — post-build correction for Evidence's hardcoded X attribution
- `static/robots.txt` and `static/sitemap.xml` — crawler policy and canonical dashboard URL
- `evidence.config.yaml` — Evidence plugins, theme, and color config
- `app.css` — shared site-shell tokens and custom dashboard styling (cypherpunk dark theme, JetBrains Mono + Syne)

## Quarterly newsletter exports

Use the same built Dashboard and exporter with the quarterly profile:

```bash
npm run export:newsletter -- --profile quarterly --report-date YYYY-MM-DD --output-dir /absolute/new/export-directory
```

This exports six Dashboard views: the four-year price outlook (anchor `#price-outlook`,
including the January 1 annual marker), Stock Market Index Performance, Sector Performance, Macro Asset
Class Performance, Bitcoin Industry Performance, and Relative Valuation. The relative-value
capture uses a wider canvas to fit its existing columns. The default weekly profile still
exports its five established visuals. Both profiles validate the built data date, PNG bytes,
dimensions, and exact output inventory; the quarterly pipeline also checks producer commits.

A quarterly report pairs these six captures from one committed quarter-end release with
the savings images from the matching Investment Strategy export and a frozen Chart Library
YTD chart from the same release. The exporter produces only the Dashboard's six; it is not
a complete quarterly report bundle.

## Shared price-outlook chart

The price-outlook section uses the Chart Library's shared Lightweight Charts
renderer, theme, legend, range/scale controls, candles and PNG compositor. Scenario
cards stay above the chart; dashed scenario levels and the outlook-year marker
remain inside it. The initial view is weekly candles over four years through the
outlook year's end, with a linear scale and no gridlines. All shared historical events are included alongside the
year-start marker. Scenario labels sit at the left behind the data.

`components/PriceOutlookChart.svelte` adapts the dashboard's existing price/model
query to the shared payload. Moving averages arrive precomputed in the CSV. Weekly and
monthly line observations select the corresponding candle's final observation
date; OHLC arrives already aggregated from the Report Library. Data sync verifies
`bitcoin_candles.csv.gz` against the release manifest and decodes it for Evidence.

Shared assets are vendored in `static/shared-chart/`, so dashboard builds and
frozen newsletter captures do not depend on a live chart website or sibling
checkout. `source-manifest.json` records the source hashes. To refresh the shared
renderer after a Chart Library change, run from the Chart Library checkout:

```bash
.venv/bin/python scripts/sync-dashboard.py ../Bitcoin-Report-Library/dashboard/static/shared-chart
```

The newsletter capture waits for the iframe's chart-ready contract, verifies its
report date, and includes its legend text in existing visual-manifest checks.

The compact legend keeps Bitcoin first, then orders the model lines by their latest
values. Right/Left scale labels, YTD / 1Y / 4Y / 10Y / All ranges, Show all, Remove all
(retaining Bitcoin), isolation and event controls come from the shared renderer.
Scenario text is compact on the dashboard and remains above the exported chart.

The displayed moving averages are the `3-month MA`, `1-year MA` and `200-week MA`
columns of `onchain_price_models.csv`: 90-day, 364-day and 1,400-day calendar windows on
the canonical daily close, computed over the full history by the Report Library, and
recomputed independently by `validate_outputs.py` before each release. A window with missing
closes stays null. Other consumers read the same published columns, so any level they quote
matches the chart. The file also carries 50-day and 200-day averages, which the chart does
not draw.
