# Bitcoin Report Dashboard

An Evidence.dev dashboard for the Bitcoin Report Library outputs used by the Secret Satoshis research stack.

**Live:** [dashboard.secretsatoshis.com](https://dashboard.secretsatoshis.com)

The dashboard can read CSVs from local report outputs or from the published GitHub Pages CSV endpoint:

- Local outputs: `../csv/`
- Published outputs: `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/`

Source repo: `https://github.com/SecretSatoshis/Bitcoin-Report-Library`

## Requirements

- Node.js 24 (the verified and pinned runtime in `.nvmrc`)
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

## How It Works

1. `npm run sync:local` copies the dashboard CSV subset from `../csv/`.
2. `npm run sync:remote` downloads the same CSV subset from GitHub Pages.
3. Evidence reads CSV files from `sources/bitcoin_report_library/`.
4. `pages/index.md` defines the dashboard and SQL queries.
5. `npm run build` writes the deployable static site to `build/`.

## Dashboard Data Scope

The sync script intentionally uses only the CSVs required by the dashboard:

- `summary_table.csv`
- `summary_history.csv`
- `fundamentals_table.csv`
- `performance_table.csv`
- `monthly_heatmap_data.csv`
- `relative_value_comparison.csv`
- `1k_bucket_table.csv`
- `5k_bucket_table.csv`
- `roi_table.csv`
- `onchain_price_models.csv`
- `mtd_returns_history.csv`
- `ytd_returns_history.csv`
- `price_outlook.csv`

Wide files such as `master_metrics_data.csv.gz` and `cagr_data.csv` are intentionally excluded because they can slow or hang Evidence CSV type inference.

## Production Deploy

The dashboard is published at [dashboard.secretsatoshis.com](https://dashboard.secretsatoshis.com). Its Cloudflare Pages Git integration is configured outside this repository to rebuild when relevant `dashboard/` or `csv/` changes reach `main`. The repository's daily data-refresh workflow runs at 16:00 UTC, tests and regenerates the report, validates the outputs, and commits refreshed CSVs; that commit triggers the same-day dashboard rebuild.

The production build sequence is `npm ci → sync:remote → sources → build`, with the static `build/` folder served behind a CDN. Because the hosting integration is external, verify those build settings in Cloudflare when changing the Node version or production command.

## Key Files

- `pages/index.md` — main dashboard page and SQL queries
- `pages/+layout.svelte` — Evidence layout, logo, and page chrome
- `sources/bitcoin_report_library/connection.yaml` — CSV datasource config
- `scripts/download-data.mjs` — local/remote CSV sync script
- `evidence.config.yaml` — Evidence plugins, theme, and color config
- `app.css` — custom brand styling (cypherpunk dark theme, JetBrains Mono + Syne)
