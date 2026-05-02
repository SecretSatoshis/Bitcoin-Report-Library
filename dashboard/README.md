# Bitcoin Report Dashboard

An Evidence.dev dashboard for the Secret Satoshis Weekly Bitcoin Recap data outputs.

**Live:** [dashboard.secretsatoshis.com](https://dashboard.secretsatoshis.com)

The dashboard can read CSVs from local report outputs or from the published GitHub Pages CSV endpoint:

- Local outputs: `../csv/`
- Published outputs: `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/`

Source repo: `https://github.com/SecretSatoshis/Bitcoin-Report-Library`

## Requirements

- Node.js
- npm

## Setup

From inside `dashboard/`:

```bash
npm install
npm run sync:local
npm run sources
npm run dev
```

Use `npm run sync:remote` to pull published CSVs from GitHub Pages instead of local report outputs.

## How It Works

1. `npm run sync:local` copies the dashboard CSV subset from `../csv/`.
2. `npm run sync:remote` downloads the same CSV subset from GitHub Pages.
3. Evidence reads CSV files from `sources/bitcoin_report_library/`.
4. `pages/index.md` defines the dashboard and SQL queries.

## Dashboard Data Scope

The sync script intentionally uses only the CSVs required by the dashboard:

- `summary_table.csv`
- `summary_history.csv`
- `fundamentals_table.csv`
- `performance_table.csv`
- `monthly_heatmap_data.csv`
- `relative_value_comparison.csv`
- `ohlc_data.csv`
- `1k_bucket_table.csv`
- `5k_bucket_table.csv`
- `roi_table.csv`
- `onchain_price_models.csv`
- `mtd_returns_history.csv`
- `ytd_returns_history.csv`
- `price_outlook.csv`

Wide files such as `master_metrics_data.csv.gz` and `cagr_data.csv` are intentionally excluded because they can slow or hang Evidence CSV type inference.

## Production Deploy

The dashboard is published at [dashboard.secretsatoshis.com](https://dashboard.secretsatoshis.com) and rebuilds automatically on every push to `main` that touches `dashboard/` or `csv/`. The daily data-refresh workflow commits fresh CSVs at 16:00 UTC, which triggers a same-day dashboard rebuild.

The build pipeline is `npm ci → sync:remote → sources → build`, with the static `build/` folder served behind a CDN.

## Key Files

- `pages/index.md` — main dashboard page and SQL queries
- `pages/+layout.svelte` — Evidence layout, logo, and page chrome
- `sources/bitcoin_report_library/connection.yaml` — CSV datasource config
- `scripts/download-data.mjs` — local/remote CSV sync script
- `evidence.config.yaml` — Evidence plugins, theme, and color config
- `app.css` — custom brand styling (cypherpunk dark theme, JetBrains Mono + Syne)
