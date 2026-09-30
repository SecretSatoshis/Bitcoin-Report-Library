# Bitcoin Report Library

Bitcoin market and on-chain analytics pipeline powering the Secret Satoshis research stack. The system delivers validated, internally consistent datasets optimized for downstream modeling, reporting, and visualization.

**This is the canonical producer for Secret Satoshis report and chart datasets.** All
data fetching, base metric calculation, and feature engineering for these datasets
happens here. Downstream consumers—including
[Bitcoin-Chart-Library](https://github.com/SecretSatoshis/Bitcoin-Chart-Library), the
bundled `dashboard/`—read the validated CSV outputs rather than duplicating their
calculations.

The bundled Dashboard also supports deterministic PNG exports built from the same frozen
CSV run with a hashed `visual-manifest.json`; see `dashboard/README.md`.

## Features

- **On-Chain Analytics**: Hash rate, difficulty, transaction metrics, UTXO age bands, address activity, miner revenue, and supply dynamics
- **Market Data Integration**: Multi-asset price data spanning equities, ETFs, indices, commodities and the US dollar index
- **Valuation Models**: Metcalfe, time-based power law, Thermocap, NVT, MVRV, Reserve Risk, electricity-cost and Hayes production-cost models, and relative valuation metrics
- **Mining Signals**: Hash-rate trend metrics including strategy-aligned 30/60-day Hash Ribbons
- **Performance Tracking**: Rolling returns (7d, 90d, MTD, YTD, and YOY for Bitcoin), correlation analysis, volatility, and 4-year CAGR calculations
- **Cycle Analysis**: ATH drawdown tracking, halving epoch comparisons, and market cycle low indexing
- **Report Tables**: Pre-built tables for fundamentals summaries, ROI comparisons, monthly heatmaps, and performance comparisons
- **Chart-Ready Exports**: Pre-computed datasets for downstream visualization (drawdowns, cycle lows, halving eras)

## Architecture

```
Bitcoin-Report-Library/
├── main.py              # Pipeline orchestrator
├── sources.py           # Fetches and merges BRK, Yahoo Finance and the miner sheet
├── freshness.py         # Checks that decide whether a run may publish
├── metrics.py           # Every calculated metric, change, CAGR and correlation
├── cycles.py            # Drawdown, cycle-low and halving series
├── report_tables.py     # Published report tables
├── data_definitions.py  # Configuration and constants
├── candle_data.py       # Daily/weekly/monthly candles and period metric snapshots
├── data_validation.py   # Shared calendar and candle contracts
├── release_manifest.py  # Writes the hashed release manifest
├── validate_outputs.py  # Local publication checks used by CI
├── build_release_page.py # Generates the public data landing page and sitemap
├── index.html           # Generated public data-release landing page
├── sitemap.xml          # Generated public release sitemap
├── tests/               # One test file per module
├── csv/                 # Output directory (consumed by Chart Library + dashboard)
├── dashboard/           # Live web dashboard (Evidence.dev)
├── .github/workflows/   # Daily data refresh
├── pyproject.toml       # Python 3.12 dependency contract
└── uv.lock              # Exact reproducible dependency graph
```

| Module | Responsibility |
|--------|----------------|
| `main.py` | Orchestrates end-to-end execution: fetch, publish checks, metrics, tables, then writes every output in one step |
| `sources.py` | Fetches BRK on-chain series and daily candles, Yahoo Finance prices and historical market caps, and the Coin Metrics miner-efficiency sheet, and merges them onto the BRK daily calendar |
| `freshness.py` | Bounds the market-data fill and refuses to publish on stale, gapped or missing on-chain data or on an out-of-date reference figure |
| `metrics.py` | On-chain valuation models, relative-value prices, network models, electricity costs, changes, CAGRs and correlations |
| `cycles.py` | Drawdown, cycle-low and halving-era series for Chart Library |
| `report_tables.py` | Builds the published tables: summary, fundamentals, performance, ROI, MTD/YTD comparisons, heatmap, OHLC, relative value |
| `data_definitions.py` | Central configuration: tickers, API settings, reference data, metric templates, constants |
| `candle_data.py` | Aggregates BRK daily candles into weekly and monthly periods through the report date, and the matching weekly and monthly metric snapshots |
| `data_validation.py` | Shared contracts: complete daily/weekly calendars and valid OHLC candles |
| `release_manifest.py` | Writes `release_manifest.json` (report date plus SHA-256 and size of exactly the files the run published) after all exports finish |
| `validate_outputs.py` | Re-checks the finished release against the report date recorded in the manifest before CI publishes it |
| `build_release_page.py` | Reads the files the manifest lists and regenerates its crawlable landing page, Dataset structured data, file inventory, and sitemap |

### Data Flow

```
Sources (BRK, Yahoo Finance, Google Sheets)
    │
    ▼
sources.py  ──►  Fetches and merges every source
    │
    ▼
freshness.py  ──►  Refuses to publish stale or incomplete data
    │
    ▼
metrics.py / cycles.py  ──►  Calculates every metric and cycle series
    │
    ▼
report_tables.py  ──►  Builds the published tables
    │
    ▼
csv/  ──►  All outputs exported as CSV
    │
    ├─►  Bitcoin-Chart-Library     (interactive HTML charts)
    └─►  dashboard/  ──►  Evidence.dev  ──►  Vercel
                                              dashboard.secretsatoshis.com
```

## Installation

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Node.js 24 and npm 12 (dashboard only; pinned in `dashboard/`)

### Setup

```bash
# Clone the repository
git clone https://github.com/SecretSatoshis/Bitcoin-Report-Library.git
cd Bitcoin-Report-Library

# Create the Python 3.12 environment from the reviewed lockfile
uv sync --locked
```

## Usage

```bash
uv run --no-sync python main.py
```

The pipeline executes in sequence:
1. Fetches the configured on-chain and market series from the BRK API
2. Retrieves market data from Yahoo Finance
3. Pulls full daily OHLC history from BRK; weekly and monthly candles are aggregated from it through the report date
4. Calculates derived metrics, mining signals, and valuation models (Metcalfe, power law, Hash Ribbons, Reserve Risk, MVRV, NUPL, NVT, volatility, etc.)
5. Runs performance analysis (7d, 90d, MTD and YTD changes for the tracked prices, plus Bitcoin YOY)
6. Generates report tables
7. Computes cycle analysis (drawdowns, halving eras, cycle lows)
8. Exports all outputs to `csv/`

**Note:** The CSV output is consumed by
[Bitcoin-Chart-Library](https://github.com/SecretSatoshis/Bitcoin-Chart-Library) for
visualization. Run this pipeline first when Chart Library is configured with a local
`REPORT_CSV_DIR`; its default mode instead reads the latest published GitHub Pages CSVs.

## Data Sources

| Source | Data Type | Endpoint |
|--------|-----------|----------|
| **BRK (Bitview) API** | On-chain metrics, difficulty, supply data | `bitview.space/api` |
| **Yahoo Finance** | Equities, ETFs, indices, commodities, US dollar index | `yfinance` library |
| **Google Sheets** | Miner efficiency data | CSV export |

## Configuration

All configuration is centralized in `data_definitions.py`:

- **Tickers**: Asset symbols organized by category (stocks, which also get market caps; Bitcoin equities, price only; ETFs, indices, commodities, forex)
- **Reference Data**: Fiat money supply, precious metals supply, and gold allocation breakdown, each with an explicit reviewed vintage and maximum age
- **API Settings**: Configured BRK series, endpoint URLs, timeout values
- **Model Parameters**: Metcalfe address input, Bitcoin genesis anchor, Hash Ribbon windows, electricity-tariff scenarios, and unit conversions
- **Report Settings**: Analysis columns, correlation data columns, metrics templates

## Outputs

All data outputs are written to `csv/` and served from the GitHub Pages base path `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/` for remote consumption by downstream projects. The repository root publishes a generated data-release landing page and sitemap covering every `.csv` and `.csv.gz` file; the release date comes from `release_manifest.json`, and file coverage, counts, and download links are read from the same completed release.

The master metrics dataset is exported as gzipped CSV (`.csv.gz`) to keep the file under GitHub's size limits. `pd.read_csv()` reads `.csv.gz` files natively — no manual decompression needed.

**Naming.** Data files use snake_case columns and a `date` column for the day each row
describes; period files (`bitcoin_candles`, weekly and monthly metrics) are keyed by
`period_start` with the source row's `observation_date`. Report tables meant for display
(summary, fundamentals, performance, relative value, ROI, return comparisons, heatmap)
use Title Case headers. Ticker symbols keep their case (`SPY_close`,
`NVDA_market_cap`). Change columns end `_7d_change`, `_90d_change`, `_mtd_change`,
`_ytd_change` and `_yoy_change`; CAGRs end `_4y_cagr`. All percentages are in percentage
points. Series that need a Bitcoin price are blank before the first traded price
(2010-08-16), never `0`.

**Model parameters.** The fitted power-law exponent and scale and the Metcalfe scale are
constants, so they are published once in `release_manifest.json` under
`model_parameters` rather than repeated on every row.

`metrics.calculate_nvt_price_models` publishes `nvt_price_30d`, `nvt_price_90d`
and `nvt_price_365d`. Each multiplies the 730-day median NVT by the corresponding
rolling median of BRK USD transfer volume, then divides by current supply.
The Chart Library displays these three input-smoothed NVT Price models. The
unsmoothed daily `nvt_price` and its existing averages remain available for
compatibility. The 365-day model is an additional long-term reference alongside
the standard 30/90-day pair. All calculations remain in Report Library.
`nvt_price_multiple_30d`, `nvt_price_multiple_90d` and
`nvt_price_multiple_365d` divide the Bitcoin closing price by each corresponding
model. A multiple of 1.0 means price equals the model; missing or nonpositive
model values remain unavailable. The Chart Library defaults to the 90-day multiple.

Power Law valuation boundaries are also published as prepared USD curves:
`power_law_price_band_058`, `power_law_price_band_173` and
`power_law_price_band_300`. They multiply the existing fitted model by the same
0.58×, 1.73× and 3× thresholds used for dashboard valuation labels; the existing
`power_law_price` supplies the 1× boundary. The curves share the model's fit date
and are included in daily, weekly and monthly metrics. Chart shading is a visual
interpretation of these fixed reviewed ranges, not a forecast confidence interval.

### Report Tables

| File | Description |
|------|-------------|
| `master_metrics_data.csv.gz` | Complete dataset with all calculated metrics, change calculations and 4-year CAGRs (gzipped) |
| `fundamentals_table.csv` | Network performance, security, economics, valuation metrics |
| `summary_table.csv` | Labeled summary metrics with `Metric`, `Value`, and `Category` columns. Investor Sentiment is all on-chain: supply in profit (%), a Fear & Greed label from the NUPL zone of the 7-day average (Capitulation, Hope / Fear, Optimism / Anxiety, Belief / Denial, Euphoria / Greed), and a valuation label from price against the power-law fair value in standard-deviation bands (Undervalued below 0.58×, Below Fair Value to 1.00×, Above Fair Value to 1.73×, Overvalued to 3.00×, Extremely Overvalued above) |
| `performance_table.csv` | Multi-asset performance comparison: one Bitcoin row (category `Bitcoin`), then equity indexes, sectors, macro assets and Bitcoin-industry stocks. The release fails if any asset lacks a price or return. The 90-day BTC correlation pairs each asset's returns between its own trading days with BTC's return over the same span, so weekends and holidays add no artificial zero returns |
| `mtd_return_comparison.csv` | Month-to-date return from the latest positive close before the month began, plus the historical median projection |
| `ytd_return_comparison.csv` | Year-to-date return from the latest positive close before January 1, plus the historical median projection |
| `relative_value_comparison.csv` | Bitcoin's price if its market cap matched each asset (M0 supplies, gold, silver, large stocks): `Market Cap (USD)`, `BTC Price at Market Cap` and `Move Needed (%)`. Gold and silver use each day's futures close |
| `roi_table.csv` | ROI over each time frame (1 Day to 10 Year) with its `Start Date` and `Start Price` |
| `monthly_heatmap_data.csv` | Monthly and yearly returns measured from the latest positive prior-period close, one row per `Year` plus 4-Year Average, Median and Average rows |
| `report_ohlc_summary.csv` | One row: the report-date daily candle (`daily_*`) and the week-to-date candle (`week_to_date_*`) from `week_start` |
| `summary_history.csv` | 31 daily endpoints spanning 30 calendar days for dashboard sparklines + exact 30d deltas |
| `onchain_price_models.csv` | Daily valuation models (Metcalfe, power law, Realized, STH/LTH Realized, canonical $0.05/kWh power expense, and 3× Realized) joined to BTC price from the first traded price through `report_date`, plus the 50-day, 3-month, 200-day, 1-year and 200-week moving averages (50, 90, 200, 364 and 1,400 daily closes; empty until the window is complete). The Dashboard price chart draws the 3-month, 1-year and 200-week averages |
| `mtd_price_paths.csv` | Each year's month-to-date price path rebased to this month's starting close, one column per year; row 0 is the shared prior-month close and the current year stops at `report_date` |
| `ytd_price_paths.csv` | Each year's year-to-date price path rebased to this year's starting close, one column per year; row 0 is the shared prior-year close, dates align across leap years, and the current year stops at `report_date` |
| `price_outlook.csv` | Hand-maintained Bear/Base/Bull cases, their forecast year, and retained support/resistance reference data; the website and bundled dashboard render the three case lines |

### Chart-Ready Datasets

These CSV files are pre-computed for downstream visualization by [Bitcoin-Chart-Library](https://github.com/SecretSatoshis/Bitcoin-Chart-Library):

| File | Description |
|------|-------------|
| `drawdown_data.csv` | ATH drawdown cycles with days since ATH and percentage decline |
| `cycle_low_data.csv` | Market cycle performance indexed from the lowest positive price observed inside each configured cycle window |
| `release_manifest.json` | Shared release ID, report date, generation time, fitted model parameters, and SHA-256/size records for every published CSV |
| `halving_data.csv` | Performance indexed from each Bitcoin halving with a positive day-0 source price; the pre-price Genesis era is omitted |

## Dashboard

The `dashboard/` subfolder is an [Evidence.dev](https://evidence.dev) BI-as-code dashboard that consumes the same CSV outputs as Chart Library and renders them as an interactive web report inside the shared Secret Satoshis navigation, hero, and footer design.

- **Live URL:** [dashboard.secretsatoshis.com](https://dashboard.secretsatoshis.com)
- **Local dev:** see [`dashboard/README.md`](dashboard/README.md) — `cd dashboard && npm ci && npm run sync:local && npm run sources && npm run dev`

## Daily Refresh

The GitHub Actions workflow is scheduled for **00:30 UTC**, shortly after the UTC day
closes and safely after the New York market close. GitHub starts scheduled workflows
late, so in practice runs have begun around 05:15–05:45 UTC. It refreshes the source data,
runs the regression and output-validation suites, rebuilds the public release page and sitemap, and commits the
validated CSV and public release outputs. That publication supplies the dashboard and
the downstream Chart Library, which checks hourly for a new release rather than
assuming a fixed time.

The run fails closed if BRK has no on-chain row for the exact report date, miner
revenue or supply contains an internal gap (neither is ever forward-filled), a
hand-maintained reference dataset exceeds its reviewed-age budget, or a dated export
(including the latest weekly candle) extends past the completed report date. The fitted
power-law and Metcalfe coefficients are kept in the master file for every date, so fitted
valuation series remain reproducible.

The power-law valuation bands are fixed in `data_definitions.py` (one standard deviation
of the log multiple since 2015, reviewed yearly), so the label never drifts on its own.

## Dependencies

```
pandas==3.0.6
pyarrow==25.0.1
numpy==2.5.3
requests==2.34.2
yfinance==1.7.0
```

## Local Development

Run the complete regression suite from the repository root:

```bash
uv run --no-sync python -m unittest discover -s tests -t . -v
```

Refresh every live data source, rebuild the report outputs, and validate them:

```bash
uv run --no-sync python main.py
uv run --no-sync python validate_outputs.py
uv run --no-sync python build_release_page.py
```

`validate_outputs.py` validates against the report date recorded in
`csv/release_manifest.json`, so a run that finishes after UTC midnight still validates
the day it built. Pass `--report-date YYYY-MM-DD` to check a specific release.

`main.py` contacts the configured live APIs and rewrites files in `csv/`. To view the
existing local CSVs without refreshing them first, skip the three release commands above.

Launch the dashboard from a second VS Code terminal:

```bash
cd dashboard
nvm use                 # when using nvm; .nvmrc selects the required Node 24 runtime
npm ci                  # reproducible install from package-lock.json
npm run sync:local
npm run sources
npm run dev
```

Open the local URL printed by Evidence. After rerunning `main.py`, stop the dashboard,
repeat `npm run sync:local` and `npm run sources`, then start `npm run dev` again.
Use `npm run sync:remote` instead when you want the published GitHub CSVs.

Before a release, verify a production-style static bundle against the current local outputs:

```bash
cd dashboard
nvm use
npm run sync:local
npm run sources
npm run build
```

The generated `dashboard/build/` directory and compiled Evidence caches are ignored by
Git. Only source code, configuration, the lockfile, and report CSV outputs are committed.

The CI workflow runs the regression suite on every pull request and every push to `main`.
On the schedule or a manual run it also regenerates the report into an empty `csv/`,
validates it, rebuilds the public release page and sitemap, reruns the release-page checks
against the new data, and replaces the published `csv/` folder, so an output the pipeline
no longer writes is removed rather than left behind. The output validator rejects missing
or truncated required files, non-finite values, implausible row counts, report-date
disagreements, Bitcoin prices or returns that disagree between files, recomputed moving
averages, investor-sentiment rows or fundamentals that disagree with the master data,
performance rows without a price or return, invalid cycle-low baselines, halving eras
without a valid day-zero anchor, and a manifest whose file list or hashes do not match.

## Frozen chart candles

The daily pipeline fetches BRK daily OHLC history, trims its leading all-zero
pre-market era, and prepares chart data through the completed report date.
`candle_data.py` exports `bitcoin_candles.csv.gz` (daily, Monday–Sunday weekly,
and calendar-month OHLC with period/observation dates and completion flags),
plus `weekly_metrics_data.csv.gz` and `monthly_metrics_data.csv.gz`. Metric
snapshots are keyed by `period_start` and copy the master row for the period's
`observation_date` (its last included day), preserving missing values. An initial partial historical week/month is omitted;
the latest partial period is included through the report date.

These files are part of the verified release manifest. Daily candle closes must
match the master prices; missing daily observations and inconsistent candles
fail the build. Chart Library performs no source gathering or OHLC aggregation.


## Dashboard presentation

The dashboard price outlook uses the shared Chart Library renderer: weekly candles,
a linear four-year view, historical events and the annual start marker. Scenario
cards remain above the plot and its PNG export. The former Trading Range graphics
have been removed from the dashboard. See [dashboard/README.md](dashboard/README.md) for local builds, renderer
sync and frozen newsletter exports.

## License

GPLv3. The vendored TradingView renderer and bundled fonts retain their own license
and notice files under `dashboard/static/shared-chart/assets/`.
