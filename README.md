# Bitcoin Report Library

Bitcoin market and on-chain analytics pipeline powering the Secret Satoshis research stack. The system delivers validated, internally consistent datasets optimized for downstream modeling, reporting, and visualization.

**This is the canonical producer for Secret Satoshis report and chart datasets.** All
data fetching, base metric calculation, and feature engineering for these datasets
happens here. Downstream consumers—including
[Bitcoin-Chart-Library](https://github.com/SecretSatoshis/Bitcoin-Chart-Library), the
bundled `dashboard/`—read the validated CSV outputs rather than duplicating their
calculations.

## Features

- **On-Chain Analytics**: Hash rate, difficulty, transaction metrics, UTXO age bands, address activity, miner revenue, and supply dynamics
- **Market Data Integration**: Multi-asset price data spanning equities, ETFs, commodities, forex, and cryptocurrencies
- **Valuation Models**: Metcalfe, time-based power law, Stock-to-Flow, Thermocap, NVT, MVRV, Reserve Risk, energy-based pricing models, and relative valuation metrics
- **Mining Signals**: Hash-rate trend metrics including strategy-aligned 30/60-day Hash Ribbons
- **Performance Tracking**: Rolling returns (7d, 90d, MTD, YTD, YOY), correlation analysis, volatility, and CAGR calculations
- **Cycle Analysis**: ATH drawdown tracking, halving epoch comparisons, and market cycle low indexing
- **Report Tables**: Pre-built tables for fundamentals summaries, ROI comparisons, monthly heatmaps, and performance comparisons
- **Chart-Ready Exports**: Pre-computed datasets for downstream visualization (drawdowns, cycle lows, halving eras, CAGR)

## Architecture

```
Bitcoin-Report-Library/
├── main.py              # Pipeline orchestrator
├── data_format.py       # Data access and feature engineering
├── report_tables.py     # Table generation and formatting
├── data_definitions.py  # Configuration and constants
├── validate_outputs.py  # Local publication checks used by CI
├── tests/               # Regression tests for calculations and ingestion
├── csv/                 # Output directory (consumed by Chart Library + dashboard)
├── dashboard/           # Live web dashboard (Evidence.dev)
├── .github/workflows/   # Daily data refresh
├── pyproject.toml       # Python 3.12 dependency contract
└── uv.lock              # Exact reproducible dependency graph
```

| Module | Responsibility |
|--------|----------------|
| `main.py` | Orchestrates end-to-end execution: data ingestion, metric calculation, table assembly, cycle analysis, CSV export |
| `data_format.py` | Fetches raw data from APIs, normalizes timestamps, engineers features, calculates derived metrics, computes cycle analysis (drawdowns, halvings, cycle lows) |
| `report_tables.py` | Builds tabular outputs: fundamentals, ROI, performance comparisons, valuations, heatmaps, OHLC |
| `data_definitions.py` | Central configuration: tickers, API settings, reference data, metric templates, constants |

### Data Flow

```
Sources (BRK, Yahoo Finance, CoinGecko, Alternative.me, Google Sheets)
    │
    ▼
data_format.py  ──►  Fetches & calculates all metrics
    │
    ▼
report_tables.py  ──►  Generates formatted report tables
    │
    ▼
csv/  ──►  All outputs exported as CSV
    │
    ├─►  Bitcoin-Chart-Library     (interactive HTML charts)
    └─►  dashboard/  ──►  Evidence.dev  ──►  Cloudflare Pages
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
2. Retrieves market data from Yahoo Finance and CoinGecko
3. Pulls weekly and recent daily OHLC data from BRK
4. Calculates derived metrics, mining signals, and valuation models (Metcalfe, power law, Hash Ribbons, Reserve Risk, MVRV, NVT, volatility, etc.)
5. Runs performance analysis (7d, 90d, MTD, YTD, YOY changes)
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
| **Yahoo Finance** | Equities, ETFs, indices, commodities, forex | `yfinance` library |
| **CoinGecko** | Altcoin prices, market caps, BTC dominance | Public API |
| **Alternative.me** | Fear & Greed Index | Public API |
| **Google Sheets** | Miner efficiency data | CSV export |

## Configuration

All configuration is centralized in `data_definitions.py`:

- **Tickers**: Asset symbols organized by category (stocks, ETFs, indices, commodities, forex, crypto)
- **Reference Data**: Fiat money supply, precious metals supply, gold allocation breakdown
- **API Settings**: Configured BRK series, endpoint URLs, timeout values
- **Model Parameters**: Metcalfe address bands, Bitcoin genesis anchor, Hash Ribbon windows, electricity-tariff scenarios, mining overhead assumptions, trading days, and unit conversions
- **Report Settings**: Analysis columns, correlation data columns, metrics templates

## Outputs

All outputs are written to `csv/` and served via GitHub Pages at `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/` for remote consumption by downstream projects.

The master metrics dataset is exported as gzipped CSV (`.csv.gz`) to keep the file under GitHub's size limits. `pd.read_csv()` reads `.csv.gz` files natively — no manual decompression needed.

### Report Tables

| File | Description |
|------|-------------|
| `master_metrics_data.csv.gz` | Complete dataset with all calculated metrics and change calculations (gzipped) |
| `fundamentals_table.csv` | Network performance, security, economics, valuation metrics |
| `summary_table.csv` | Labeled summary metrics with `Metric`, `Value`, and `Category` columns |
| `performance_table.csv` | Multi-asset performance comparison |
| `mtd_return_comparison.csv` | Month-to-date return from the latest positive close before the month began, plus the historical median projection |
| `ytd_return_comparison.csv` | Year-to-date return from the latest positive close before January 1, plus the historical median projection |
| `relative_value_comparison.csv` | Relative valuation metrics |
| `roi_table.csv` | Historical ROI by labeled time frame and entry date |
| `eoy_model_data.csv` | End-of-year price model inputs and 4-year growth rates, capped at `report_date` |
| `5k_bucket_table.csv` | Positive-price trading-day distribution in $5,000 buckets, capped at `report_date` |
| `1k_bucket_table.csv` | Positive-price trading-day distribution in $1,000 buckets, capped at `report_date` |
| `monthly_heatmap_data.csv` | Monthly/yearly returns measured from the latest positive prior-period close |
| `ohlc_data.csv` | BRK weekly OHLC price data using week-start labels |
| `report_ohlc_summary.csv` | Report-date daily OHLC plus week-to-date context capped at the report date |
| `summary_history.csv` | 31 daily endpoints spanning 30 calendar days for dashboard sparklines + exact 30d deltas |
| `onchain_price_models.csv` | Daily valuation models (Metcalfe, power law, Realized, STH/LTH Realized, canonical $0.05/kWh power expense, and 3× Realized) joined to BTC price through `report_date` |
| `electricity_cost_scenarios.csv` | Daily network energy inputs, $0.03–$0.07/kWh power-expense scenarios, break-even tariff, and retained legacy/Production Cost/Hayes/Energy Value comparisons through `report_date` |
| `network_model_metrics.csv` | Daily Metcalfe inputs and four address-band values, fitted power-law inputs/parameters, and 30/60-day Hash Ribbon metrics through `report_date` |
| `mtd_returns_history.csv` | Indexed MTD paths with row 0 as the shared prior-month close anchor; day 1 retains its actual move and the current series is capped at `report_date` |
| `ytd_returns_history.csv` | Indexed YTD paths with row 0 as the shared prior-year close anchor; calendar dates align across leap years and the current series is capped at `report_date` |
| `price_outlook.csv` | Hand-maintained Bear/Base/Bull cases plus retained support/resistance reference data; the bundled dashboard renders only the three case lines |

### Chart-Ready Datasets

These CSV files are pre-computed for downstream visualization by [Bitcoin-Chart-Library](https://github.com/SecretSatoshis/Bitcoin-Chart-Library):

| File | Description |
|------|-------------|
| `drawdown_data.csv` | ATH drawdown cycles with days since ATH and percentage decline |
| `cycle_low_data.csv` | Market cycle performance indexed from the lowest positive price observed inside each configured cycle window |
| `halving_data.csv` | Performance indexed from each Bitcoin halving with a positive day-0 source price; the pre-price Genesis era is omitted |
| `cagr_data.csv` | Rolling 2-year (730-row) and 4-year (1,460-row) CAGR values for the configured 13 downstream metrics, expressed in percentage points |

### Raw Data

| File | Description |
|------|-------------|
| `brk_onchain_raw.csv` | Raw BRK API on-chain data before transformations |

## Dashboard

The `dashboard/` subfolder is an [Evidence.dev](https://evidence.dev) BI-as-code dashboard that consumes the same CSV outputs as Chart Library and renders them as an interactive web report inside the shared Secret Satoshis navigation, hero, and footer design.

- **Live URL:** [dashboard.secretsatoshis.com](https://dashboard.secretsatoshis.com)
- **Local dev:** see [`dashboard/README.md`](dashboard/README.md) — `cd dashboard && npm ci && npm run sync:local && npm run sources && npm run dev`

## Dependencies

```
pandas==2.2.0
pyarrow==15.0.0
numpy==1.26.4
requests==2.32.3
yfinance==0.2.60
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
```

`main.py` contacts the configured live APIs and rewrites files in `csv/`. To view the
existing local CSVs without refreshing them first, skip those two commands.

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

The CI workflow runs the regression suite, regenerates the report, and then runs
`uv run --no-sync python validate_outputs.py` before committing refreshed CSVs. The
validator is local-only and rejects missing or truncated required files, non-finite
values, implausible row counts, report-date disagreements, invalid cycle-low baselines,
halving eras without a valid day-zero anchor, and inconsistent electricity, Metcalfe,
power-law, or Hash Ribbon calculations.

## License

GPLv3
