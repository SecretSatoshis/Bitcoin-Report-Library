# Bitcoin Report Library

Bitcoin market and on-chain analytics pipeline powering the Secret Satoshis research stack. The system delivers deterministic, reproducible datasets optimized for downstream modeling, reporting, and visualization.

**This is the single data source for the entire Secret Satoshis analytics stack.** All data fetching, metric calculation, and feature engineering happens here. Downstream projects (e.g., [Bitcoin-Chart-Library](https://github.com/SecretSatoshis/Bitcoin-Chart-Library)) consume the CSV output directly — no API calls or duplicated logic.

## Features

- **On-Chain Analytics**: Hash rate, difficulty, transaction metrics, UTXO age bands, address activity, miner revenue, and supply dynamics
- **Market Data Integration**: Multi-asset price data spanning equities, ETFs, commodities, forex, and cryptocurrencies
- **Valuation Models**: Stock-to-Flow, Thermocap, NVT, MVRV, Reserve Risk, energy-based pricing models, and relative valuation metrics
- **Performance Tracking**: Rolling returns (7d, 90d, MTD, YTD, YOY), correlation analysis, Sharpe ratios, and CAGR calculations
- **Cycle Analysis**: ATH drawdown tracking, halving epoch comparisons, and market cycle low indexing
- **Report Generation**: Pre-built tables for weekly recaps, fundamentals summaries, ROI comparisons, and monthly heatmaps
- **Chart-Ready Exports**: Pre-computed datasets for downstream visualization (drawdowns, cycle lows, halving eras, CAGR)
- **Report Specs**: Agent-facing markdown reference for the Secret Satoshis Weekly Bitcoin Recap and its current CSV inputs

## Architecture

```
Bitcoin-Report-Library/
├── main.py              # Pipeline orchestrator
├── data_format.py       # Data access and feature engineering
├── report_tables.py     # Table generation and formatting
├── data_definitions.py  # Configuration and constants
├── csv/                 # Output directory (consumed by Chart Library)
├── reports/             # Report prompt reference
└── requirements.txt     # Python dependencies
```

| Module | Responsibility |
|--------|----------------|
| `main.py` | Orchestrates end-to-end execution: data ingestion, metric calculation, table assembly, cycle analysis, CSV export |
| `data_format.py` | Fetches raw data from APIs, normalizes timestamps, engineers features, calculates derived metrics, computes cycle analysis (drawdowns, halvings, cycle lows) |
| `report_tables.py` | Builds tabular outputs: fundamentals, ROI, performance comparisons, valuations, heatmaps, OHLC |
| `data_definitions.py` | Central configuration: tickers, API settings, reference data, metric templates, constants |
| `reports/` | Markdown report reference for the Weekly Bitcoin Recap and its current CSV inputs |

### Data Flow

```
APIs (BRK, yfinance, CoinGecko, Kraken)
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
    ▼
Bitcoin-Chart-Library  ──►  Reads CSVs, generates charts
```

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/SecretSatoshis/Bitcoin-Report-Library.git
cd Bitcoin-Report-Library

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The pipeline executes in sequence:
1. Fetches on-chain data from BRK API (~90 metrics)
2. Retrieves market data from Yahoo Finance and CoinGecko
3. Pulls OHLC data from Kraken
4. Calculates derived metrics and valuation models (Reserve Risk, MVRV, NVT, volatility, etc.)
5. Runs performance analysis (7d, 90d, MTD, YTD, YOY changes)
6. Computes rolling CAGR for all metrics
7. Generates report tables
8. Computes cycle analysis (drawdowns, halving eras, cycle lows)
9. Exports all outputs to `csv/`

**Note:** The CSV output is consumed by [Bitcoin-Chart-Library](https://github.com/SecretSatoshis/Bitcoin-Chart-Library) for visualization. Run this pipeline first before generating charts.

## Data Sources

| Source | Data Type | Endpoint |
|--------|-----------|----------|
| **BRK (Bitview) API** | On-chain metrics, difficulty, supply data | `bitview.space/api` |
| **Yahoo Finance** | Equities, ETFs, indices, commodities, forex | `yfinance` library |
| **CoinGecko** | Altcoin prices, market caps, BTC dominance | Public API |
| **Kraken** | Bitcoin OHLC price data | Public API |
| **Alternative.me** | Fear & Greed Index | Public API |
| **Google Sheets** | Miner efficiency data | CSV export |

## Configuration

All configuration is centralized in `data_definitions.py`:

- **Tickers**: Asset symbols organized by category (stocks, ETFs, indices, commodities, forex, crypto)
- **Reference Data**: Fiat money supply, precious metals supply, gold allocation breakdown
- **API Settings**: BRK metrics list (~90 on-chain metrics), endpoint URLs, timeout values
- **Model Parameters**: Electricity costs, trading days, unit conversions
- **Report Settings**: Analysis columns, correlation data columns, metrics templates

## Report Specs

The `reports/` folder stores the documentation reference for the canonical Secret Satoshis weekly report format that sits on top of the CSV pipeline.

- `reports/weekly_bitcoin_recap.md` documents the Weekly Bitcoin Recap structure and prompts

This file is not executable code. It is a reference document for agents and maintainers who need to understand which current Report Library datasets support each recap section.

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
| `mtd_return_comparison.csv` | Month-to-date return comparison |
| `ytd_return_comparison.csv` | Year-to-date return comparison |
| `relative_value_comparison.csv` | Relative valuation metrics |
| `roi_table.csv` | Historical ROI by labeled time frame and entry date |
| `eoy_model_data.csv` | End-of-year price model projections |
| `5k_bucket_table.csv` | Price distribution in $5,000 buckets with current bucket markers |
| `1k_bucket_table.csv` | Price distribution in $1,000 buckets with current bucket markers |
| `monthly_heatmap_data.csv` | Monthly returns heatmap data |
| `ohlc_data.csv` | OHLC price data |

### Chart-Ready Datasets

These CSV files are pre-computed for downstream visualization by [Bitcoin-Chart-Library](https://github.com/SecretSatoshis/Bitcoin-Chart-Library):

| File | Description |
|------|-------------|
| `drawdown_data.csv` | ATH drawdown cycles with days since ATH and percentage decline |
| `cycle_low_data.csv` | Market cycle performance indexed from cycle lows |
| `halving_data.csv` | Performance indexed from each Bitcoin halving (5 eras) |
| `cagr_data.csv` | Rolling CAGR calculations for all metrics (1Y, 2Y, 3Y, 4Y) |

### Raw Data

| File | Description |
|------|-------------|
| `brk_onchain_raw.csv` | Raw BRK API on-chain data before transformations |

## Dependencies

```
pandas==2.2.0
numpy==1.26.4
requests==2.32.3
yfinance==0.2.60
```

## License

GPLv3
