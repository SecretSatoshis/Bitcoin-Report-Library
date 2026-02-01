# Bitcoin Report Library

Bitcoin market and on-chain analytics pipeline powering the Secret Satoshis research stack. The system delivers deterministic, reproducible datasets optimized for downstream modeling, reporting, and visualization.

## Features

- **On-Chain Analytics**: Hash rate, difficulty, transaction metrics, UTXO age bands, address activity, miner revenue, and supply dynamics
- **Market Data Integration**: Multi-asset price data spanning equities, ETFs, commodities, forex, and cryptocurrencies
- **Valuation Models**: Stock-to-Flow, Thermocap, NVT, MVRV, energy-based pricing models, and relative valuation metrics
- **Performance Tracking**: Rolling returns (7d, 90d, MTD, YTD), correlation analysis, Sharpe ratios, and CAGR calculations
- **Report Generation**: Pre-built tables for weekly recaps, fundamentals summaries, ROI comparisons, and monthly heatmaps

## Architecture

```
Bitcoin-Report-Library/
├── main.py              # Pipeline orchestrator
├── data_format.py       # Data access and feature engineering
├── report_tables.py     # Table generation and formatting
├── data_definitions.py  # Configuration and constants
├── csv/                 # Output directory
└── requirements.txt     # Python dependencies
```

| Module | Responsibility |
|--------|----------------|
| `main.py` | Orchestrates end-to-end execution: data ingestion, metric calculation, table assembly, CSV export |
| `data_format.py` | Fetches raw data from APIs, normalizes timestamps, engineers features, calculates derived metrics |
| `report_tables.py` | Builds tabular outputs: fundamentals, ROI, performance comparisons, valuations, heatmaps, OHLC |
| `data_definitions.py` | Central configuration: tickers, API settings, reference data, metric templates, constants |

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
1. Fetches on-chain data from BRK API
2. Retrieves market data from Yahoo Finance and CoinGecko
3. Pulls OHLC data from Kraken
4. Calculates derived metrics and valuation models
5. Generates report tables
6. Exports all outputs to `csv/`

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
- **API Settings**: BRK metrics list, endpoint URLs, timeout values
- **Model Parameters**: Electricity costs, trading days, unit conversions
- **Report Settings**: Analysis columns, correlation data columns, metrics templates

## Outputs

All outputs are written to `csv/` as unstyled CSV files for direct ingestion into spreadsheets, BI tools, or research notebooks.

| File | Description |
|------|-------------|
| `master_metrics_data.csv` | Complete dataset with all calculated metrics |
| `fundamentals_table.csv` | Network performance, security, economics, valuation metrics |
| `summary_table.csv` | Summary metrics for report |
| `performance_table.csv` | Multi-asset performance comparison |
| `mtd_return_comparison.csv` | Month-to-date return comparison |
| `ytd_return_comparison.csv` | Year-to-date return comparison |
| `relative_value_comparison.csv` | Relative valuation metrics |
| `roi_table.csv` | Historical ROI by entry date |
| `eoy_model_data.csv` | End-of-year price model projections |
| `5k_bucket_table.csv` | Price distribution in $5,000 buckets |
| `1k_bucket_table.csv` | Price distribution in $1,000 buckets |
| `monthly_heatmap_data.csv` | Monthly returns heatmap data |
| `ohlc_data.csv` | OHLC price data |

## Dependencies

```
pandas==2.2.0
numpy==1.26.4
requests==2.32.3
yfinance==0.2.60
```

## License

GPLv3
