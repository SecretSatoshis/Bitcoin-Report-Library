# Bitcoin Report Library

Welcome to the **Bitcoin Report Library**, the repository powering the reports and tables featured in the **Secret Satoshis Newsletter**. 

---

## How It Works

The codebase is modular and structured around the following core components:

- **`main.py`** – Orchestrates the report pipeline. It pulls data, generates tables/charts, and compiles everything into a final report.
- **`data_format.py`** – Retrieves and formats time series data, calculates key metrics, and aligns all datasets into a master frame.
- **`report_tables.py`** – Builds tables and charts including OHLC analysis, ROI, fundamentals, comparative performance, and valuation models.
- **`datapane_weekly_bitcoin_recap.py`** – Assembles all visual content into an interactive HTML report using the Datapane library.
- **`data_definitions.py`** – Central location for defining tickers, label mappings, and parameters used throughout the report.

---

## Output

Each run of the tool produces:

- **HTML Report**:
  - Located in `html/Weekly_Bitcoin_Recap.html`
  - Includes charts, performance tables, and metrics overview

- **CSV Exports**:
  - Stored in the `csv/` folder
  - Includes key datasets used in the report

---

## Usage

You can check this repository daily for updated metrics — the reports are automatically refreshed using GitHub Actions. Alternatively, you can run the report locally at any time.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/SecretSatoshis/Bitcoin-Report-Library.git
   cd Bitcoin-Report-Library
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the Report:**
   ```bash
   python main.py
   ```

---

## Data Sources

The project integrates data from the following providers:

- **Bitcoin Data**:
  - CoinMetrics
  - CoinGecko
  - Blockstream API
  - Alternative.me (Fear and Greed Index)
- **Macro Data**:
  - CryptoVoices
- **Traditional Finance Data**:
  - Yahoo Finance

---

## License

This project is licensed under the terms of the GPLv3 license.
