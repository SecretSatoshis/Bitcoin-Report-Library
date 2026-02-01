"""
Data definitions and configuration for Bitcoin analytics pipeline.

This module contains all static configuration, ticker lists, reference data,
and API settings used throughout the Bitcoin report generation system.

Sections:
    - Market Data: Tickers, dates, and asset categories
    - Reference Data: Fiat supply, precious metals supply
    - Report Configuration: Metrics, columns, and templates
    - API Configuration: BRK metrics, URLs, and request settings
    - Model Parameters: Electric price model constants
"""
import datetime
import pandas as pd


# =============================================================================
# MARKET DATA CONFIGURATION
# =============================================================================

# Asset tickers organized by category for yfinance/CoinGecko API calls
tickers = {
    "stocks": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "BRK-A",
        "BRK-B",
        "TSM",
        "V",
        "JPM",
        "PYPL",
        "GS",
        "COIN",
        "XYZ",
        "MSTR",
        "MARA",
        "RIOT",
    ],
    "etfs": [
        "BITQ",
        "CLOU",
        "ARKK",
        "XLK",
        "QQQ",
        "IUIT.L",
        "VTI",
        "TLT",
        "LQD",
        "JNK",
        "GLD",
        "XLF",
        "XLRE",
        "SHY",
        "XLE",
        "FANG.AX",
        "SPY",
        "IEMG",
        "AGG",
        "WGMI",
        "VXUS",
    ],
    "indices": ["^GSPC", "^VIX", "^IXIC", "^TNX", "^TYX", "^FVX", "^IRX", "^BCOM"],
    "commodities": ["GC=F", "CL=F", "SI=F"],
    "forex": [
        "DX=F",
        "AUDUSD=X",
        "CHFUSD=X",
        "CNYUSD=X",
        "EURUSD=X",
        "GBPUSD=X",
        "HKDUSD=X",
        "INRUSD=X",
        "JPYUSD=X",
        "RUBUSD=X",
    ],
    "crypto": ["ethereum", "ripple", "dogecoin", "binancecoin", "tether"],
}

# Stock tickers extracted for market cap calculations
stock_tickers = tickers["stocks"]

# Start date for historical TradFi data (format: YYYY-MM-DD)
market_data_start_date = "2010-01-01"

# First Bitcoin halving date - used as start date for statistics calculations
stats_start_date = "2012-11-28"

# Report date defaults to yesterday (T-1) to ensure data availability
report_date = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=1))


# =============================================================================
# REFERENCE DATA
# =============================================================================

# Global fiat money supply (M0) by country in USD trillions
# Source: Central bank data, updated periodically
fiat_money_data_top10 = pd.DataFrame(
    {
        "Country": [
            "United States",
            "China",
            "Eurozone",
            "Japan",
            "United Kingdom",
            "Switzerland",
            "India",
            "Australia",
            "Russia",
            "Hong Kong",
            "Global Fiat Supply",
        ],
        "US Dollar Trillion": [
            5.73,
            5.11,
            5.19,
            4.20,
            1.09,
            0.58,
            0.56,
            0.24,
            0.30,
            0.25,
            26.1,
        ],
    }
)

# Above-ground precious metals supply in troy ounces
# Gold: ~6.1B oz, Silver: ~30.9B oz (World Gold Council estimates)
gold_silver_supply = pd.DataFrame(
    {
        "Metal": ["Gold", "Silver"],
        "Supply in Billion Troy Ounces": [6100000000, 30900000000],
    }
)

# Gold market allocation by use case (World Gold Council)
gold_supply_breakdown = pd.DataFrame(
    {
        "Gold Supply Breakdown": [
            "Jewellery",
            "Private Investment",
            "Official Country Holdings",
            "Other",
        ],
        "Percentage Of Market": [47.00, 22.00, 17.00, 14.00],
    }
)


# =============================================================================
# REPORT CONFIGURATION
# =============================================================================

# Metrics for which to calculate 7/30/365-day moving averages
moving_avg_metrics = [
    "hash_rate",
    "daily_active_addresses_sending",
    "TxCnt",
    "sent_usd",
    "TxTfrValMeanUSD",
    "TxTfrValMedUSD",
    "fee_usd_average",
    "fee_btc_average",
    "subsidy_btc_sum",
    "coinbase_usd_sum",
    "nvt_price",
    "nvt_price_adj",
]

# Columns that need change calculations (7d, 90d, MTD, YTD)
# These are the only columns passed to run_data_analysis()
analysis_columns = [
    # Bitcoin price and on-chain metrics
    "price_close",
    "hash_rate",
    "TxCnt",
    "sent_usd",
    "7_day_ma_TxTfrValMeanUSD",
    "daily_active_addresses_sending",
    "addrs_over_10k_sats_addr_count",
    "coinbase_usd_sum",
    "fee_usd_sum",
    "supply_pct_1_year_plus",
    "usd_velocity",
    # Equity ETFs
    "SPY_close",
    "QQQ_close",
    "VTI_close",
    "VXUS_close",
    # Sector ETFs
    "XLK_close",
    "XLF_close",
    "XLE_close",
    "XLRE_close",
    # Macro indicators
    "DX=F_close",
    "GLD_close",
    "AGG_close",
    "^BCOM_close",
    # Bitcoin-related equities
    "MSTR_close",
    "XYZ_close",
    "COIN_close",
    "WGMI_close",
]

# Column names for correlation analysis
correlation_data = [
    "price_close",
    "AAPL_close",
    "MSFT_close",
    "GOOGL_close",
    "AMZN_close",
    "NVDA_close",
    "TSLA_close",
    "META_close",
    "BRK-A_close",
    "BRK-B_close",
    "TSM_close",
    "V_close",
    "JPM_close",
    "PYPL_close",
    "GS_close",
    "FANG.AX_close",
    "BITQ_close",
    "CLOU_close",
    "ARKK_close",
    "XLK_close",
    "QQQ_close",
    "IUIT.L_close",
    "VTI_close",
    "TLT_close",
    "LQD_close",
    "JNK_close",
    "GLD_close",
    "XLF_close",
    "XLRE_close",
    "SHY_close",
    "XLE_close",
    "SPY_close",
    "IEMG_close",
    "AGG_close",
    "WGMI_close",
    "VXUS_close",
    "^GSPC_close",
    "^VIX_close",
    "^IXIC_close",
    "^TNX_close",
    "^TYX_close",
    "^FVX_close",
    "^IRX_close",
    "GC=F_close",
    "CL=F_close",
    "SI=F_close",
    "DX=F_close",
    "AUDUSD=X_close",
    "^BCOM_close",
    "CHFUSD=X_close",
    "CNYUSD=X_close",
    "EURUSD=X_close",
    "GBPUSD=X_close",
    "HKDUSD=X_close",
    "INRUSD=X_close",
    "JPYUSD=X_close",
    "RUBUSD=X_close",
    "ethereum_close",
    "ripple_close",
    "dogecoin_close",
    "binancecoin_close",
    "tether_close",
    "COIN_close",
    "XYZ_close",
    "MSTR_close",
    "MARA_close",
    "RIOT_close",
]

# Template for weekly fundamentals table: {section: {label: (column, format_type)}}
metrics_template = {
    "Network Performance": {
        "Total Address Count": ("addrs_over_1sat_addr_count", "number"),
        "Address Count > $10": ("addrs_over_10k_sats_addr_count", "number"),
        "Active Addresses": ("daily_active_addresses_sending", "number"),
        "Supply Held 1+ Year %": ("supply_pct_1_year_plus", "percent"),
        "Transaction Count": ("TxCnt", "number"),
        "Transaction Volume": ("sent_usd", "currency"),
        "Transaction Fee USD": ("fee_usd_sum", "currency"),
    },
    "Network Security": {
        "Hash Rate": ("hash_rate", "number"),
        "Network Difficulty": ("difficulty", "number"),
        "Miner Revenue": ("coinbase_usd_sum", "currency"),
        "Fee % Of Reward": ("pct_fee_of_reward", "percent"),
    },
    "Network Economics": {
        "Bitcoin Supply": ("supply_btc", "number"),
        "% Supply Issued": ("pct_supply_issued", "percent"),
        "Bitcoin Mined Per Day": ("subsidy_btc_sum", "number"),
        "Annual Inflation Rate": ("inflation_rate", "percent"),
        "Velocity": ("usd_velocity", "number2"),
    },
    "Network Valuation": {
        "Market Cap": ("market_cap", "currency"),
        "Bitcoin Price": ("price_close", "currency"),
        "Realised Price": ("realised_price", "currency"),
        "Thermocap Price": ("thermocap_price", "currency"),
    },
}


# =============================================================================
# BRK API CONFIGURATION
# =============================================================================

BRK_BULK_URL = "https://bitview.space/api/metrics/bulk"

BRK_METRICS = [
    "timestamp",
    "price_close",
    "market_cap",
    "block_count",
    "difficulty",
    "difficulty_adjustment",
    "hash_rate",
    "realized_price",
    "realized_cap",
    "sth_realized_price",
    "sth_realized_cap",
    "lth_realized_price",
    "lth_realized_cap",
    "coindays_destroyed",
    "utxo_count",
    "supply_btc",
    "supply_usd",
    "sth_supply",
    "lth_supply",
    "fee_usd_sum",
    "fee_btc_sum",
    "subsidy_usd_sum",
    "subsidy_btc_sum",
    "coinbase_usd_sum",
    "coinbase_btc_sum",
    "fee_usd_average",
    "fee_btc_average",
    "fee_rate_average",
    "fee_dominance",
    "utxos_over_1y_old_supply_rel_to_circulating_supply",
    "tx_v1",
    "tx_v2",
    "tx_v3",
    "btc_velocity",
    "usd_velocity",
    "sent_usd",
    "inflation_rate",
    "addrs_over_1sat_addr_count",
    "addrs_over_10k_sats_addr_count",
    "address_activity_sending_average",
    "address_activity_receiving_average",
    "addrs_under_1btc_addr_count",
    "addrs_under_10btc_addr_count",
    "addrs_under_10k_sats_addr_count",
    "addrs_under_1k_sats_addr_count",
    "addrs_under_10sats_addr_count",
    "utxos_1h_to_1d_old_supply",
    "utxos_under_1m_old_supply",
    "utxos_under_3m_old_supply",
    "utxos_under_6m_old_supply",
    "utxos_under_1y_old_supply",
    "utxos_under_2y_old_supply",
    "utxos_under_3y_old_supply",
    "utxos_under_4y_old_supply",
    "utxos_under_5y_old_supply",
    "utxos_under_10y_old_supply",
]

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# Bitcoin mining electricity cost model parameters
ELECTRICITY_COST = 0.05  # USD per kWh (global average estimate)
PUE = 1.1  # Power Usage Effectiveness (datacenter overhead factor)
ELEC_TO_TOTAL_COST_RATIO = 0.6  # Electricity as fraction of total mining cost

# Bitcoin unit conversion
SATS_PER_BTC = 100_000_000  # Satoshis per Bitcoin

# Trading days per year by asset class
STOCK_TRADING_DAYS = 252  # Traditional financial markets
CRYPTO_TRADING_DAYS = 365  # Cryptocurrency markets (24/7)


# =============================================================================
# EXTERNAL DATA SOURCES
# =============================================================================

# Google Sheets URL for miner efficiency data
MINER_DATA_SHEET_URL = "https://docs.google.com/spreadsheets/d/1GXaY6XE2mx5jnCu5uJFejwV95a0gYDJYHtDE0lmkGeA/edit?usp=sharing"


# =============================================================================
# API CONFIGURATION
# =============================================================================

# Default timeout for HTTP requests (seconds)
API_TIMEOUT = 30
