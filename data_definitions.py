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
        "AVGO",
        "TSLA",
        "LLY",
        "MU",
        "META",
        "BRK-A",
        "BRK-B",
        "TSM",
        "SPCX",
        "2222.SR",
        "005930.KS",
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
    "indices": [
        "^GSPC",
        "^VIX",
        "^IXIC",
        "^TNX",
        "^TYX",
        "^FVX",
        "^IRX",
        "^SPGSCI",
    ],
    "commodities": ["GC=F", "CL=F", "SI=F"],
    "forex": [
        "DX-Y.NYB",
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

# Yahoo's historical shares-outstanding feed is useful and reasonably complete from 2015
# onward. Keep the broader price history above, but do not invent stock market caps before
# Yahoo supplies a historical share count.
market_cap_history_start_date = "2015-01-01"

# Yahoo keys historical share counts to the ticker that was active at the time. Prices for
# the current symbols already span these renames, so only the shares feed needs stitching.
yahoo_share_ticker_aliases = {
    "META": ["FB", "META"],
    "XYZ": ["SQ", "XYZ"],
}

# Yahoo reports historical Close and shares in each listing's trading currency. Convert
# non-USD listings before publishing the project's ``*_MarketCap`` columns, whose contract
# is absolute USD. TSM is a USD-traded ADR and therefore needs no conversion here.
yahoo_market_cap_fx_tickers = {
    "2222.SR": "SARUSD=X",
    "005930.KS": "KRWUSD=X",
}

# First Bitcoin halving date - used as start date for statistics calculations
stats_start_date = "2012-11-28"

# The report represents the last completed UTC day. GitHub-hosted runners currently use
# UTC, but making the clock explicit keeps local and CI runs identical across timezones.
report_date = (
    pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    - pd.Timedelta(days=1)
)


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

# Fixed price outlook levels used by the dashboard and weekly report.
# `color` is the single source of truth for case styling — the dashboard reads it for
# both the headline cards and the chart's reference lines, so they cannot drift apart.
# Values are the brand cypherpunk red/gold/green.
price_outlook_levels = pd.DataFrame(
    [
        {"label": "Bull Case", "price": 160000, "type": "case", "color": "#00FF88"},
        {"label": "Base Case", "price": 120000, "type": "case", "color": "#FFD700"},
        {"label": "Bear Case", "price": 70000, "type": "case", "color": "#FF3B30"},
        {
            "label": "Resistance $126,219 - 2025 ATH",
            "price": 126219,
            "type": "resistance",
            "color": "#9ca3af",
        },
        {
            "label": "Resistance $108,287 - 2024 ATH",
            "price": 108287,
            "type": "resistance",
            "color": "#9ca3af",
        },
        {
            "label": "Resistance $100,000 - Psychological Level",
            "price": 100000,
            "type": "resistance",
            "color": "#9ca3af",
        },
        {
            "label": "Resistance $80,600 - Nov 2025 Low",
            "price": 80600,
            "type": "resistance",
            "color": "#9ca3af",
        },
        {
            "label": "Support $73,757 - 2024 Prior ATH",
            "price": 73757,
            "type": "support",
            "color": "#9ca3af",
        },
        {
            "label": "Support $60,132 - 2026 Low",
            "price": 60132,
            "type": "support",
            "color": "#9ca3af",
        },
    ]
)


# =============================================================================
# REPORT CONFIGURATION
# =============================================================================

# Columns for which CAGR is actually needed downstream.
# Chart Library uses the _close price CAGRs; report_tables uses the valuation model CAGRs.
# Limiting to these 13 columns instead of all 400+ cuts CAGR compute time by ~97%.
cagr_columns = [
    # Close prices — used in chart CAGR comparison and cagr_data.csv export
    "price_close",
    "SPY_close",
    "QQQ_close",
    "XLK_close",
    "XLF_close",
    "GLD_close",
    "AGG_close",
    "DX-Y.NYB_close",
    "WGMI_close",
    # Valuation models — used in EOY price model table (report_tables.create_eoy_model_table)
    "realized_price",
    "thermocap_price",
    "200_day_ma_price_close",
    "Lagged_Energy_Value",
]

# Metrics for which to calculate 7/30/365-day moving averages
moving_avg_metrics = [
    "hash_rate",
    "daily_active_addresses_sending",
    "tx_count_sum_24h",
    "transfer_volume_sum_24h_usd",
    "fees_average_24h_usd",
    "fees_average_24h",
    "subsidy_sum_24h",
    "coinbase_sum_24h_usd",
    "nvt_price",
    "nvt_price_adj",
]

# Columns that need change calculations (7d, 90d, MTD, YTD)
# These are the only columns passed to run_data_analysis()
analysis_columns = [
    # Bitcoin price and on-chain metrics
    "price_close",
    "hash_rate",
    "tx_count_sum_24h",
    "transfer_volume_sum_24h_usd",
    "daily_active_addresses_sending",
    "addrs_over_10k_sats_addr_count",
    "addrs_over_1btc_addr_count",
    "coinbase_sum_24h_usd",
    "fees_sum_24h_usd",
    "supply_pct_1_year_plus",
    "velocity_usd",
    # BDD/VOCD/Reserve Risk metrics
    "coindays_destroyed_sum_24h",
    "adjusted_bdd",
    "vocd",
    "mvocd",
    "hodl_bank_calc",
    "reserve_risk_calc",
    # Volatility metrics
    "VtyDayRet30d",
    "VtyDayRet180d",
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
    "DX-Y.NYB_close",
    "GLD_close",
    "AGG_close",
    "^SPGSCI_close",
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
    "AVGO_close",
    "TSLA_close",
    "LLY_close",
    "MU_close",
    "META_close",
    "BRK-A_close",
    "BRK-B_close",
    "TSM_close",
    "SPCX_close",
    "2222.SR_close",
    "005930.KS_close",
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
    "DX-Y.NYB_close",
    "AUDUSD=X_close",
    "^SPGSCI_close",
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
        "Supply Held 1+ Year %": ("supply_pct_1_year_plus", "percent_point"),
        "Transaction Count": ("tx_count_sum_24h", "number"),
        "Transaction Volume": ("transfer_volume_sum_24h_usd", "currency"),
        "Transaction Fee USD": ("fees_sum_24h_usd", "currency"),
    },
    "Network Security": {
        "Hash Rate": ("hash_rate", "hashrate_ehs"),
        "Network Difficulty": ("difficulty", "difficulty_t"),
        "Miner Revenue": ("coinbase_sum_24h_usd", "currency"),
        "Fee % Of Reward": ("pct_fee_of_reward", "percent_point"),
    },
    "Network Economics": {
        "Bitcoin Supply": ("supply", "number"),
        "% Supply Issued": ("pct_supply_issued", "percent_ratio"),
        "Bitcoin Mined Per Day": ("subsidy_sum_24h", "number"),
        "Annual Inflation Rate": ("inflation_rate", "percent_point"),
        "Velocity": ("velocity_usd", "number2"),
    },
    "Network Valuation": {
        "Market Cap": ("market_cap", "currency"),
        "Bitcoin Price": ("price_close", "currency"),
        "Realized Price": ("realized_price", "currency"),
        "Thermocap Price": ("thermocap_price", "currency"),
    },
}


# =============================================================================
# BRK API CONFIGURATION
# =============================================================================

# BRK v0.2+ uses the canonical /api/series/bulk endpoint.
# The legacy /api/metrics/bulk route still exists, but is deprecated.
BRK_BULK_URL = "https://bitview.space/api/series/bulk"

BRK_METRICS = [
    "timestamp",
    "price_close",
    "market_cap",
    "difficulty",
    "difficulty_adjustment",
    "hash_rate",
    "realized_price",
    "realized_cap",
    "sth_realized_price",
    "sth_realized_cap",
    "lth_realized_price",
    "lth_realized_cap",
    "coindays_destroyed_sum_24h",
    "utxo_count",
    "supply",
    "supply_usd",
    "sth_supply",
    "lth_supply",
    "fees_sum_24h_usd",
    "fees_sum_24h",
    "subsidy_sum_24h_usd",
    "subsidy_sum_24h",
    "coinbase_sum_24h_usd",
    "coinbase_sum_24h",
    "fees_average_24h_usd",
    "fees_average_24h",
    "effective_fee_rate_median",
    "fee_dominance",
    "utxos_over_1y_old_supply",
    "tx_count_sum_24h",
    "velocity_btc",
    "velocity_usd",
    "transfer_volume_sum_24h_usd",
    "inflation_rate",
    # Valuation and profitability metrics
    "nvt",
    "puell_multiple",
    "liveliness",
    "realized_profit_sum_24h",
    "realized_loss_sum_24h",
    "net_realized_pnl_sum_24h",
    "supply_in_profit",
    "supply_in_loss",
    "sopr_24h",
    # Active supply and hash price metrics
    "active_supply",
    "active_supply_sats",
    "active_supply_usd",
    "hash_price_ths",
    "hash_price_phs",
    # Total non-zero address count used by the headline Metcalfe model.
    "addr_count",
    # Address counts by threshold (cumulative)
    "addrs_over_1sat_addr_count",
    "addrs_over_10sats_addr_count",
    "addrs_over_100sats_addr_count",
    "addrs_over_1k_sats_addr_count",
    "addrs_over_10k_sats_addr_count",
    "addrs_over_100k_sats_addr_count",
    "addrs_over_1m_sats_addr_count",
    "addrs_over_10m_sats_addr_count",
    "addrs_over_1btc_addr_count",
    "addrs_over_10btc_addr_count",
    "addrs_over_100btc_addr_count",
    "addrs_over_1k_btc_addr_count",
    "addrs_over_10k_btc_addr_count",
    "addrs_over_100k_btc_addr_count",
    # Address activity metrics (24h rolling average of unique active addresses)
    "active_addrs_average_24h",
    # Legacy sats-based address counts (used by report tables)
    "addrs_under_1btc_addr_count",
    "addrs_under_10btc_addr_count",
    "addrs_under_10k_sats_addr_count",
    "addrs_under_1k_sats_addr_count",
    "addrs_under_10sats_addr_count",
    # UTXO age band supply
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

# Strategy-aligned network model anchors. Metcalfe scale and the power-law
# scale/exponent are fitted through the report date; these values define only
# the fixed inputs and equation structure.
BITCOIN_GENESIS_DATE = pd.Timestamp("2009-01-03")
METCALFE_ADDRESS_COLUMNS = {
    "addr_count": "any_balance",
    "addrs_over_100k_sats_addr_count": "0p001_btc",
    "addrs_over_1m_sats_addr_count": "0p01_btc",
    "addrs_over_10m_sats_addr_count": "0p1_btc",
}
HASH_RIBBON_FAST_WINDOW = 30
HASH_RIBBON_SLOW_WINDOW = 60

# Bitcoin mining electricity-cost assumptions. Power expense is published across
# a tariff range because miners do not pay one representative global rate.
ELECTRICITY_TARIFFS_USD_PER_KWH = (0.03, 0.04, 0.05, 0.06, 0.07)
ELECTRICITY_BASE_TARIFF_USD_PER_KWH = 0.05
# Backward-compatible constant for external imports. New calculations should use
# the explicitly named base tariff above.
ELECTRICITY_COST = ELECTRICITY_BASE_TARIFF_USD_PER_KWH
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
