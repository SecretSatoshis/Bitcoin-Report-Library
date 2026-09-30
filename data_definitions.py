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

# Asset tickers organized by category for yfinance API calls
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
        "BRK-B",
        "TSM",
        "SPCX",
        "2222.SR",
        "005930.KS",
        "V",
        "JPM",
        "COIN",
        "XYZ",
        "MSTR",
    ],
    "etfs": [
        "BITQ",
        "XLK",
        "QQQ",
        "VTI",
        "TLT",
        "GLD",
        "XLF",
        "XLRE",
        "XLE",
        "SPY",
        "IEMG",
        "AGG",
        "WGMI",
        "VXUS",
    ],
    "indices": [
        "^TNX",
        "^TYX",
        "^FVX",
        "^IRX",
        "^SPGSCI",
    ],
    "commodities": ["GC=F", "CL=F", "SI=F"],
    "forex": [
        "DX-Y.NYB",
    ],
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

# The report represents the last completed UTC day. GitHub-hosted runners currently use
# UTC, but making the clock explicit keeps local and CI runs identical across timezones.
report_date = (
    pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    - pd.Timedelta(days=1)
)


# =============================================================================
# REFERENCE DATA
# =============================================================================

# Vintages for the hand-maintained reference figures below. Every other input in this
# pipeline carries a source observation date and an age budget; these are scalars typed in
# by hand, broadcast across the whole daily history, and published as `{Country}_btc_price`
# and the gold market-cap series — so they need the same treatment.
#
# NOTE: the repository history was squashed at the public baseline (2026-08-22), so these
# dates record when the figures were last confirmed present, not when they were sourced.
# Bump each one to the day you actually re-check the underlying figure.
FIAT_MONEY_AS_OF = pd.Timestamp("2026-08-22")
PRECIOUS_METALS_AS_OF = pd.Timestamp("2026-08-22")
GOLD_BREAKDOWN_AS_OF = pd.Timestamp("2026-08-22")

# How far behind the report date a reference figure may fall before the build fails.
# Above-ground gold grows ~1.7%/yr and global M0 moves considerably faster, so a figure
# more than a year old is materially wrong, not merely dusty.
REFERENCE_DATA_MAX_AGE_DAYS = 365

REFERENCE_DATA_VINTAGES = {
    "fiat_money_data_top10": FIAT_MONEY_AS_OF,
    "gold_silver_supply": PRECIOUS_METALS_AS_OF,
    "gold_supply_breakdown": GOLD_BREAKDOWN_AS_OF,
}


# Global fiat money supply (M0) by country in USD trillions
# Source: Central bank data. Vintage: FIAT_MONEY_AS_OF.
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
        ],
    }
)

# Above-ground precious metals supply in troy ounces
# Gold: ~6.1B oz, Silver: ~30.9B oz (World Gold Council estimates)
# Vintage: PRECIOUS_METALS_AS_OF.
gold_silver_supply = pd.DataFrame(
    {
        "Metal": ["Gold", "Silver"],
        "Supply in Billion Troy Ounces": [6100000000, 30900000000],
    }
)

# Gold market allocation by use case (World Gold Council). Vintage: GOLD_BREAKDOWN_AS_OF.
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

# The calendar year the case levels below forecast. Published once a year in the Year
# Ahead Outlook. `assert_price_outlook_current` fails the build when this falls behind
# the report date, so a stale forecast cannot be presented as the current one — the
# homepage tracker and the dashboard both label these levels with this year.
PRICE_OUTLOOK_YEAR = 2026

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

# Columns that get a rolling 4-year CAGR. Chart Library's CAGR charts read these
# from the master file; nothing reads any other CAGR.
cagr_columns = [
    "price_close",
    "SPY_close",
    "QQQ_close",
    "XLK_close",
    "XLF_close",
    "GLD_close",
    "AGG_close",
    "DX-Y.NYB_close",
    "WGMI_close",
]

# Metrics that get 30-day and 365-day moving averages (Chart Library lines)
moving_avg_metrics = [
    "hash_rate",
    "daily_active_addresses_sending",
    "tx_count_sum_24h",
    "transfer_volume_sum_24h_usd",
    "subsidy_sum_24h",
    "coinbase_sum_24h_usd",
    "nvt_price",
]

# Price columns that get 7-day, 90-day, MTD and YTD changes. These feed the
# performance tables, Chart Library's return comparisons and the quarterly report.
analysis_columns = [
    "price_close",
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

# Only Bitcoin's year-over-year change is read (Chart Library's YoY chart).
yoy_columns = ["price_close"]

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
    "BRK-B_close",
    "TSM_close",
    "SPCX_close",
    "2222.SR_close",
    "005930.KS_close",
    "V_close",
    "JPM_close",
    "BITQ_close",
    "XLK_close",
    "QQQ_close",
    "VTI_close",
    "TLT_close",
    "GLD_close",
    "XLF_close",
    "XLRE_close",
    "XLE_close",
    "SPY_close",
    "IEMG_close",
    "AGG_close",
    "WGMI_close",
    "VXUS_close",
    "^TNX_close",
    "^TYX_close",
    "^FVX_close",
    "^IRX_close",
    "GC=F_close",
    "CL=F_close",
    "SI=F_close",
    "DX-Y.NYB_close",
    "^SPGSCI_close",
    "COIN_close",
    "XYZ_close",
    "MSTR_close",
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
    "lth_realized_price",
    "coindays_destroyed_sum_24h",
    "supply",
    "sth_supply",
    "lth_supply",
    "fees_sum_24h_usd",
    "fees_sum_24h",
    "subsidy_sum_24h",
    "coinbase_sum_24h_usd",
    "coinbase_sum_24h",
    "utxos_over_1y_old_supply",
    "tx_count_sum_24h",
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
    # Hash price
    "hash_price_ths",
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
    # Address activity metrics (24h rolling average of unique active addresses)
    "active_addrs_average_24h",
    # UTXO age band supply
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
}
HASH_RIBBON_FAST_WINDOW = 30
HASH_RIBBON_SLOW_WINDOW = 60

# Bitcoin mining electricity-cost assumptions. Power expense is published across
# a tariff range because miners do not pay one representative global rate.
ELECTRICITY_TARIFFS_USD_PER_KWH = (0.03, 0.04, 0.05, 0.06, 0.07)
ELECTRICITY_BASE_TARIFF_USD_PER_KWH = 0.05

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


# =============================================================================
# INVESTOR SENTIMENT
# =============================================================================

# Fear & Greed is the NUPL zone (net unrealized profit/loss, calculated from BRK market cap
# and realized cap) using the widely published emotion-cycle bands. The label comes from the
# 7-day average so it does not flicker at a boundary on a single day's move. Each entry is
# (upper bound, label); the last applies above.
NUPL_SENTIMENT_ZONES = [
    (0.0, "Capitulation"),
    (0.25, "Hope / Fear"),
    (0.5, "Optimism / Anxiety"),
    (0.75, "Belief / Denial"),
    (float("inf"), "Euphoria / Greed"),
]
NUPL_SENTIMENT_WINDOW_DAYS = 7

# Valuation is price against the power-law fair value (`power_law_price_multiple`), banded
# at standard deviations around the fair-value line: -1, 0, +1 and +2 sigma, where sigma is
# the log10 spread of the multiple since 2015 (0.2388 as reviewed on 2026-09-29). The
# thresholds are fixed so labels never drift on their own; review them once a year.
POWER_LAW_VALUATION_REVIEWED = "2026-09-29"
POWER_LAW_VALUATION_BANDS = [
    (0.58, "Undervalued"),
    (1.00, "Below Fair Value"),
    (1.73, "Above Fair Value"),
    (3.00, "Overvalued"),
    (float("inf"), "Extremely Overvalued"),
]
