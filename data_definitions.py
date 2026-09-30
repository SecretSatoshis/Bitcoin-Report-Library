"""Pipeline configuration: tickers, report date, hand-maintained reference data, the
columns each calculation covers, BRK series, model parameters and sentiment bands."""
import pandas as pd


# =============================================================================
# MARKET DATA CONFIGURATION
# =============================================================================

# Yahoo tickers by group. Market caps are built for `stocks` only.
TICKERS = {
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
    ],
    # Price only (performance table); no market cap is built for these.
    "bitcoin_equities": [
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

STOCK_TICKERS = TICKERS["stocks"]

# Start of the fetched market and on-chain history (candles start at genesis).
MARKET_DATA_START_DATE = "2010-01-01"

# Yahoo's share-count history is reliable from 2015; market caps start there.
MARKET_CAP_HISTORY_START_DATE = "2015-01-01"

# Former tickers whose share counts are stitched onto the current one. Yahoo prices
# already span the rename.
YAHOO_SHARE_TICKER_ALIASES = {
    "META": ["FB", "META"],
}

# FX pairs that convert non-USD listings to USD market caps. TSM is a USD-traded ADR.
YAHOO_MARKET_CAP_FX_TICKERS = {
    "2222.SR": "SARUSD=X",
    "005930.KS": "KRWUSD=X",
}

# The last completed UTC day, whatever the machine's timezone.
REPORT_DATE = (
    pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    - pd.Timedelta(days=1)
)


# =============================================================================
# REFERENCE DATA
# =============================================================================

# When each hand-maintained figure below was last checked. The build fails once one is
# older than REFERENCE_DATA_MAX_AGE_DAYS; bump the date whenever you re-check a figure.
# The 2026-08-22 dates are the public baseline, not the original sourcing dates.
FIAT_MONEY_AS_OF = pd.Timestamp("2026-08-22")
PRECIOUS_METALS_AS_OF = pd.Timestamp("2026-08-22")
GOLD_BREAKDOWN_AS_OF = pd.Timestamp("2026-08-22")
POWER_LAW_BANDS_AS_OF = pd.Timestamp("2026-09-29")

# Gold supply grows ~1.7% a year and M0 faster, so a year-old figure is materially off.
REFERENCE_DATA_MAX_AGE_DAYS = 365

REFERENCE_DATA_VINTAGES = {
    "FIAT_MONEY_SUPPLY": FIAT_MONEY_AS_OF,
    "GOLD_SILVER_SUPPLY": PRECIOUS_METALS_AS_OF,
    "GOLD_SUPPLY_BREAKDOWN": GOLD_BREAKDOWN_AS_OF,
    "POWER_LAW_VALUATION_BANDS": POWER_LAW_BANDS_AS_OF,
}


# M0 money supply by country, USD trillions (central bank data).
FIAT_MONEY_SUPPLY = pd.DataFrame(
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

# Above-ground supply in troy ounces (World Gold Council estimates).
GOLD_SILVER_SUPPLY = pd.DataFrame(
    {
        "Metal": ["Gold", "Silver"],
        "Supply Troy Ounces": [6_100_000_000, 30_900_000_000],
    }
)

# Share of above-ground gold by use (World Gold Council).
GOLD_SUPPLY_BREAKDOWN = pd.DataFrame(
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

# The year the case levels below forecast, set by the annual Year Ahead Outlook. The
# build fails once the report date moves past it.
PRICE_OUTLOOK_YEAR = 2026

# Price outlook levels for the dashboard and weekly report. The dashboard styles its
# cards and chart lines from `color`.
PRICE_OUTLOOK_LEVELS = pd.DataFrame(
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

# Columns that get a rolling 4-year CAGR (Chart Library's CAGR charts).
CAGR_COLUMNS = [
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

# Metrics that get 30-day and 365-day moving averages (Chart Library lines).
MOVING_AVERAGE_METRICS = [
    "hash_rate",
    "daily_active_addresses_sending",
    "tx_count_sum_24h",
    "transfer_volume_sum_24h_usd",
    "subsidy_sum_24h",
    "coinbase_sum_24h_usd",
    "nvt_price",
]

# Price columns that get 7-day, 90-day, MTD and YTD changes (performance table, Chart
# Library return comparisons, quarterly report).
CHANGE_COLUMNS = [
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

# Columns that get a year-over-year change (Chart Library's YoY chart).
YOY_COLUMNS = ["price_close"]

# Columns correlated with Bitcoin over 7, 30, 90 and 365 days.
CORRELATION_COLUMNS = [
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

# Fundamentals table rows: {section: {label: (column, format_type)}}.
FUNDAMENTALS_TEMPLATE = {
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
    "hash_price_ths",
    # Non-zero address count (Metcalfe model)
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
    # Unique active addresses per day
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

# BRK series that need a Bitcoin price. BRK reports them as 0 before the first traded
# price (2010-08-16); those rows are blanked. A realized price of 0 means an empty cohort
# and is blanked wherever it occurs.
BRK_PRICE_DEPENDENT_METRICS = [
    "price_close",
    "market_cap",
    "realized_price",
    "realized_cap",
    "sth_realized_price",
    "lth_realized_price",
    "fees_sum_24h_usd",
    "coinbase_sum_24h_usd",
    "velocity_usd",
    "transfer_volume_sum_24h_usd",
    "nvt",
    "puell_multiple",
    "realized_profit_sum_24h",
    "realized_loss_sum_24h",
    "net_realized_pnl_sum_24h",
    "supply_in_profit",
    "supply_in_loss",
    "sopr_24h",
    "hash_price_ths",
]
BRK_REALIZED_PRICE_METRICS = ["realized_price", "sth_realized_price", "lth_realized_price"]

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# Network model inputs. The Metcalfe and power-law coefficients are fitted through the
# report date and published in the release manifest; these are the fixed parts.
BITCOIN_GENESIS_DATE = pd.Timestamp("2009-01-03")
METCALFE_ADDRESS_COLUMN = "addr_count"
HASH_RIBBON_FAST_WINDOW = 30
HASH_RIBBON_SLOW_WINDOW = 60

# Electricity tariffs (USD/kWh) for the mining cost models. A range is published because
# miners do not pay one global rate.
ELECTRICITY_TARIFFS_USD_PER_KWH = (0.03, 0.04, 0.05, 0.06, 0.07)
ELECTRICITY_BASE_TARIFF_USD_PER_KWH = 0.05

SATS_PER_BTC = 100_000_000

# =============================================================================
# EXTERNAL DATA SOURCES
# =============================================================================

# Coin Metrics monthly miner efficiency
MINER_DATA_SHEET_URL = "https://docs.google.com/spreadsheets/d/1GXaY6XE2mx5jnCu5uJFejwV95a0gYDJYHtDE0lmkGeA/edit?usp=sharing"


# =============================================================================
# API CONFIGURATION
# =============================================================================

# HTTP request timeout, seconds
API_TIMEOUT = 30


# =============================================================================
# INVESTOR SENTIMENT
# =============================================================================

# Market sentiment label: the NUPL emotion-cycle zone of the 7-day average NUPL, averaged
# so one day's move cannot flip it. Entries are (upper bound, label).
NUPL_SENTIMENT_ZONES = [
    (0.0, "Capitulation"),
    (0.25, "Hope / Fear"),
    (0.5, "Optimism / Anxiety"),
    (0.75, "Belief / Denial"),
    (float("inf"), "Euphoria / Greed"),
]
NUPL_SENTIMENT_WINDOW_DAYS = 7

# Valuation label from `power_law_price_multiple`, banded at -1, 0, +1 and +2 standard
# deviations of the multiple's log10 spread since 2015 (0.2388 at the last review). The
# thresholds are fixed; re-check them yearly and bump POWER_LAW_BANDS_AS_OF.
POWER_LAW_VALUATION_BANDS = [
    (0.58, "Undervalued"),
    (1.00, "Below Fair Value"),
    (1.73, "Above Fair Value"),
    (3.00, "Overvalued"),
    (float("inf"), "Extremely Overvalued"),
]
