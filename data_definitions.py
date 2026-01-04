import datetime
import pandas as pd

# TradFi Data
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

# Start date for TradFi Data
market_data_start_date = "2010-01-01"

# First Halving Date Start Stats Calculation
stats_start_date = "2012-11-28"

# Fiat Money Supply M0 Data
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

# Gold And Silver Supply Data
gold_silver_supply = pd.DataFrame(
    {
        "Metal": ["Gold", "Silver"],
        "Supply in Billion Troy Ounces": [6100000000, 30900000000],
    }
)

# Gold Market Supply Breakdown
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

# Just stock tickers for marketcap calculation
stock_tickers = tickers["stocks"]

# Get today's date
today = datetime.date.today()

# Get yesterday's date
yesterday = today - datetime.timedelta(days=1)

# Create report data and convert to pandas.Timestamp
report_date = pd.Timestamp(yesterday)

# On-Chain Metrics to create moving averages of for data smoothing
moving_avg_metrics = [
    "HashRate",
    "AdrActCnt",
    "TxCnt",
    "TxTfrValAdjUSD",
    "TxTfrValMeanUSD",
    "TxTfrValMedUSD",
    "FeeMeanUSD",
    "FeeMeanNtv",
    "IssContNtv",
    "RevUSD",
    "nvt_price",
    "nvt_price_adj",
]

# Define report data and fields
filter_data_columns = {
    "Report_Metrics": [
        "SplyCur",
        "7_day_ma_IssContNtv",
        "365_day_ma_IssContNtv",
        "TxCnt",
        "7_day_ma_TxCnt",
        "365_day_ma_TxCnt",
        "HashRate",
        "7_day_ma_HashRate",
        "365_day_ma_HashRate",
        "PriceUSD",
        "50_day_ma_priceUSD",
        "200_day_ma_priceUSD",
        "200_day_multiple",
        "200_week_ma_priceUSD",
        "TxTfrValAdjUSD",
        "7_day_ma_TxTfrValMeanUSD",
        "7_day_ma_TxTfrValAdjUSD",
        "365_day_ma_TxTfrValAdjUSD",
        "RevUSD",
        "pct_fee_of_reward",
        "IssContPctAnn",
        "IssContNtv",
        "pct_supply_issued",
        "AdrActCnt",
        "7_day_ma_priceUSD",
        "30_day_ma_AdrActCnt",
        "365_day_ma_AdrActCnt",
        "FeeTotUSD",
        "thermocap_price",
        "thermocap_multiple",
        "thermocap_price_multiple_4",
        "thermocap_price_multiple_8",
        "thermocap_price_multiple_16",
        "thermocap_price_multiple_32",
        "nvt_price",
        "nvt_price_adj",
        "nvt_price_multiple",
        "30_day_ma_nvt_price",
        "nvt_price_multiple_ma",
        "365_day_ma_nvt_price",
        "NVTAdj",
        "NVTAdj90",
        "realised_price",
        "VelCur1yr",
        "supply_pct_1_year_plus",
        "mvrv_ratio",
        "realizedcap_multiple_3",
        "realizedcap_multiple_5",
        "nupl",
        "CapMrktCurUSD",
        "AdrBalUSD10Cnt",
        "AdrBalCnt",
        "DiffLast",
        "AAPL_close",
        "^IXIC_close",
        "^GSPC_close",
        "XLF_close",
        "XLRE_close",
        "GC=F_close",
        "SI=F_close",
        "DX=F_close",
        "SHY_close",
        "TLT_close",
        "^TNX_close",
        "^TYX_close",
        "^FVX_close",
        "^IRX_close",
        "XLE_close",
        "BITQ_close",
        "^BCOM_close",
        "FANG.AX_close",
        "SPY_close",
        "IEMG_close",
        "AGG_close",
        "WGMI_close",
        "QQQ_close",
        "VTI_close",
        "GLD_close",
        "XLK_close",
        "VXUS_close",
        "AAPL_mc_btc_price",
        "AAPL_MarketCap",
        "AUDUSD=X_close",
        "CHFUSD=X_close",
        "CNYUSD=X_close",
        "EURUSD=X_close",
        "GBPUSD=X_close",
        "HKDUSD=X_close",
        "INRUSD=X_close",
        "JPYUSD=X_close",
        "RUBUSD=X_close",
        "silver_marketcap_billion_usd",
        "gold_marketcap_billion_usd",
        "30_day_ma_IssContNtv",
        "30_day_ma_TxCnt",
        "30_day_ma_HashRate",
        "30_day_ma_TxTfrValAdjUSD",
        "30_day_ma_RevUSD",
        "365_day_ma_RevUSD",
        "30_day_ma_TxTfrValMeanUSD",
        "30_day_ma_TxTfrValMedUSD",
        "liquid_supply",
        "illiquid_supply",
        "United_Kingdom_btc_price",
        "United_Kingdom_cap",
        "United_States_btc_price",
        "United_States_cap",
        "Global_Fiat_Supply_btc_price",
        "SF_Predicted_Price",
        "SF_Multiple",
        "China_btc_price",
        "Eurozone_btc_price",
        "Japan_btc_price",
        "Switzerland_btc_price",
        "India_btc_price",
        "Australia_btc_price",
        "Russia_btc_price",
        "MSFT_mc_btc_price",
        "GOOGL_mc_btc_price",
        "NVDA_mc_btc_price",
        "AMZN_mc_btc_price",
        "V_mc_btc_price",
        "TSLA_mc_btc_price",
        "JPM_mc_btc_price",
        "PYPL_mc_btc_price",
        "GS_mc_btc_price",
        "META_mc_btc_price",
        "gold_marketcap_btc_price",
        "silver_marketcap_btc_price",
        "gold_jewellery_marketcap_btc_price",
        "gold_private_investment_marketcap_btc_price",
        "gold_official_country_holdings_marketcap_btc_price",
        "gold_other_marketcap_btc_price",
        "sat_per_dollar",
        "Lagged_Energy_Value",
        "Hayes_Network_Price_Per_BTC",
        "Electricity_Cost",
        "Bitcoin_Production_Cost",
        "CM_Energy_Value",
        "Energy_Value_Multiple",
        "SF_Predicted_Price_MA365",
        "ethereum_close",
        "ripple_close",
        "dogecoin_close",
        "binancecoin_close",
        "tether_close",
        "bitcoin_dominance",
        "btc_trading_volume",
        "ethereum_market_cap",
        "ripple_market_cap",
        "dogecoin_market_cap",
        "binancecoin_market_cap",
        "tether_market_cap",
        "COIN_close",
        "XYZ_close",
        "MSTR_close",
        "MARA_close",
        "RIOT_close",
        "CL=F_close",
        "qtm_price",
        "qtm_price_multiple_2",
        "qtm_price_multiple_5",
        "qtm_price_multiple_10",
    ]
}

# Timeframes to calculate volatility 
volatility_windows = [30, 90, 180, 365]

# Data to run correlations on
correlation_data = [
    "PriceUSD",
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

# Weekly Fundamentals Table Metrics
metrics_template = {
    "Network Performance": {
        "Total Address Count": ("AdrBalCnt", "number"),
        "Address Count > $10": ("AdrBalUSD10Cnt", "number"),
        "Active Addresses": ("AdrActCnt", "number"),
        "Supply Held 1+ Year %": ("supply_pct_1_year_plus", "percent"),
        "Transaction Count": ("TxCnt", "number"),
        "Transaction Volume": ("TxTfrValAdjUSD", "currency"),
        "Transaction Fee USD": ("FeeTotUSD", "currency"),
    },
    "Network Security": {
        "Hash Rate": ("HashRate", "number"),
        "Network Difficulty": ("DiffLast", "number"),
        "Miner Revenue": ("RevUSD", "currency"),
        "Fee % Of Reward": ("pct_fee_of_reward", "percent"),
    },
    "Network Economics": {
        "Bitcoin Supply": ("SplyCur", "number"),
        "% Supply Issued": ("pct_supply_issued", "percent"),
        "Bitcoin Mined Per Day": ("IssContNtv", "number"),
        "Annual Inflation Rate": ("IssContPctAnn", "percent"),
        "Velocity": ("VelCur1yr", "number2"),
    },
    "Network Valuation": {
        "Market Cap": ("CapMrktCurUSD", "currency"),
        "Bitcoin Price": ("PriceUSD", "currency"),
        "Realised Price": ("realised_price", "currency"),
        "Thermocap Price": ("thermocap_price", "currency"),
    },
}

# Chart Template OHLC
chart_template = {
    "title": "Bitcoin Weekly Price Chart",
    "x_label": "Date",
    "y1_label": "USD Price",
    "filename": "png/Bitcoin_OHLC",
    "events": [
        {"name": "CME Futures", "dates": ["2017-12-17"], "orientation": "v"},
        {"name": "Bitcoin Winter", "dates": ["2018-12-15"], "orientation": "v"},
        {"name": "Coinbase IPO", "dates": ["2021-04-14"], "orientation": "v"},
        {"name": "FTX Bankrupt", "dates": ["2022-11-11"], "orientation": "v"},
    ],
}


BRK_BULK_URL = "https://eu1.bitview.space/api/metrics/bulk"


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
    "fee_usd_avg",
    "fee_btc_avg",
    "fee_rate_avg",
    "fee_dominance",
    "utxos_at_least_1y_old_supply_rel_to_circulating_supply",
    "tx_v1",
    "tx_v2",
    "tx_v3",
    "tx_btc_velocity",
    "tx_usd_velocity",
    "sent_usd",
    "inflation_rate",
    "addrs_above_1sat_addr_count",
    "addrs_above_10k_sats_addr_count",
    "addrs_above_1sat_sent",
    # Address balance distribution (address counts)
    "addrs_under_1btc_addr_count",
    "addrs_under_10btc_addr_count",
    "addrs_under_10k_sats_addr_count",
    "addrs_under_1k_sats_addr_count",
    "addrs_under_10sats_addr_count",

    # Active supply buckets (UTXO age band supply)
    "utxos_up_to_1d_supply",
    "utxos_up_to_1m_old_supply",
    "utxos_up_to_3m_old_supply",
    "utxos_up_to_6m_old_supply",
    "utxos_up_to_1y_old_supply",
    "utxos_up_to_2y_old_supply",
    "utxos_up_to_3y_old_supply",
    "utxos_up_to_4y_old_supply",
    "utxos_up_to_5y_old_supply",
    "utxos_up_to_10y_old_supply",
]


BRK_RENAME = {
    "price_close": "PriceUSD",
    "market_cap": "CapMrktCurUSD",
    "realized_cap": "CapRealUSD",
    "realized_price": "RealizedPriceUSD",
    "supply_btc": "SplyCur",

    # Difficulty / hash
    "difficulty": "DiffLast",
    "hash_rate": "HashRate",

    # Velocity (keep both)
    "tx_btc_velocity": "VelCur1yr_BTC",
    "tx_usd_velocity": "VelCur1yr",

    # Tx volume (your TxTfrValAdjUSD input)
    "sent_usd": "TxTfrValAdjUSD",

    # Inflation / issuance %
    "inflation_rate": "IssContPctAnn",

    # Fee mean native
    "transactioncoinmining_fee_btc": "FeeMeanNtv",

    # Active/addrs (per your selection)
    "addrs_above_1sat_sent": "AdrActCnt",
    "addrs_above_1sat_addr_count": "AdrBalCnt",
    "addrs_above_10k_sats_addr_count": "AdrBalUSD10Cnt",

   
    "fee_usd_avg": "FeeMeanUSD",
    "fee_btc_avg": "FeeMeanNtv",
    "fee_rate_avg": "FeeRateAvg",
    # Fees (daily totals)
    "fee_usd_sum": "FeeTotUSD",
    "fee_btc_sum": "FeeTotNtv",   # optional but useful

    # Subsidy (daily totals)
    "subsidy_usd_sum": "IssContUSD",
    "subsidy_btc_sum": "IssContNtv",

    # Total miner revenue (fees + subsidy)
    "coinbase_usd_sum": "RevUSD",
    "coinbase_btc_sum": "RevNtv",

    "addrs_under_1btc_addr_count":  "AdrBalUSD1MCnt",
    "addrs_under_10btc_addr_count": "AdrBalUSD10MCnt",
    "addrs_under_10k_sats_addr_count": "AdrBalUSD10KCnt",
    "addrs_under_1k_sats_addr_count":  "AdrBalUSD1KCnt",
    "addrs_under_10sats_addr_count":   "AdrBalUSD1Cnt",

    "utxos_up_to_1d_supply":   "SplyAct1d",
    "utxos_up_to_1m_old_supply": "SplyAct30d",
    "utxos_up_to_3m_old_supply": "SplyAct90d",
    "utxos_up_to_6m_old_supply": "SplyAct180d",
    "utxos_up_to_1y_old_supply": "SplyAct1yr",
    "utxos_up_to_2y_old_supply": "SplyAct2yr",
    "utxos_up_to_3y_old_supply": "SplyAct3yr",
    "utxos_up_to_4y_old_supply": "SplyAct4yr",
    "utxos_up_to_5y_old_supply": "SplyAct5yr",
    "utxos_up_to_10y_old_supply": "SplyAct10yr",

}
