import datapane as dp
import pandas as pd


def generate_report_layout_weekly_bitcoin_recap(
    weekly_summary_big_numbers,
    ohlc_plot,
    equity_performance_table,
    sector_performance_table,
    macro_performance_table,
    bitcoin_performance_table,
    plotly_heatmap_chart,
    trading_range_chart,
    roi_table,
    fundamentals_table,
    fundamentals_weekly_table,
    report_data,
    report_date,
):
    # Extract the report date from report_data
    report_date_data = report_data.loc[report_date].name
    formatted_date = pd.to_datetime(report_date_data).strftime("%B %d, %Y")

    # Weekly Market Summary Intro
    welcome_text = dp.Text(f"""
  ## Secret Satoshis | Weekly Bitcoin Recap
  Data As Of: {formatted_date}   

  Data Sources: Coinmetrics, Yahoo Finance, Coingecko, Kraken, Alternative.me
  
  Welcome to the Secret Satoshis Weekly Bitcoin Recap an exclusive resource for Secret Satoshis newsletter subscribers, providing a continuously updated report to analyze Bitcoin’s market performance. Designed to empower informed decisions, this report offers daily insights into Bitcoin data, market performance, and more, structured for easy navigation and deep analysis of the evolving Bitcoin market trends.
  
  ## Report Navigation:
  
  The report is organized into two main sections to provide you with the essential tools for analyzing the Bitcoin market effectively:
  
  1. **Bitcoin Market Data & Insights**: This main report delivers a comprehensive view of Bitcoin’s current market performance through interactive graphs, tables, and dashboards.
  
  2. **Report Definitions/Glossary**: This section serves as a resource for understanding the data, providing detailed definitions and explanations of the metrics, tables, and concepts featured in the main report.
  
  To navigate between different sections, click on the respective tab.
""")

    # Weekly Market Summary Header
    weekly_header = dp.Group(
        blocks=[welcome_text],
        columns=1,
    )

    # Weekly OHLC Bitcoin Price Chart Summary Header
    header_weekly_price_section = dp.Text("### Bitcoin Weekly OHLC Price Chart")

    # Performance Report Summary Section Components
    performance_header = dp.Text("# Bitcoin Price Return Relative Performance Tables")
    performance_description = dp.Text("""
    A comparative view of Bitcoin's price performance against key Equity Indexes, Sector ETFs, Macro Assets, and Bitcoin Stocks & ETFs.
      """)

    # Fundamentals Report Summary Section Components
    fundamentals_header = dp.Text("# Historical Fundamentals Table")
    fundamentals_description = dp.Text("""
  A detailed analysis of Bitcoin's on-chain activity, illustrating network security, economic activity, and overall user engagement.
      """)
    fundamentals_footer = dp.Text(
        "For a deeper dive into the fundamental metrics, visit the Report Definitions Tab above."
    )

    # Fundamentals Report Summary Section Components
    fundamentals_weekly_table_header = dp.Text("# Weekly Fundamentals Table")
    fundamentals_weekly_table_description = dp.Text("""
    A detailed analysis of Bitcoin's on-chain activity, illustrating network security, economic activity, and overall user engagement.
      """)
    
    # Define individual headers
    header_trading_range = dp.Text("### Count Of Days In Trading Ranges | 1K ($) Groups")
    header_roi = dp.Text("### Bitcoin Investment ROI Timeframe Comparison")
    # Create individual groups for each table/chart with their headers
    trading_range_block = dp.Group(blocks=[header_trading_range, trading_range_chart])
    roi_block = dp.Group(blocks=[header_roi, roi_table])

    # Second row: Arrange trading range table, ROI table, and MA growth table in three columns
    tables_row = dp.Group(blocks=[trading_range_block, roi_block], columns=2)
    # Monthly Heatmap Summary Header
    header_heatmap_section = dp.Text("### Monthly Bitcoin Price Return Heatmap")

    # Define individual headers for each table
    header_equity = dp.Text("### Stock Market Index Performance")
    header_sector = dp.Text("### Stock Market Sector ETF Performance")
    header_macro = dp.Text("### Macro Asset Performance")
    header_bitcoin = dp.Text("### Bitcoin Industry Stock & ETF Performance")

    # Create individual groups for each table with its corresponding header
    equity_performance_block = dp.Group(
        blocks=[header_equity, equity_performance_table]
    )
    sector_performance_block = dp.Group(
        blocks=[header_sector, sector_performance_table]
    )
    macro_block = dp.Group(blocks=[header_macro, macro_performance_table])
    bitcoin_block = dp.Group(blocks=[header_bitcoin, bitcoin_performance_table])

    # Combine these groups into a single Group block with two columns and two rows
    performance_tables_section = dp.Group(
        blocks=[
            equity_performance_block,
            sector_performance_block,
            macro_block,
            bitcoin_block,
        ],
        columns=2,
    )

    header_definition = dp.Text("### Report Section Definitions")

    # Weekly Market Newsletter Promo
    promo_header = dp.Text(
        "## Continue Your Bitcoin Journey With The Secret Satoshis Newsletter: <a href='https://www.newsletter.secretsatoshis.com/' target='_blank'>Subscribe Now</a>"
    )
    promo_description = dp.Text("""
    Subscribe to the Secret Satoshis Newsletter for exclusive, weekly Bitcoin market updates and data-driven analysis powered by Agent 21. Stay informed with concise insights that help you navigate the Bitcoin landscape with confidence.

    Subscribe Now to Stay Ahead: <a href="https://www.newsletter.secretsatoshis.com/" target="_blank">Subscribe Now</a>.""")

    # Difficulty Report Summary
    weekly_bitcoin_recap_layout = dp.Group(
        weekly_summary_big_numbers,
        header_weekly_price_section,
        ohlc_plot,
        performance_header,
        performance_description,
        performance_tables_section,
        header_heatmap_section,
        plotly_heatmap_chart,
        tables_row,
        fundamentals_header,
        fundamentals_description,
        fundamentals_table,
        fundamentals_footer,
        fundamentals_weekly_table_header,
        fundamentals_weekly_table_description,
        fundamentals_weekly_table,
        promo_header,
        promo_description,
        columns=1,
    )

    # Weekly Summary Table
    weekly_bitcoin_recap_table_content = """
    ## Weekly Summary Table
    **Purpose:** Provides a comprehensive overview of the current Bitcoin market situation, focusing on key metrics.

    **Key Metrics:**
    - **Bitcoin Price USD:** The current market price of Bitcoin in US dollars.
    - **Bitcoin Marketcap:** The total market capitalization of Bitcoin.
    - **Sats Per Dollar:** The number of satoshis (smallest unit of Bitcoin) per US dollar.
    - **Bitcoin Dominance:** Bitcoin's market cap dominance compared to all other crypto assets.
    - **Bitcoin Miner Revenue (30-Day Daily Average):** The average daily revenue generated by Bitcoin miners over the past 30 days.
    - **Bitcoin Transaction Volume (30-Day Daily Average):** The average daily transaction volume on the Bitcoin network over the past 30 days.
    - **Bitcoin Market Sentiment:** The current market sentiment towards Bitcoin, as indicated by the Fear & Greed Index.
    - **Bitcoin Valuation:** The current valuation status of Bitcoin (Overvalued, Undervalued, Fair Value).

    **Insights:** This table provides a quick and comprehensive snapshot of Bitcoin's market status. It helps readers understand the asset's valuation, market dominance, and trading trends, which are critical for making informed investment decisions.
    """
    summary_table = dp.Text(weekly_bitcoin_recap_table_content)

    # OHLC Chart Overview
    ohlc_chart_overview = dp.Text("""
  ## Weekly Open-High-Low-Close (OHLC) Price Chart
  **Purpose:** Presents an OHLC chart of Bitcoin prices, offering a detailed view of its price movements within specific time frames.
  
  **Insights:** Essential for technical analysis, providing insights into market sentiment and potential price directions. Helps in identifying patterns like bullish or bearish trends, breakouts, or reversals.
  """)

    # Combine the sections for the first part of the report
    first_section = dp.Group(summary_table, ohlc_chart_overview)

    # Equity Index Performance Table
    equity_index_performance_overview = dp.Text("""
    ## Stock Market Index Performance Table
    **Purpose:** Provides a comparative view of Bitcoin's performance against major stock indices like Nasdaq, and S&P500.

    **Insights:** By comparing Bitcoin with major indices, this table offers an understanding of how Bitcoin performs in relation to traditional financial markets.
    """)

    # Sector ETF Performance Table
    sector_etf_performance_overview = dp.Text("""
    ## Stock Market Sector ETF Performance Table
    **Purpose:** Analyzes the performance of Bitcoin alongside sector-specific ETFs such as technology, financials, energy, and real estate.

    **Insights:** This table highlights sector trends and correlations, helping to understand how different industries are performing relative to Bitcoin.
    """)

    # Macro Asset Performance Table
    macro_asset_performance_overview = dp.Text("""
    ## Macro Asset Performance Table
    **Purpose:** Evaluates Bitcoin's performance relative to macroeconomic indicators such as the US Dollar Index, Gold, Bonds, and the Commodity Index.

    **Insights:** This table provides a macroeconomic perspective on Bitcoin's performance within the global financial landscape.
    """)

    # Bitcoin-Related Stocks & ETFs Performance Table
    bitcoin_related_equities_performance_overview = dp.Text("""
    ## Bitcoin Industry Stock & ETF Performance Table
    **Purpose:** Focuses on equities and ETFs directly related to Bitcoin and the cryptocurrency market, such as MicroStrategy (MSTR), Coinbase (COIN), and Bitcoin Miners ETF (WGMI).

    **Insights:** This table illustrates the performance of Bitcoin-related stocks, offering insights into investor sentiment and sector-specific movements.
    """)

    # Combine the sections for the second part of the report
    second_section = dp.Group(
        equity_index_performance_overview,
        sector_etf_performance_overview,
        macro_asset_performance_overview,
        bitcoin_related_equities_performance_overview,
    )

    # Monthly Heatmap of Returns
    monthly_heatmap_text = """
    ## Monthly Bitcoin Return Heatmap
    **Purpose:** Presents monthly and yearly Bitcoin returns in a heatmap format, providing a quick visual overview of performance over time.
    
    **Insights:** Allows for easy identification of periods with high returns or significant losses. Can be used to spot seasonal patterns or annual trends in Bitcoin's market performance.
    """
    
     # Price Buckets Analysis
    price_buckets_analysis_text = """
    ## Price Buckets Analysis
    **Purpose:** Categorizes Bitcoin prices into defined buckets, providing a view of how many days Bitcoin traded within specific price ranges.
    
    **Insights:** Offers a historical perspective on the price distribution of Bitcoin. Helps in understanding which price ranges have been most common, potentially indicating key support or resistance levels.
    """
     # Return on Investment (ROI) Table
    roi_table_text = """
    ## Return on Investment (ROI) Table
    **Purpose:** Calculates and presents the ROI for Bitcoin over various time frames, providing a snapshot of its investment performance.
    
    **Insights:** Allows investors to gauge the historical profitability of investing in Bitcoin over different periods. Helps in comparing short-term versus long-term investment returns.
    """
    # Combine the sections for the third part of the report
    historical_trading_section = dp.Group(
        dp.Text(price_buckets_analysis_text),
        dp.Text(roi_table_text),
    )
    fundamentals_summary = dp.Text("""
    #### Fundamentals Table Summary
    The Fundamentals Table offers a concise yet comprehensive glimpse into Bitcoin's operational dynamics through the lens of on-chain data. Leveraging the transparent and immutable nature of the blockchain, this table showcases key metrics such as hashrate and transaction volume, providing investors with a unique opportunity to gauge network health and economic activity. This analysis underscores the revolutionary transparency and verifiability that Bitcoin brings to the financial landscape, empowering investors to craft well-informed strategies grounded in on-chain data.
    """)

    fundamentals_columns = dp.Text("""
    ### Fundamentals Table Column Overview
    **Value:**
    - Description: Indicates the current measurement or numerical value associated with the specified metric, providing a snapshot of the latest data available.

    **1 Day Change:**
    - Description: Represents the change in the metric's value over the past 24 hours.
    - Interpretation & Usage: Allows the tracking of daily fluctuations in various metrics, helping to identify short-term trends and market movements.
    
    **Difficulty Period Change:**
    - Description: Denotes the change in the metric during the most recent Bitcoin difficulty adjustment period, providing a focused perspective on recent trends.
    - Interpretation & Usage: A vital tool to monitor performance over a specific timeframe aligned with Bitcoin's mining dynamics, facilitating informed decision-making based on recent trends.
    
    **MTD Change:**
    - Description: Represents the change in the metric from the beginning of the current month to the present.
    - Interpretation & Usage: This metric provides insight into the metric's short-term performance.
    
    **90 Day Change:**
    - Description:  Illustrates the change in the metric over the past quarter.
    - Interpretation & Usage: This metric provides a gauge of quarterly performance allowing investors to gauge the metric's medium-term trend.
    
    **YTD Change:**
    - Description: Details the metric's performance change since the beginning of the year.
    - Interpretation & Usage: An essential tool for investors to assess annual performance changes and align their investment strategies with yearly benchmarks and goals.

    **4 Year CAGR:**
    - Description: Represents the metric's Compound Annual Growth Rate (CAGR) over a period of four years, providing a smoothed, annualized performance over the time span.
    - Interpretation & Usage: This column helps investors to understand the long-term trend in the metric's growth, aiding in the evaluation of potential long-term investments and strategies.
    
    **52 Week Low:**
    - Description: Marks the lowest recorded value of the metric within the last 52 weeks, showcasing the metric's minimum level over the past year.
    - Interpretation & Usage: This column aids investors in identifying potential support levels and the metric's resilience under adverse market conditions, offering insights into market valuation and stability.
    
    **52 Week High:**
    - Description: Indicates the highest recorded value of the metric within the preceding 52 weeks, highlighting the peak level over the past year.
    - Interpretation & Usage: Offers a glimpse into the metric's maximum value over the past year, illustrating the peak performance of a metric over a year-long span.

      """)
    fundamentals_metrics = dp.Text("""
    ### Fundamentals Table Metrics Overview
    **Hashrate:**
    - Description: Represents the total computational power dedicated to mining and processing Bitcoin transactions.
    - Interpretation & Usage: A higher hashrate indicates a more secure network. Monitoring trends in the hashrate can provide insights into the network's security and potential profitability for miners.

    **Transaction Count:**
    - Description: The total number of Bitcoin transactions processed.
    - Interpretation & Usage: This metric reflects the network's activity level, with a higher transaction count indicating increased usage, potentially signifying greater adoption and utility of the Bitcoin network.
    - Data: 7 Day Average Hashrate 

    **Transaction Volume:**
    - Description: The total amount of Bitcoin transacted in USD terms.
    - Interpretation & Usage: A higher transaction volume can indicate heightened activity on the network, potentially pointing to larger capital flows and increased utility of the network for transferring value.
    - Data: 7 Day Average Adjusted Transaction Volume

    **Average Transaction Size:**
    - Description: Represents the average USD value of individual transactions on the Bitcoin network.
    - Interpretation & Usage: This metric can help gauge the scale of transactions occurring on the network, providing insights into whether the network is being used for larger, potentially more significant transactions or a higher volume of smaller transactions.
    - Data: 7 Day Average Mean Tx Size 

    **Active Address Count:**
    - Description: The number of unique addresses that were active in the network (either as a sender or receiver) during the period.
    - Interpretation & Usage: A higher number of active addresses can indicate increased network usage and adoption, potentially pointing to a more vibrant and active Bitcoin ecosystem.

    **+$10 USD Address Balances:**
    - Description: The total number of Bitcoin addresses holding a balance greater than $10.
    - Interpretation & Usage: This metric can serve as an indicator of retail participation in the Bitcoin network, with a higher count potentially pointing to increased retail interest and activity in the Bitcoin market.
    
    **Miner Revenue:**
    - Description: The total USD value earned by miners through block rewards and transaction fees.
    - Interpretation & Usage: This metric reflects the economic incentives for miners to secure the network. A higher miner revenue can indicate a more profitable and secure network, potentially attracting more miners to participate.
    
    **Fees In USD:**
    - Description: The total USD value of transaction fees paid to miners.
    - Interpretation & Usage: This metric can indicate the demand for block space on the Bitcoin network, with higher fees potentially pointing to increased competition among users to have their transactions included in blocks.
    
    **1+ Year Supply %:**
    - Description:  The percentage of the total Bitcoin supply that has not moved in over one year.
    - Interpretation & Usage: A higher percentage can indicate a greater level of holding among Bitcoin holders, potentially signaling a belief in Bitcoin's future appreciation and a lower likelihood of selling pressure.
    
    **1 Year Velocity:**
    - Description: A measure of how frequently Bitcoin is being transacted, calculated as the ratio of the transaction volume over the past year to the current supply.
    - Interpretation & Usage: This metric can provide insights into the level of activity and liquidity in the Bitcoin market, with higher velocity indicating more active trading and circulation of Bitcoin in the network.
      """)

    # Fundamentals Tab Tables
    fundamentals_tabs = dp.Group(
        fundamentals_summary,
        fundamentals_columns,
        fundamentals_metrics,
    )

    fundamentals_table_header = dp.Text("## Fundamentals Table Summary")

    # fundamentals Table Summary
    fundamentals_definition_layout = dp.Group(
        fundamentals_table_header, fundamentals_tabs, columns=1
    )

    # Combine the sections for the third part of the report
    third_section = dp.Group(
        dp.Text(monthly_heatmap_text),
    )


     # Network Performance Description
    network_performance_text = """
    ## Network Performance
    **Purpose:** Analyzes various metrics that reflect the overall health and activity of the Bitcoin network.
    
    **Key Metrics:**
    - **Total Address Count & Address Count > $10:** These metrics give insights into the number of unique Bitcoin addresses and those holding more than $10, reflecting user adoption and distribution of wealth within the network.
    - **Active Addresses:** Indicates the number of unique addresses active in transactions, serving as a barometer of network engagement.
    - **Supply Held 1+ Year %:** Shows the percentage of Bitcoin supply that hasn't moved in over a year, highlighting investor sentiment and potential long-term holding behavior.
    - **Transaction Count & Transfer Count:** SRepresents the total number of transactions and transfers on the Bitcoin network, indicating network utilization and activity levels.
    - **Transaction Volume:** The total value of all Bitcoin transactions, offering insights into the economic throughput of the network.
    - **Transaction Fee USD:** Reflects the total fees paid for Bitcoin transactions, signifying network demand and miner revenue from fees.
    """

    # Network Security Description
    network_security_text = """
    ## Network Security
    **Purpose:** Provides insights into the security and mining dynamics of the Bitcoin network.
    
    **Key Metrics:**
    - **Hash Rate:** Measures the total computational power used in Bitcoin mining and transaction processing, indicating network security and mining activity.
    - **Network Difficulty:** Represents the complexity of mining a Bitcoin block, adjusting to maintain consistent block times and ensuring network stability.
    - **Miner Revenue:** Total earnings of Bitcoin miners from block rewards and transaction fees, providing insights into the economic incentives and health of the mining sector.
    - **Fee % Of Reward:** The proportion of miner revenue derived from transaction fees as opposed to block rewards, highlighting the economic dynamics of mining.
    """

    # Network Economics Description
    network_economics_text = """
    ## Network Economics
    **Purpose:** Examines the economic aspects of the Bitcoin network, including supply dynamics and inflation rate.
    
    **Key Metrics:**
    - **Bitcoin Supply & Bitcoin Supply In 10 Years:** Tracks the total and projected Bitcoin supply, shedding light on the scarcity dynamics of Bitcoin.
    - **% Supply Issued:** The percentage of the total possible Bitcoin supply that has been mined, indicating the progression towards the 21 million cap.
    - **Bitcoin Mined Per Day:** The daily rate of new Bitcoin creation, reflecting the pace of supply expansion.
    Annual Inflation Rate: The rate at which the Bitcoin supply is increasing, providing insights into its inflationary or deflationary nature.
    - **Velocity:** A measure of how frequently Bitcoin is transacted, offering insights into its use as a medium of exchange versus a store of value.
    """

    # Network Valuation Description
    network_valuation_text = """
    ## Network Valuation
    **Purpose:** Evaluates Bitcoin's market position and valuation through various metrics.
    
    **Key Metrics:**
    - **Market Cap:** The total market value of all mined Bitcoin, reflecting its market significance and investor sentiment.
    - **Bitcoin Price:** The current market price of Bitcoin, directly representing market sentiment and investment value.
    - **Realised Price:** A metric that considers the price at which each Bitcoin last moved, offering a different perspective on market valuation.
    - **Thermocap Price:** A valuation model comparing the market cap with the total miner revenue, providing insights into the market's valuation of Bitcoin's security and miner commitment.
    """

    # Combine the sections for the fourth part of the report
    fourth_section = dp.Group(
        dp.Text(network_performance_text),
        dp.Text(network_security_text),
        dp.Text(network_economics_text),
        dp.Text(network_valuation_text),
    )

    # Definition Tab Tables
    definition_tabs = dp.Select(
        blocks=[
            dp.Group(first_section, label="Weekly Bitcoin Recap"),
            dp.Group(second_section, label="Performance Tables"),
            dp.Group(third_section, label="Monthly Heatmap"),
            dp.Group(historical_trading_section,label="Historical Trading Table"),
            dp.Group(fundamentals_definition_layout, label="Fundamentals Table"),
            dp.Group(fourth_section,label="Fundamentals Weekly Table"),
        ]
    )

    # Definition Summary
    definition_layout = dp.Group(header_definition, definition_tabs, columns=1)
    report_tabs = dp.Select(
        blocks=[
            dp.Group(weekly_bitcoin_recap_layout, label="Weekly Bitcoin Recap"),
            dp.Group(definition_layout, label="Report Definitions / Glossary"),
        ]
    )

    # Combine all parts for final report
    report_blocks = [welcome_text, report_tabs]

    # Return the final layout structure
    return report_blocks
