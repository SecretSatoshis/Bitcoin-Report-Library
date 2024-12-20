import datapane as dp


def generate_report_layout_weekly(
    weekly_summary_big_numbers,
    crypto_performance_table,
    index_performance_table,
    macro_performance_table,
    equities_performance_table,
    trading_range_table,
    plotly_dp_chart,
    roi_table,
    table_fig,
    plotly_heatmap_chart,
    plotly_weekly_heatmap_chart,
    yoy_plot,
    ohlc_plot,
):
    # Weekly Market Summary Intro
    welcome_text = dp.Text("""
  ## Secret Satoshis | Weekly Market Update
  
  Welcome to the Secret Satoshis Weekly Market Update an exclusive resource for Secret Satoshis newsletter subscribers, providing a continuously updated report to analyze Bitcoin’s market performance. Designed to empower informed decisions, this report offers daily insights into Bitcoin’s price trends, trading volume, market sentiment, and more, structured for easy navigation and deep analysis of the evolving Bitcoin market trends.
  
  ## Report Navigation:
  
  The report is organized into two main sections to provide you with the essential tools for analyzing the Bitcoin market effectively:
  
  1. **Bitcoin Market Data & Insights**: This main report delivers a comprehensive view of Bitcoin’s current market performance through interactive graphs, tables, and dashboards.
  
  2. **Report Definitions/Glossary**: This section serves as a resource for understanding the data, providing detailed definitions and explanations of the metrics, tables, and concepts featured in the main report.
  
  To navigate between different sections, click on the respective tab.
""")

    # Weekly Market Summary Header
    weekly_header = dp.Text("# Weekly Market Summary")

    # Performance Report Summary Section Components
    performance_header = dp.Text("# Performance Table")
    performance_description = dp.Text("""
    A comparative view of Bitcoin's performance against key assets and indices.
      """)

    # Fundamentals Report Summary Section Components
    fundamentals_header = dp.Text("# Fundamentals Table")
    fundamentals_description = dp.Text("""
    A detailed analysis of Bitcoin's on-chain activity, illustrating network security, economic activity, and overall user engagement.
      """)

    # Trade Report Summary Section Components
    trade_header = dp.Text("# Marekt Analysis")
    trade_description = dp.Text("""
    A comprehensive guide to Bitcoin's price through various analytical lenses, aiding in understanding its current and historical price performance.
      """)

    # Weekly Market Newsletter Promo
    promo_header = dp.Text(
        "## Continue Your Bitcoin Journey With The Secret Satoshis Newsletter: <a href='https://www.newsletter.secretsatoshis.com/' target='_blank'>Subscribe Now</a>"
    )
    promo_description = dp.Text("""
    Subscribe to the Secret Satoshis Newsletter for exclusive, daily Bitcoin market updates and data-driven analysis powered by Agent 21. Stay informed with concise insights that help you navigate the Bitcoin landscape with confidence.

    Subscribe Now to Stay Ahead: <a href="https://www.newsletter.secretsatoshis.com/" target="_blank">Subscribe Now</a>.""")

    # Define individual headers for each table
    header_crypto = dp.Text("### Crypto Performance")
    header_stocks = dp.Text("### Stocks Performance")
    header_indexes = dp.Text("### Indexes Performance")
    header_macro = dp.Text("### Macro Performance")

    # Create individual groups for each table with its corresponding header
    crypto_performance_block = dp.Group(
        blocks=[header_crypto, crypto_performance_table]
    )
    stocks_performance_block = dp.Group(
        blocks=[header_stocks, equities_performance_table]
    )
    indexes_performance_block = dp.Group(
        blocks=[header_indexes, index_performance_table]
    )
    macro_performance_block = dp.Group(blocks=[header_macro, macro_performance_table])

    # Combine these groups into a single Group block with two columns and two rows
    performance_tables_section = dp.Group(
        blocks=[
            crypto_performance_block,
            stocks_performance_block,
            indexes_performance_block,
            macro_performance_block,
        ],
        columns=2,
    )

    # Define individual headers
    header_trading_range = dp.Text("### Bitcoin Price Trading Ranges")
    header_roi = dp.Text("### Bitcoin Investment ROI")
    # Create individual groups for each table/chart with their headers
    trading_range_block = dp.Group(blocks=[header_trading_range, trading_range_table])
    roi_block = dp.Group(blocks=[header_roi, roi_table])
    # First row: Plotly chart on its own row with the trading range header
    plotly_chart_row = dp.Group(
        blocks=[header_trading_range, plotly_dp_chart], columns=1
    )

    # Second row: Arrange trading range table, ROI table, and MA growth table in three columns
    tables_row = dp.Group(blocks=[trading_range_block, roi_block], columns=2)

    header_yoy_section = dp.Text("### Bitcoin Price Performance Year Over Year")
    header_heatmap_section = dp.Text("### Bitcoin Price Performacne Heatmap")
    header_weekly_price_section = dp.Text("### Bitcoin Weekly Price Chart")
    header_definition = dp.Text("### Report Section Definitions")

    # Difficulty Report Summary
    difficulty_summary_layout = dp.Group(
        weekly_header,
        weekly_summary_big_numbers,
        header_weekly_price_section,
        ohlc_plot,
        performance_header,
        performance_description,
        performance_tables_section,
        trade_header,
        trade_description,
        plotly_chart_row,
        tables_row,
        header_yoy_section,
        yoy_plot,
        header_heatmap_section,
        plotly_heatmap_chart,
        plotly_weekly_heatmap_chart,
        fundamentals_header,
        fundamentals_description,
        table_fig,
        promo_header,
        promo_description,
        columns=1,
    )

    # Weekly Summary Table
    weekly_summary_table_content = """
  ## Weekly Summary Table
  **Purpose:** Provides a comprehensive overview of the current Bitcoin market situation, focusing on key metrics for a specific date.
  **Key Metrics:**
  - **Bitcoin Price USD:** The current market price of Bitcoin in US dollars.
  - **Report Date:** The specific date the report was run.
  - **Bitcoin Marketcap:** The total market capitalization of Bitcoin.
  - **Sats Per Dollar:** The number of satoshis (smallest unit of Bitcoin) per US dollar.
  - **Bitcoin Dominance:** Bitcoin's market dominance compared to other cryptocurrencies.
  - **Bitcoin Trading Volume:** The total trading volume of Bitcoin in the specified period.
  - **Bitcoin Market Sentiment:** An indicator of the current market sentiment towards Bitcoin.
  - **Bitcoin Market Trend:** The general market trend (Bullish, Neutral, Bearish) during the reporting period.
  - **Bitcoin Valuation:** The current valuation status of Bitcoin (Overvalued, Undervalued, Fair Value).
  
  **Insights:** This table gives a quick and comprehensive view of Bitcoin's market status. It helps readers understand the current valuation, market dominance, and trading trends of Bitcoin, which are critical for making informed investment decisions.
  """
    weekly_summary_table = dp.Text(weekly_summary_table_content)

    # OHLC Chart Overview
    ohlc_chart_overview = dp.Text("""
  ## Open-High-Low-Close (OHLC) Chart
  **Purpose:** Presents an OHLC chart of Bitcoin prices, offering a detailed view of its price movements within specific time frames.
  
  **Insights:** Essential for technical analysis, providing insights into market sentiment and potential price directions. Helps in identifying patterns like bullish or bearish trends, breakouts, or reversals.
  """)

    # Combine the sections for the first part of the report
    first_section = dp.Group(weekly_summary_table, ohlc_chart_overview)

    # Cryptocurrency Performance Table
    crypto_performance_overview = dp.Text("""
  ## Cryptocurrency Performance Table
  **Purpose:** Compares the performance of various cryptocurrencies over a specified period.
  
  **Insights:** This table allows readers to compare the performance of major cryptocurrencies, providing insights into their relative strengths, market movements, and correlations with Bitcoin.
  """)

    # Index Performance Table
    index_performance_overview = dp.Text("""
  ## Index Performance Table
  **Purpose:** Analyzes the performance of Bitcoin against major stock indices like Nasdaq, S&P500, and ETFs.
  
  **Insights:** By comparing Bitcoin with major indices, this table provides an understanding of how Bitcoin performs in relation to traditional financial markets.
  """)

    # Macro Performance Table
    macro_performance_overview = dp.Text("""
  ## Macro Performance Table
  **Purpose:** Evaluates Bitcoin's performance in relation to macroeconomic indicators like the US Dollar Index, Gold Futures, and others.
  
  **Insights:** This table is crucial for understanding Bitcoin's position and performance in the global economic landscape.
  """)

    # Equities Performance Table
    equities_performance_overview = dp.Text("""
  ## Equities Performance Table
  **Purpose:** Focuses on the performance of equities related to Bitcoin and the cryptocurrency market, such as COIN, SQ, MSTR, MARA, RIOT, etc.
  
  **Insights:** This table illustrates how Bitcoin-related equities perform in the stock market, providing insights into the investor sentiment.
  """)

    # Combine the sections for the second part of the report
    second_section = dp.Group(
        crypto_performance_overview,
        index_performance_overview,
        macro_performance_overview,
        equities_performance_overview,
    )

    # Price Buckets Analysis
    price_buckets_analysis_text = """
  ## Price Buckets Analysis
  **Purpose:** Categorizes Bitcoin prices into defined buckets, providing a view of how many days Bitcoin traded within specific price ranges.
  
  **Insights:** Offers a historical perspective on the price distribution of Bitcoin. Helps in understanding which price ranges have been most common, potentially indicating key support or resistance levels.
  """

    # Monthly Heatmap of Returns
    monthly_heatmap_text = """
  ## Monthly Heatmap of Returns
  **Purpose:** Presents monthly and yearly Bitcoin returns in a heatmap format, providing a quick visual overview of performance over time.
  
  **Insights:** Allows for easy identification of periods with high returns or significant losses. Can be used to spot seasonal patterns or annual trends in Bitcoin's market performance.
  """

    # Weekly Heatmap
    weekly_heatmap_text = """
  ## Weekly Heatmap
  **Purpose:** Shows Bitcoin's weekly returns over the last 5 years in a heatmap format.
  
  **Insights:** Useful for spotting short-term trends and weekly patterns in Bitcoin's price movements.
  """

    # Year-Over-Year (YOY) Change Chart
    yoy_change_chart_text = """
  ## Year-Over-Year (YOY) Change Chart
  **Purpose:** Plots Bitcoin's year-over-year percentage change, providing insights into its long-term growth trajectory.
  
  **Insights:** Highlights periods of significant growth or decline. Useful for investors focusing on long-term trends.
  """

    # Return on Investment (ROI) Table
    roi_table_text = """
  ## Return on Investment (ROI) Table
  **Purpose:** Calculates and presents the ROI for Bitcoin over various time frames, providing a snapshot of its investment performance.
  
  **Insights:** Allows investors to gauge the historical profitability of investing in Bitcoin over different periods. Helps in comparing short-term versus long-term investment returns.
  """

    # Moving Averages (MA) Table
    ma_table_text = """
  ## Moving Averages (MA) Table
  **Purpose:** Computes and displays various moving averages for Bitcoin, giving insights into its trending behavior.

  **Insights:** Moving averages are key indicators used in technical analysis to smooth out price trends and identify momentum. Helps in determining bullish or bearish market sentiments.
  """

    # Combine the sections for the third part of the report
    third_section = dp.Group(
        dp.Text(price_buckets_analysis_text),
        dp.Text(monthly_heatmap_text),
        dp.Text(weekly_heatmap_text),
        dp.Text(yoy_change_chart_text),
        dp.Text(roi_table_text),
        dp.Text(ma_table_text),
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
            dp.Group(first_section, label="Market Summary"),
            dp.Group(second_section, label="Performance Tables"),
            dp.Group(third_section, label="Market Analysis"),
            dp.Group(fourth_section, label="Fundamentals Table"),
        ]
    )

    # Definition Summary
    definition_layout = dp.Group(header_definition, definition_tabs, columns=1)
    report_tabs = dp.Select(
        blocks=[
            dp.Group(difficulty_summary_layout, label="Weekly Market Update"),
            dp.Group(definition_layout, label="Report Definitions / Glossary"),
        ]
    )

    # Combine all parts for final report
    report_blocks = [welcome_text, report_tabs]

    # Return the final layout structure
    return report_blocks
