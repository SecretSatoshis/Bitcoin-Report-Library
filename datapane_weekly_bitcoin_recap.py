import datapane as dp
import pandas as pd


def generate_report_layout_weekl_bitcoin_recap(
    weekly_summary_big_numbers,
    ohlc_plot,
    equity_performance_table,
    sector_performance_table,
    macro_performance_table,
    bitcoin_performance_table,
    plotly_heatmap_chart,
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

    # Performance Report Summary Section Components
    performance_header = dp.Text("# Performance Tables")
    performance_description = dp.Text("""
    A comparative view of Bitcoin's performance against key Equity Indexes, Sector ETFs, Macro Assets, and Bitcoin Stocks & ETFs.
      """)

    # Monthly Heatmap Summary Header
    header_heatmap_section = dp.Text("### Monthly Bitcoin Price Performacne Heatmap")

    # Weekly OHLC Bitcoin Price Chart Summary Header
    header_weekly_price_section = dp.Text("### Bitcoin Weekly OHLC Price Chart")

    # Define individual headers for each table
    header_equity = dp.Text("### Equity Index Performance")
    header_sector = dp.Text("### Sector ETF Performance")
    header_macro = dp.Text("### Macro Asset Performance")
    header_bitcoin = dp.Text("### Bitcoin Stock & ETF Performance")

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
  ## Open-High-Low-Close (OHLC) Chart
  **Purpose:** Presents an OHLC chart of Bitcoin prices, offering a detailed view of its price movements within specific time frames.
  
  **Insights:** Essential for technical analysis, providing insights into market sentiment and potential price directions. Helps in identifying patterns like bullish or bearish trends, breakouts, or reversals.
  """)

    # Combine the sections for the first part of the report
    first_section = dp.Group(summary_table, ohlc_chart_overview)

    # Equity Index Performance Table
    equity_index_performance_overview = dp.Text("""
    ## Equity Index Performance Table
    **Purpose:** Provides a comparative view of Bitcoin's performance against major stock indices like Nasdaq, and S&P500.

    **Insights:** By comparing Bitcoin with major indices, this table offers an understanding of how Bitcoin performs in relation to traditional financial markets.
    """)

    # Sector ETF Performance Table
    sector_etf_performance_overview = dp.Text("""
    ## Sector ETF Performance Table
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
    ## Bitcoin-Related Stocks & ETFs Performance Table
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
  ## Monthly Heatmap of Returns
  **Purpose:** Presents monthly and yearly Bitcoin returns in a heatmap format, providing a quick visual overview of performance over time.
  
  **Insights:** Allows for easy identification of periods with high returns or significant losses. Can be used to spot seasonal patterns or annual trends in Bitcoin's market performance.
  """

    # Combine the sections for the third part of the report
    third_section = dp.Group(
        dp.Text(monthly_heatmap_text),
    )

    # Definition Tab Tables
    definition_tabs = dp.Select(
        blocks=[
            dp.Group(first_section, label="Weekly Bitcoin Recap"),
            dp.Group(second_section, label="Performance Tables"),
            dp.Group(third_section, label="Monthly Heatmap"),
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
