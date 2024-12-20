import datapane as dp


def generate_report_layout_difficulty(
    difficulty_big_numbers, performance_table, fundamentals_table, valuation_table
):
    welcome_text = dp.Text("""
  ## Secret Satoshis | Bitcoin Difficulty Adjustment Report

  Welcome to the Secret Satoshis Difficulty Adjustment Report a comprehensive resource tailored for subscribers of the Secret Satoshis newsletter offering an in-depth analysis of key metrics, trends, and insights to keep you informed about the state of the Bitcoin network.

  ## Report Navigation:

  The report is divided into two main sections for a structured and streamlined analysis:

  1. **Difficulty Adjustment Data**: Explore a detailed analysis of the latest difficulty adjustment, including network difficulty, hash rate trends, miner revenues, and Bitcoin price changes over the adjustment period.

  2. **Report Definitions/Glossary**: This section serves as a glossary, providing detailed definitions of the terms, metrics, and concepts used throughout the report to enhance your understanding.

  To navigate between different sections, simply click on the respective tab.
  """)

    # Difficulty Report Summary Section Components
    difficulty_header = dp.Text("# Difficulty Adjustment Report Summary")
    difficulty_footer = dp.Text(
        "For a deeper dive into the key metrics, visit the Report Definitions Tab above."
    )

    # Performance Report Summary Section Components
    performance_header = dp.Text("# Performance Table")
    performance_description = dp.Text("""
  A comparative view of Bitcoin's performance against key assets and indices.
      """)
    performance_footer = dp.Text(
        "For a deeper dive into the performance metrics, visit the Report Definitions Tab above."
    )

    # Fundamentals Report Summary Section Components
    fundamentals_header = dp.Text("# Fundamentals Table")
    fundamentals_description = dp.Text("""
  A detailed analysis of Bitcoin's on-chain activity, illustrating network security, economic activity, and overall user engagement.
      """)
    fundamentals_footer = dp.Text(
        "For a deeper dive into the fundamental metrics, visit the Report Definitions Tab above."
    )

    # Valuation Report Summary Section Components
    valuation_header = dp.Text("# Valuation Table")
    valuation_description = dp.Text("""
  A comprehensive guide to Bitcoin's valuation through various analytical lenses, aiding in understanding its current market cycle and position.
      """)
    valuation_footer = dp.Text(
        "For a deeper dive into the valuation metrics, visit the Report Definitions Tab above."
    )

    # Difficulty Adjustment Newsletter Promo
    promo_header = dp.Text(
        "## Continue Your Bitcoin Journey With The Secret Satoshis Newsletter: <a href='https://www.newsletter.secretsatoshis.com/' target='_blank'>Subscribe Now</a>"
    )
    promo_description = dp.Text("""
    Subscribe to the Secret Satoshis Newsletter for exclusive, daily Bitcoin market updates and data-driven analysis powered by Agent 21. Stay informed with concise insights that help you navigate the Bitcoin landscape with confidence.

    Subscribe Now to Stay Ahead: <a href="https://www.newsletter.secretsatoshis.com/" target="_blank">Subscribe Now</a>.""")

    # Difficulty Report Summary
    difficulty_summary_layout = dp.Group(
        difficulty_header,
        difficulty_big_numbers,
        difficulty_footer,
        performance_header,
        performance_description,
        performance_table,
        performance_footer,
        fundamentals_header,
        fundamentals_description,
        fundamentals_table,
        fundamentals_footer,
        valuation_header,
        valuation_description,
        valuation_table,
        valuation_footer,
        promo_header,
        promo_description,
        columns=1,
    )

  
    # Definitions
    difficulty_summary = dp.Text("""
  #### Difficulty Table Summary
  The Difficulty Table aggregates crucial metrics necessary for analyzing Bitcoin's operational health and market position. It presents data on Bitcoin's supply scarcity and mining indicators, serving as a central resource for understanding the primary factors influencing Bitcoin's value.
  """)

    difficulty_metrics = dp.Text("""
  ### Difficulty Table Metrics Overview
  
  **Report Date:**
  - Description: The specific date the report was compiled.
  - Interpretation & Usage:  Establishes a time reference, critical for analyzing trends and data variations across different periods.

  
  **Bitcoin Supply:**
  - Description: Represents the circulating bitcoins.
  - Interpretation & Usage: As we approach the 21 million cap, this metric becomes increasingly significant for understanding market scarcity and its potential implications on market dynamics.

  **7 Day Average Hashrate:**
  - Description: Represents the aggregate computational power utilized for mining and processing transactions on the Bitcoin network.
  - Interpretation & Usage: An indicator of network security, where a higher hashrate denotes a more secure network.
  
  **Difficulty:**
  - Description: A parameter indicating the computational complexity involved in mining a new block on the Bitcoin network.
  - Interpretation & Usage: A key indicator of the competition among miners and the network's security level, offering insights into the current mining landscape.

  **Last Difficulty Adjustment Block Height:**
  - Description: Indicates the block height at which the last difficulty adjustment occurred on the Bitcoin network.
  - Interpretation & Usage: This metric serves as a critical reference point in evaluating the frequency and implications of difficulty adjustments in the network.

  **Last Difficulty Change:**
  - Description: The change in mining difficulty since the last adjustment.
  - Interpretation & Usage: A mirror reflecting changes in network hashrate and mining competition dynamics, serving as a tool for detecting shifts in mining activity and potential network adjustments.
  
  **Price USD:**
  - Description: Bitcoin's valuation in US dollars as of the report date.
  - Interpretation & Usage: A direct gauge of market sentiment, essential in identifying potential investment opportunities and assessing market dynamics.
  
  **Marketcap:**
  - Description: The aggregate market valuation of circulating bitcoins.
  - Interpretation & Usage: A measure of Bitcoin's market significance, key for understanding its overall market position and relative size in the market.
  
  **Sats Per Dollar:**
  - Description: Indicates the number of satoshis that can be purchased with one US dollar, highlighting the dollar's value in Bitcoin terms.
  - Interpretation & Usage: This metric, observed over time, offers insights into the dollar's purchasing power in the Bitcoin ecosystem, aiding investors in understanding currency value shifts, particularly during significant Bitcoin appreciations.
      """)

    difficulty_table_header = dp.Text("## Difficulty Table Summary")

    # Difficulty Tab Tables
    difficulty_tabs = dp.Group(difficulty_summary, difficulty_metrics)

    # Difficuclty Table Summary
    difficulty_definition_layout = dp.Group(
        difficulty_table_header, difficulty_tabs, columns=1
    )
    # Defintions
    performance_summary = dp.Text("""
  #### Performance Table Summary
  The Performance Table offers a high-level analysis of Bitcoin's market behavior in comparison with vital financial indices and assets such as the S&P 500, gold, and sector-specific ETFs. This table encompasses a range of carefully selected metrics, including current valuations, historical performance markers, and return percentages over specified periods, providing a comprehensive overview that is instrumental for investment analysis.
  """)
    performance_columns = dp.Text("""
  ### Performance Table Column Overview
  **Asset | Ticker:**
  - Description:  Indicates the specific asset or index being analyzed, which could range from bitcoin to stock indices and other financial assets.
  - Interpretation & Usage: A foundational reference point that guides investors in navigating through different investment options, facilitating detailed comparative analyses.
  
  **Price:**
  - Description: Specifies the current market valuation of the indicated asset or index.
  - Interpretation & Usage: Serving as an immediate reflection of market sentiment, this metric allows investors to gauge the current market value and identify potential investment opportunities.

   **1 Day Return:**
  - Description: Reflects the asset's performance over the past 24 hours, illustrating short-term market fluctuations.
  - Interpretation & Usage: Enables investors to monitor daily performance trends, vital for those employing short-term investment strategies or seeking to capitalize on market volatility.
  
  **Difficulty Period Return:**
  - Description: Represents the asset's return during the most recent Bitcoin difficulty adjustment period.
  - Interpretation & Usage: Offers a nuanced view of asset performance within the unique context of Bitcoin's mining dynamics.
  
  **MTD Return:**
  - Description: Represents the asset's performance from the beginning of the current month up to the present date.
  - Interpretation & Usage: This metric provides insight into the asset's short-term performance.
  
  **90 Day Return:**
  - Description: Indicates the asset's performance trajectory over the past quarter.
  - Interpretation & Usage: ffers a broader perspective for investors to evaluate the asset's quarterly performance, facilitating a deeper understanding of underlying market trends.
  
  **YTD Return:**
  - Description: Represents the asset's performance since the beginning of the current year.
  - Interpretation & Usage:  A comprehensive tool enabling investors to assess the asset's annual performance, aligning investment strategies with annual market benchmarks and trends.

  **4 Year CAGR (Compound Annual Growth Rate):**
  - Description: Reflects the mean annual growth rate of an investment over a four year period.
  - Interpretation & Usage:  Facilitates long-term investment planning by providing insights into the asset's historical growth trajectory, aiding investors in crafting strategies grounded in historical performance data.

  **4 Year Sharpe Ratio:**
  - Description: A measure of the performance of an investment compared to a risk-free asset, after adjusting for its risk.
  - Interpretation & Usage:  Assists investors in understanding the risk-adjusted performance of assets.
  - Risk Free Rate - 3 Month Treasury Yield - Ticker IRX

  **90 Day BTC Correlation:**
  - Description: Indicates the degree to which the asset's performance is correlated with Bitcoin's performance over the past 90 days.
  - Interpretation & Usage:   A vital tool for investors seeking to understand the asset's relationship with Bitcoin, facilitating the creation of diversified portfolios and risk management strategies.

  **52 Week Low:**
  - Description: Marks the lowest valuation the asset has reached in the past 52 weeks.
  - Interpretation & Usage: By showcasing the asset's lowest valuation over the past year, this metric offers insights into its resilience, potential support levels, and market valuation under adverse conditions.
  
  **52 Week High:**
  - Description: Represents the highest price point the asset achieved in the past 52 weeks.
  - Interpretation & Usage: Highlighting the peak of the asset's performance, this metric gives a glimpse of its maximum momentum and potential resistance levels.
      """)
    performance_metrics = dp.Text("""
  ### Performance Table Metrics Overview
  **Bitcoin | BTC:**
  - Description: Bitcoin is a decentralized cryptocurrency, leveraging blockchain technology to enable peer-to-peer transactions without a central authority.
  - Significance: As the first and most prominent cryptocurrency, Bitcoin sets the standard for the industry, offering insights into the broader digital currency landscape.
  
  **Nasdaq | IXIC:**
  - Description: The Nasdaq Composite Index measures all Nasdaq domestic and international stocks.
  - Significance: It provides a snapshot of the tech-centric stock market, indicating the health and trends of the technology and innovation sectors.
  
  **S&P500 | GSPC:**
  - Description: The S&P 500 is a stock market index that tracks the performance of 500 large companies listed on U.S. stock exchanges.
  - Significance: Representing the broader U.S. equities market, the S&P 500 is a key barometer of the health of the U.S. economy and is often used as a benchmark for many investment portfolios.
  
  **Financials ETF | XLF:**
  - Description: The Financial Select Sector SPDR Fund (XLF) tracks the financial sector, including banks, investment banks, and insurance companies.
  - Significance: It offers a lens into the health and performance of the financial sector, a cornerstone of the economy.
  
  **Energy ETF | XLE:**
  - Description: The Energy Select Sector SPDR Fund (XLE) tracks the energy sector, encompassing companies in exploration, production, and marketing of energy.
  - Significance: It provides a view into the energy market, capturing trends in oil, gas, and renewable energy sources.

  **FANG+ ETF | FANG.AX:**
  - Description: FANG+ ETF tracks the performance of next-generation technology and tech-enabled companies.
  - Significance: It serves as a metric to gauge the performance of high-growth companies in the technology and consumer sectors.
  - Data: 2 year Sharpe & CAGR data used
  
  **Crypto Industry ETF | BITQ:**
  - Description: BITQ tracks a market-cap-weighted index of bitcoin-focused companies.
  - Significance: It offers a broader perspective on the bitcoin industry's performance, beyond just individual cryptocurrencies.
  - Data: 2 year Sharpe & CAGR data used
  
  **Gold | GC:**
  - Description: Gold, a precious metal, has been a store of value for millennia.
  - Significance: Often seen as a hedge against economic uncertainties, gold's performance is a key indicator of broader economic sentiments.
  
  **US Dollar | DX:**
  - Description: The U.S. Dollar Index (DX) gauges the value of the U.S. dollar against a basket of foreign currencies.
  - Significance: It offers insights into the strength or weakness of the U.S. dollar, influencing global trade, commodity prices, and interest rates.
  
  **Treasury Bond ETF | TLT:**
  - Description:  The TLT Treasury Bond ETF tracks the performance of U.S. Treasury securities with a remaining maturity of twenty years or more.
  - Significance:  A pivotal indicator of investor sentiment towards the U.S. government's fiscal policy and economic outlook, it offers insights into the long-term trajectory of U.S. government bonds, influencing various financial markets and investment strategies.
      """)

    # Performance Tab Tables
    performance_tabs = dp.Group(
        performance_summary,
        performance_columns,
        performance_metrics,
    )

    performance_table_header = dp.Text("## Performance Table Summary")

    # Performance Table Summary
    performance_definition_layout = dp.Group(
        performance_table_header, performance_tabs, columns=1
    )
    # Defintions
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

    # Defintions
    valuation_summary = dp.Text("""
  #### Valuation Table Summary
  The Valuation Table offers deep insights into Bitcoin's current market position through an array of valuation frameworks. By integrating both technical price, on-chain, and relative valuation models, it aids investors in discerning Bitcoin's current market cycle phase. It not only delineates potential areas of overvaluation or undervaluation but also assists in understanding how Bitcoin's current value stacks up against various benchmarks and economic indicators. This strategic blend of analytical perspectives is designed to help facilitate well-rounded investment decisions by providing a comprehensive view of Bitcoin's market dynamics and potential growth trajectories.
  """)

    valuation_columns = dp.Text("""
  ### Valuation Table Column Overview
  **Valuation Model:**
  - Description: The specific model or metric used to assess Bitcoin's valuation.
  - Interpretation & Usage: Provides a framework or methodology for understanding Bitcoin's value in various contexts. Use to compare and contrast different valuation perspectives.
  
  **Model Price:**
  - Description: The price or value of Bitcoin as suggested by the specific valuation model.
  - Interpretation & Usage: Indicates the perceived value of Bitcoin according to the model's parameters. Compare with the current BTC price to determine if Bitcoin is overvalued or undervalued based on the model.
  
  **Difficulty Period Change:**
  - Description: The change in the specific valuation metric over the most recent difficulty adjustment period.
  - Interpretation & Usage: Provides insights into short-term shifts in valuation perspectives. Monitor for sudden changes which can indicate shifts in market sentiment or fundamentals.
  
  **BTC Price:**
  - Description: The current market price of Bitcoin.
  - Interpretation & Usage: Represents the consensus value of Bitcoin among market participants at a given time. Use as a benchmark to compare against various model prices.
  
  **Buy Target:**
  - Description: The suggested price level at which the asset is considered undervalued and a potential buy according to the model.
  - Interpretation & Usage: Provides a price target for potential entry points based on the model's parameters. Use to identify potential buying opportunities.
  
  **Sell Target:**
  - Description: The suggested price level at which the asset is considered overvalued and a potential sell according to the model.
  - Interpretation & Usage: Provides a price target for potential exit points based on the model's parameters. Use to identify potential selling opportunities.
  
  **% To Model Price:**
  - Description: The percentage difference between the current BTC price and the model price.
  - Interpretation & Usage: Indicates how closely the current market price aligns with the model's valuation. Monitor to gauge market sentiment relative to the model's perspective.

  **% To Buy Target:**
  - Description: The percentage difference between the current BTC price and the model's buy target.
  - Interpretation & Usage: Indicates the gap between the current market price and the model's buy threshold. It can be used to gauge potential undervaluation and identify opportune moments to enter the market.

  **% To Sell Target:**
  - Description: The percentage difference between the current BTC price and the model's sell target.
  - Interpretation & Usage: Indicates how closely the current market price approaches the model's sell threshold. Use to assess potential overvaluation and exit strategies.
  """)

    valuation_metrics = dp.Text("""
  ### Valuation Table Metrics Overview
 
 **200 Day Moving Average:**
  - Description: This represents the average of Bitcoin's closing prices over the last 200 days.
  - Interpretation & Usage: Recognized as a significant benchmark in the financial world, this average provides insights into Bitcoin's long-term price trajectory. Deviations from this average can be indicative of market trends.
  - Model Price: Calculated as the average of Bitcoin's closing prices over the last 200 days. It's derived from PriceUSD'.rolling(window=200).mean().
  - Buy Target: 0.7 times the 200-day moving average price.
  - Sell Target: 2.2 times the 200-day moving average price.
  
  **NVT Price:**
  - Description: The NVT (Network Value to Transactions) Price is a metric that measures Bitcoin's market capitalization against the volume of transactions happening on its network. It is calculated using a formula that takes the median of the NVT ratio over a two-year period, multiplies it with the transaction volume in USD, and divides it by the current supply of bitcoins.
  - Interpretation & Usage: The NVT Price helps in evaluating the relative valuation of Bitcoin. A high NVT Price can signal that Bitcoin is overvalued compared to the value of transactions occurring on the network, while a low NVT Price might indicate undervaluation. It is a useful tool for investors to gauge the market dynamics and make informed decisions based on Bitcoin's on-chain economic activity.
  - Model Price: The NVT Price is derived from the formula: (NVTAdj.rolling(window=365*2).median() * TxTfrValAdjUSD' / 'SplyCur' which calculates the market cap to transaction volume ratio based on a 2-year median rolling window.
  - Buy Target: 0.5 times the NVT model price.
  - Sell Target: 2.0 times the NVT model price.
  
  **Realized Price:**
  - Description: This metric denotes the average price at which all available bitcoins were last transacted.
  - Interpretation & Usage: It provides a comprehensive view into the average cost basis of Bitcoin holders, offering insights into prevailing market sentiment and potential price support zones.
  - Model Price: Calculated as the total realized market capitalization divided by the current supply of bitcoins. It is represented by the formula: Realized Price = CapRealUSD / SplyCur Here, "CapRealUSD" represents the realized market capitalization and "SplyCur" represents the current supply of bitcoins.
  - Buy Target: 0.8 times the Realized Price.
  - Sell Target: 3.0 times the Realized Price.
  
  **ThermoCap Price:**
- Description: The ThermoCap Price is a metric that contrasts Bitcoin's market capitalization with the total revenue generated from mining activities, including block rewards and transaction fees. It offers insights into the financial health and intrinsic value of the Bitcoin network from a miner revenue perspective.
- Interpretation & Usage: This metric assists investors in understanding the market's valuation of Bitcoin based on the revenue generated from mining activities.
- Model Price: The ThermoCap Model Price is  the 8x ThermoCap Multiple. Which is calcualted by 8 * 'thermocap_multiple' = CapMrktCurUSD / RevAllTimeUSD' 
- Buy Target: 5 times the ThermoCap
- Sell Target: 25 times the ThermoCap.
  
  **Stock-to-Flow (S/F) Model Price:**
- Description: The Stock-to-Flow (S/F) model price is derived from a valuation model that relates the price of Bitcoin to its scarcity, which is measured by the S/F ratio. The S/F ratio is calculated as the total supply of Bitcoin divided by the annual production (flow) of new bitcoins.
- Interpretation & Usage: The S/F model price serves as a tool to estimate the intrinsic value of Bitcoin based on its scarcity. A high S/F ratio indicates increased scarcity and potentially higher value. Investors can use the S/F multiple, which is the actual price divided by the S/F model price, to identify overvalued or undervalued conditions in the market. 
- Model Price: The S/F Model Price formula is SF_Predicted_Price=exp(intercept)*exp(intercept)*(SF ratio)^slope
- Buy Target: 0.3 times the SF model price. 
- Sell Target: 3.0 times the SF model price.
  
  **Silver Market Cap:**
- **Description:** This metric calculates the price at which Bitcoin's market cap would match the entire market cap of all mined silver.
- **Interpretation & Usage:** It provides a perspective on Bitcoin's valuation, highlighting its potential to stand alongside traditional precious metals in terms of market capitalization.
- **Model Calculatoin: ** The model calculation evaluates the potential of Bitcoin achieving the market cap of Silver in 10 years. It assigns different probability scenarios (bull, base, bear) to represent the likelihood of each occurrence. Utilizing the 10 year Treasruy yield as a discount rate, these projected values are then discounted back to their present value.
- **Model Price:** The bull case value.
- **Buy Target:** The base case value.
- **Sell Target:** The current total market cap of silver, representing a scenario where Bitcoin fully realizes its potential compared to silver.
- **Probabilities:** The probabilities assigned to the bull, base, and bear cases are 95%, 80%, and 50% respectively, indicating different levels of confidence in these scenarios.
  
  **UK M0 Price:**
- **Description:** This metric projects the price at which Bitcoin's market cap would equal the total monetary base (M0) of the United Kingdom.
- **Interpretation & Usage:** By comparing Bitcoin's market cap with the UK's M0, investors can gauge Bitcoin's growth potential and its capacity to parallel traditional fiat systems in terms of market valuation.
- **Model Calculatoin: ** The model calculation evaluates the potential of Bitcoin achieving the market cap of UK M0 in 10 years. It assigns different probability scenarios (bull, base, bear) to represent the likelihood of each occurrence. Utilizing the 10 year Treasruy yield as a discount rate, these projected values are then discounted back to their present value.
- **Model Price:** The bull case value.
- **Buy Target:** The base case value.
- **Sell Target:** The current total monetary base (M0) of the United Kingdom, representing a scenario where Bitcoin fully realizes its potential compared to the UK's fiat system.
  - **Probabilities:** The probabilities assigned to the bull, base, and bear cases are 65%, 35%, and 10% respectively, indicating different levels of confidence in these scenarios.
  
  **Apple Market Cap:**
- **Description:** This metric projects the price at which Bitcoin's market cap would equal the market cap of Apple, the world's most valuable company by market capitalization.
- **Interpretation & Usage:** It offers investors a perspective on Bitcoin's potential growth, highlighting its capacity to rival the market capitalization of leading global corporations.
- **Model Calculatoin: ** The model calculation evaluates the potential of Bitcoin achieving the market cap of Apple in 10 years. It assigns different probability scenarios (bull, base, bear) to represent the likelihood of each occurrence. Utilizing the 10 year Treasruy yield as a discount rate, these projected values are then discounted back to their present value.
- **Model Price:** The bull case value.
- **Buy Target:** The base case value.
- **Sell Target:** The current total market cap of Apple, representing a scenario where Bitcoin fully realizes its potential compared to Apple.
- **Probabilities:** The probabilities assigned to the bull, base, and bear cases are 55%, 25%, and 10% respectively, indicating different levels of confidence in these scenarios.
  
  **US M0 Price:**
- **Description:** This metric forecasts the price at which Bitcoin's market cap would surpass the US's total base money supply (M0).
- **Interpretation & Usage:** It's an insightful tool for investors to understand Bitcoin's potential to challenge and possibly exceed traditional fiat systems in terms of market dominance.
- **Model Calculatoin: ** The model calculation evaluates the potential of Bitcoin achieving the market cap of US M0 in 10 years. It assigns different probability scenarios (bull, base, bear) to represent the likelihood of each occurrence. Utilizing the 10 year Treasruy yield as a discount rate, these projected values are then discounted back to their present value.
- **Model Price:** The bull case value.
- **Buy Target:** The base case value.
- **Sell Target:** The current total base money supply (M0) of the US, representing a scenario where Bitcoin fully realizes its potential compared to the US fiat system.
- **Probabilities:** The probabilities assigned to the bull, base, and bear cases are 35%, 15%, and 5% respectively, indicating different levels of confidence in these scenarios.
  
  **Gold Market Cap:**
- **Description:** This metric projects the price at which Bitcoin's market cap would equal the total market cap of all mined gold.
- **Interpretation & Usage:** It serves as a benchmark for assessing Bitcoin's potential to rival gold as a leading store of value in the global financial landscape.
- **Model Calculatoin: ** The model calculation evaluates the potential of Bitcoin achieving the market cap of Gold in 10 years. It assigns different probability scenarios (bull, base, bear) to represent the likelihood of each occurrence. Utilizing the 10 year Treasruy yield as a discount rate, these projected values are then discounted back to their present value.
- **Model Price:** The bull case value.
- **Buy Target:** The base case value.
- **Sell Target:** The current total market cap of all mined gold, representing a scenario where Bitcoin fully realizes its potential compared to gold.
- **Probabilities:** The probabilities assigned to the bull, base, and bear cases are 20%, 10%, and 5% respectively, indicating different levels of confidence in these scenarios.
""")

    # valuation Tab Tables
    valuation_tabs = dp.Group(
        valuation_summary,
        valuation_columns,
        valuation_metrics,
    )

    valuation_table_header = dp.Text("## Valuation Table Summary")

    # valuation Table Summary
    valuation_definition_layout = dp.Group(
        valuation_table_header, valuation_tabs, columns=1
    )

    definition_header = dp.Text("# Table Definitions")

    # Definition Tab Tables
    definition_tabs = dp.Select(
        blocks=[
            dp.Group(difficulty_definition_layout, label="Difficulty Table"),
            dp.Group(performance_definition_layout, label="Performance Table"),
            dp.Group(fundamentals_definition_layout, label="Fundamentals Table"),
            dp.Group(valuation_definition_layout, label="Valuation Table"),
        ]
    )

    # Definition Summary
    definition_layout = dp.Group(definition_header, definition_tabs, columns=1)
    report_tabs = dp.Select(
        blocks=[
            dp.Group(
                difficulty_summary_layout, label="Difficulty Adjustment Data Report"
            ),
            dp.Group(definition_layout, label="Report Definitions / Glossary"),
        ]
    )

    # Combine all parts for final report
    report_blocks = [welcome_text, report_tabs]

    # Return the final layout structure
    return report_blocks
