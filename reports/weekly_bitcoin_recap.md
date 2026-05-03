# Weekly Bitcoin Recap

## Purpose

The Weekly Bitcoin Recap is a newsletter-style report for readers who want a structured weekly view of Bitcoin market activity, on-chain context, price action, and valuation.

## Audience

- Secret Satoshis readers
- long-term Bitcoin investors

## Section Order

1. News Section
2. Weekly Bitcoin Recap Summary
3. Historical Performance
4. Monthly Heat Map
5. MTD Return Comparison
6. Weekly BTC/USD OHLC Analysis
7. Trading Range Analysis
8. ROI Profile
9. YTD Return Comparison
10. Relative Valuation
11. Conclusion

## Data Inputs

| Section | Input Type | Local Source | Published Source |
|---|---|---|---|
| News Section | RSS | NA | See "Prompt Specs → 1. News Section" for the canonical RSS feed list |
| Weekly Bitcoin Recap Summary | CSV | `csv/summary_table.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/summary_table.csv` |
| Historical Performance | CSV | `csv/performance_table.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/performance_table.csv` |
| Monthly Heat Map | CSV | `csv/monthly_heatmap_data.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/monthly_heatmap_data.csv` |
| MTD Return Comparison | CSV | `csv/mtd_return_comparison.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/mtd_return_comparison.csv` |
| Weekly BTC/USD OHLC Analysis | CSV | `csv/ohlc_data.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/ohlc_data.csv` |
| Trading Range Analysis | CSV | `csv/1k_bucket_table.csv`, `csv/5k_bucket_table.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/1k_bucket_table.csv`, `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/5k_bucket_table.csv` |
| ROI Profile | CSV | `csv/roi_table.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/roi_table.csv` |
| YTD Return Comparison | CSV | `csv/ytd_return_comparison.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/ytd_return_comparison.csv` |
| Relative Valuation | CSV | `csv/relative_value_comparison.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/relative_value_comparison.csv` |

## Global Execution Rules

- Use one derived report date from the latest available mapped report data.
- For each data-backed section, use the latest row or value at or before the report date.
- Do not use future-dated rows or incomplete periods unless the user explicitly provides a future report date.
- Use only mapped inputs for each section. If a mapped input is missing or inconsistent, state the gap instead of substituting unrelated data.
- Format BTC prices to the nearest dollar unless a section needs more precision.
- Format large dollar values as `$X.XX million`, `$X.XX billion`, or `$X.XX trillion`.
- Format percentages with two decimals and a `%` sign.
- Format ratios, correlations, and multiples with two decimals.
- Do not multiply values that are already labeled or documented as percentage-point values.
- Run the final assembled report through `voice.md` to remove filler, hype, generic wrap-up language, and source-voice leakage.

## Prompt Specs

### 1. News Section

```text
Generate the News Section using Bitcoin news from the 7-day report window.

Bitcoin-Native Sources:
1. Bitcoin Magazine: https://bitcoinmagazine.com/.rss/full/
2. CoinDesk: https://www.coindesk.com/arc/outboundfeeds/rss/
3. The Block: https://www.theblock.co/rss.xml
4. NoBSBitcoin: https://www.nobsbitcoin.com/feed/

Major Financial Outlets:
5. Bloomberg Markets: https://feeds.bloomberg.com/markets/news.rss
6. Financial Times Home: https://www.ft.com/rss/home
7. MarketWatch Top Stories: https://www.marketwatch.com/rss/topstories
8. Investing.com News: https://www.investing.com/rss/news.rss

Rules:
- Stay Bitcoin-focused.
- For Major Financial Outlets, filter for Bitcoin/crypto/macro 
- Use only stories published or posted within the report week.
- Prioritize Bitcoin-specific stories.
- Include broader crypto, macro, regulatory, or market-structure stories only if they materially affect Bitcoin or the broader digital asset market.
- Select one Top News Story Of The Week.
- Then select 3 to 5 Bitcoin News stories
- Then select 2 to 5 Market News stories: macro, regulatory, or financial-system stories that are materially Bitcoin-relevant.- Do not pad the section with weak stories.
- Do not invent missing facts, dates, sources, or URLs.
- Prefer source diversity when possible, but impact and relevance matter more than equal representation across sources.
- After the story list, write a short "News Impact" paragraph that synthesizes the combined market relevance of the selected stories.
- Focus on broad implications for Bitcoin adoption, market positioning, investor sentiment, and current market narrative without forecasting price impact.

Format Rules:
- Each story line is the headline only — no per-story explanation. Attribution in (Reported By: [Source], [Date]) format.
- Context, "why it matters," and synthesis go in the News Impact paragraph.

Return only:

News Stories:
Top News Story Of The Week:
[Headline]. (Reported By: [Source], [Date])
Bitcoin News:
[Headline]. (Reported By: [Source], [Date])
[Headline]. (Reported By: [Source], [Date])
[Headline]. (Reported By: [Source], [Date])
...
Market News:
[Headline]. (Reported By: [Source], [Date])
[Headline]. (Reported By: [Source], [Date])
...
News Impact:
[One concise synthesis paragraph covering why the selected stories matter, the themes connecting them, and their combined relevance to Bitcoin adoption, positioning, sentiment, and current market narrative. No price forecasts.]
```

### 2. Weekly Bitcoin Recap Summary

```text
Generate the Weekly Bitcoin Recap Summary using `summary_table.csv`.

Write in a formal tone suitable for professional investors.

Rules:
- Use only values present in the summary table.
- Read `summary_table.csv` as a labeled table with `Metric`, `Value`, and `Category` columns.
- Do not introduce fields that are not available.
- Format summary values using the global formatting rules.

Return a section titled:

Current State Of The Bitcoin Market

It should include:
- Market Activity
- On-Chain Activity
- Market Sentiment and Valuation

The narrative should explain the current market state using the summary table data only.
```

### 3. Historical Performance

```text
Generate the Historical Performance section using `performance_table.csv`.

Write in a formal tone suitable for hedge fund managers and research-oriented investors.

Rules:
- Use the current combined performance table.
- Compare Bitcoin with the asset rows present in the file.
- Use `7 Day Return` as the primary weekly ranking metric.
- Use MTD Return, YTD Return, 90 Day Return, and 90 Day BTC Correlation as secondary context where helpful.
- Interpret all return columns as percentage-point values. For example, `-4.21` means `-4.21%`, not `-0.0421%`.
- Use the `Category` column to group assets into Equity Market Indexes, Sectors, Macro Asset Classes, and Bitcoin Industry Performance.
- If `Category` is absent, infer those groups from the asset names and keep the grouping explicit in the section.
- Format `90 Day BTC Correlation` as a coefficient from `-1.00` to `1.00`, not as a percentage.
- Do not refer to assets or categories that are not in the current table.
- If Bitcoin is not the top performer by 7 Day Return, identify the top performer from the actual table.

Return a section that covers:
- Equity Market Indexes
- Sector and Equity Benchmarking
- Macro Asset Class Performance
- Bitcoin Industry Performance
- Summary
```

### 4. Monthly Heat Map

```text
Generate the Monthly Heat Map section using `monthly_heatmap_data.csv`.

Write in a formal tone suitable for professional investors.

Rules:
- This is a monthly heat map, not a weekly heat map.
- Interpret monthly heat map values as decimal returns. For example, `0.05` means `5.0%`, not `0.05%`; `-0.03` means `-3.0%`.
- Use the latest nonblank monthly value in the current year row as the current month-to-date value.
- Treat `Average` as the full historical average row.
- Treat `Median` as the historical median row.
- Treat `4-Year Average` as optional recent-cycle context.
- Compare the current month against the historical average and median rows where relevant.
- Explain the current month in context without overclaiming predictive power.
- Avoid predictive language. Frame this section as monthly context, not a forecast.

Return a section titled:

Bitcoin Monthly Heatmap Overview and Analysis

It should include:
- Report Date
- Time Context
- Monthly Heatmap Insights
- Current Data Interpretation
- Market Context for the Month
```

### 5. MTD Return Comparison

```text
Generate the Month-to-Date Return Comparison section using `mtd_return_comparison.csv`.

Rules:
- Use the numeric current-year row and the Median Projection row.
- Treat `Return (%)` and `Report Date Return (%)` as percentage-point values.
- Use the current-year row `Report Date Return (%)` as the current MTD return.
- Use the Median Projection row `Report Date Return (%)` as the median path as of the report date.
- Use the Median Projection row `Return (%)` and `End Price ($)` as the historical median end-of-month scenario.
- Calculate variance versus median as current-year `Report Date Return (%)` minus Median Projection `Report Date Return (%)`.
- Keep the language data-driven.
- Do not assume additional rows exist beyond the current year and Median Projection rows.
- Describe the median path as a historical benchmark, not a forecast.

Return a section titled:

Bitcoin Month-to-Date Return Comparison

It should explain:
- current MTD return
- report-date median projection
- variance versus median
- what the deviation suggests about current monthly positioning
```

### 6. Weekly BTC/USD OHLC Analysis

```text
Generate the Weekly BTC/USD analysis using `ohlc_data.csv`.

Rules:
- Use the most recent completed weekly OHLC row.
- Treat `Time` as the week-start timestamp.
- Use the latest OHLC row only if the report date is after that week has closed; otherwise use the prior completed weekly row.
- Base the analysis on the tabular open, high, low, and close values.
- Calculate weekly return as `(Close - Open) / Open`.
- Calculate weekly range as `High - Low`; optionally express range percentage as `(High - Low) / Open`.
- Describe weekly range, directional bias, and the relationship between open and close.

Return a section that includes:
- Reporting Period
- Open, High, Low, Close
- Price Action Overview
- Market Sentiment and Trend Analysis
- Key Support and Resistance Levels
- Potential Bullish, Base, and Bearish Scenarios
```

### 7. Trading Range Analysis

```text
Generate the Trading Range Analysis section using `1k_bucket_table.csv` and `5k_bucket_table.csv`.

Rules:
- Use the row where `Is Current Bucket = True` in each bucket table to identify the active price zone.
- Require exactly one `Is Current Bucket = True` row in each bucket table. 
- Use `Current Price` as the spot-price anchor for the section.
- Cross-check `Current Price` across the 1k and 5k bucket tables. 
- Treat `Count` as historical price observations in that range, not trading volume or liquidity.
- Use the 1k and 5k bucket counts to describe how historically active the current area has been.
- Compare current bucket activity against adjacent buckets and relevant higher-price buckets rather than letting early low-price buckets dominate the interpretation.
- Frame the analysis around historical trading concentration, not prediction.
- If both bucket tables imply different levels of granularity, use the 1k bucket for precise placement and the 5k bucket for broader context.

Return a section titled:

Trading Range Analysis

It should explain:
- the current price zone
- how active that zone has been historically
- how the narrower and broader bucket views compare
- what the current range implies about market positioning
```

### 8. ROI Profile

```text
Generate the ROI Profile section using `roi_table.csv`.

Rules:
- Use the `Time Frame` column directly.
- Interpret the ROI field as percentage points. Prefer `ROI (%)` when present; if the source still uses the legacy `ROI` column name, treat it the same way.
- Do not multiply ROI values by 100.
- Treat `Start Date` as the historical entry date for the holding period.
- Treat `BTC Price` as the historical entry price for that row.
- Group horizons explicitly as `short-term = 1 day, 3 day, 7 day, 30 day`, `medium-term = 90 day and 1 Year`, and `long-term = 2 Year, 4 Year, 5 Year, 10 Year`.
- Highlight which entry periods are positive or negative.
- Explain what the spread of outcomes says about Bitcoin's short-term volatility and long-term return profile.
- Keep the language data-driven and avoid overstating precision.
- Describe historical entry outcomes and holding-period sensitivity, not investment advice.

Return a section titled:

Bitcoin ROI Profile

It should include:
- short-term entry outcomes
- medium-term entry outcomes
- long-term entry outcomes
- what the distribution says about risk and reward across holding periods
```

### 9. YTD Return Comparison

```text
Generate the Year-to-Date Return Comparison section using `ytd_return_comparison.csv`.

Rules:
- Use the numeric current-year row and the Median Projection row.
- Treat `Return (%)` and `Report Date Return (%)` as percentage-point values.
- Use the current-year row `Report Date Return (%)` as the current YTD return.
- Use the Median Projection row `Report Date Return (%)` as the median path as of the report date.
- Use the Median Projection row `Return (%)` and `End Price ($)` as the historical median full-year scenario.
- Calculate variance versus median as current-year `Report Date Return (%)` minus Median Projection `Report Date Return (%)`.
- Treat the current-year row `End Price ($)` as the latest/current price.
- Treat the Median Projection row `End Price ($)` as the median-path year-end scenario price.
- Use the median projection to discuss context, not certainty.
- Keep the output concise and analytical.

Return a section titled:

Bitcoin Year-to-Date Return Comparison

It should include:
- YTD Performance Snapshot
- Year End Price Scenario Analysis
- Observations and Outlook
```

### 10. Relative Valuation

```text
Generate the Relative Valuation section using `relative_value_comparison.csv`.

Rules:
- Use the assets listed in the current file.
- Sort the table by parsed `Market Cap (USD)` before analyzing ranking. Use descending order for the compact parity table and for larger-target analysis; use the sorted order to identify assets Bitcoin has already surpassed or is approaching.
- Treat negative `BTC % Move to Marketcap BTC Price` values as assets Bitcoin has surpassed.
- Treat Bitcoin's `0%` row as the current reference point.
- Treat the smallest positive `BTC % Move to Marketcap BTC Price` values above Bitcoin as approaching valuation levels.
- Treat larger positive values as aspirational parity benchmarks.
- Explain which assets Bitcoin has already surpassed, which ones it is approaching, and which larger parity benchmarks remain aspirational.
- Use the current "Market Cap BTC Price" and "BTC % Move to Marketcap BTC Price" fields to frame parity analysis.
- Preserve the table's dollar and percentage units when they are already formatted.
- Treat `relative_value_comparison.csv` as authoritative for this section, even if values differ from another mapped source.
- Frame parity prices as valuation benchmarks, not forecasts, targets, or recommendations.
- Do not introduce assets that are not in the CSV.

Return a section titled:

Bitcoin Relative Valuation Analysis

It should include:
- a compact parity table
- surpassed assets
- approaching valuation levels
- aspirational targets
```

### 11. Conclusion

```text
Write the conclusion for the Weekly Bitcoin Recap using the completed report sections below.

Rules:
- Synthesize the 3 to 5 highest-signal findings from the completed sections.
- Keep any forward-looking language conditional and grounded in the data, levels, scenarios, or valuation context already discussed.
- Do not introduce new facts.
- Avoid generic wrap-up phrases, promotional language, and hype.

Sections to synthesize:
- News Section
- Weekly Bitcoin Recap Summary
- Historical Performance
- Monthly Heat Map
- MTD Return Comparison
- Weekly BTC/USD OHLC Analysis
- Trading Range Analysis
- ROI Profile
- YTD Return Comparison
- Relative Valuation

Return one concise conclusion paragraph.
```

## Final Assembly Notes

- Start with the News Section.
- Follow the section order above.
- Keep headings stable from week to week.
- Before finalizing, check that the report date is consistent across sections.
- Check that each section uses only its mapped inputs.
- Check that percentage and decimal conversions are correct.
- Check that missing fields were omitted or surfaced, not invented.
- Run the final phrasing pass with `voice.md`.
