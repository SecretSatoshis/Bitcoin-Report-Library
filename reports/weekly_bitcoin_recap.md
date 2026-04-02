# Weekly Bitcoin Recap

## Purpose

The Weekly Bitcoin Recap is a newsletter-style report for readers who want a structured weekly view of Bitcoin market activity, on-chain context, price action, and valuation.

## Audience

- Secret Satoshis readers
- long-term Bitcoin investors
- allocators and research-oriented market participants

## Section Order

1. News Section
2. Weekly Bitcoin Recap Summary
3. Historical Performance
4. Monthly Heat Map
5. MTD Return Comparison
6. Weekly BTC/USD OHLC Analysis
7. Trading Range Analysis
8. ROI Profile
9. Difficulty / Hash Rate / Network Security Snapshot
10. YTD Return Comparison
11. Model Valuation
12. On-Chain Valuation
13. Relative Valuation
14. Conclusion

## Data Inputs

| Section | Input Type | Local Source | Published Source |
|---|---|---|---|
| Weekly Bitcoin Recap Summary | CSV | `csv/summary_table.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/summary_table.csv` |
| Historical Performance | CSV | `csv/performance_table.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/performance_table.csv` |
| Monthly Heat Map | CSV | `csv/monthly_heatmap_data.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/monthly_heatmap_data.csv` |
| MTD Return Comparison | CSV | `csv/mtd_return_comparison.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/mtd_return_comparison.csv` |
| Weekly BTC/USD OHLC Analysis | CSV | `csv/ohlc_data.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/ohlc_data.csv` |
| Trading Range Analysis | CSV | `csv/1k_bucket_table.csv`, `csv/5k_bucket_table.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/1k_bucket_table.csv`, `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/5k_bucket_table.csv` |
| ROI Profile | CSV | `csv/roi_table.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/roi_table.csv` |
| Difficulty / Hash Rate / Network Security Snapshot | CSV | `csv/summary_table.csv`, `csv/master_metrics_data.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/summary_table.csv`, `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/master_metrics_data.csv` |
| YTD Return Comparison | CSV | `csv/ytd_return_comparison.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/ytd_return_comparison.csv` |
| Model Valuation | CSV | `csv/eoy_model_data.csv`, `csv/master_metrics_data.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/eoy_model_data.csv`, `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/master_metrics_data.csv` |
| On-Chain Valuation | CSV | `csv/master_metrics_data.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/master_metrics_data.csv` |
| Relative Valuation | CSV | `csv/relative_value_comparison.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/relative_value_comparison.csv` |

## Prompt Specs

### 1. News Section

```text
Generate the news section of the Weekly Bitcoin Recap using the user-provided Bitcoin news items.

Write in a formal, institutional tone for investors and portfolio managers.

Rules:
- Stay Bitcoin-focused.
- Do not invent facts beyond the provided news items.
- Format each news item as a plain line with the source attribution in parentheses.
- After the list, write a short "News Impact" paragraph that synthesizes the combined market relevance of the stories.
- Focus on broad implications for Bitcoin adoption, market positioning, investor sentiment, and near-term narrative.

Return only:

News Stories:
[Story 1]. (Reported By: [Source])
[Story 2]. (Reported By: [Source])
...

News Impact:
[One concise synthesis paragraph]
```

### 2. Weekly Bitcoin Recap Summary

```text
Generate the Weekly Bitcoin Recap Summary using `summary_table.csv`.

Write in a formal tone suitable for professional investors.

Rules:
- Use only values present in the summary table.
- Read `summary_table.csv` as a labeled table with `Metric`, `Value`, and `Category` columns.
- Do not introduce fields that are not available.
- Cover price, market capitalization, sats per dollar, supply, miner revenue, transaction volume, Bitcoin dominance, market sentiment, and valuation.
- If a field is blank in the summary table, omit commentary that depends on it rather than inventing a substitute.

Return a section titled:

Current State Of The Bitcoin Market

It should include:
- Market Activity
- On-Chain Activity
- Market Adoption

The narrative should explain the current market state using the summary table only.
```

### 3. Historical Performance

```text
Generate the Historical Performance section using `performance_table.csv`.

Write in a formal tone suitable for hedge fund managers and research-oriented investors.

Rules:
- Use the current combined performance table.
- Compare Bitcoin with the asset rows present in the file.
- Focus on 7 Day Return, MTD Return, YTD Return, 90 Day Return, and 90 Day BTC Correlation where helpful.
- Do not refer to assets or categories that are not in the current table.
- If Bitcoin is not the top performer, identify the top performer from the actual table.

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
- Use the most recent month-to-date value in the current year row.
- Compare the current month against the historical average and median rows where relevant.
- Explain the current month in context without overclaiming predictive power.

Return a section titled:

Bitcoin Monthly Heatmap Overview and Analysis

It should include:
- Report Date
- Time Context
- Monthly Heatmap Insights
- Current Data Interpretation
- Market Outlook for the Month
```

### 5. MTD Return Comparison

```text
Generate the Month-to-Date Return Comparison section using `mtd_return_comparison.csv`.

Rules:
- Use the current year row and the Median Projection row.
- Compare the current month return to the historical median path for the same point in the month.
- Keep the language data-driven.
- Do not assume additional rows exist beyond the current year and Median Projection rows.

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
- Base the analysis on the tabular open, high, low, and close values.
- Describe weekly range, directional bias, and the relationship between open and close.
- Use the week's high and low as natural reference points for resistance and support discussion.
- Do not mention chart images, image uploads, screenshots, or vision analysis.
- Keep scenario language grounded in the latest weekly range and price behavior.

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
- Use `Current Price` as the spot-price anchor for the section.
- Use the 1k and 5k bucket counts to describe how historically active the current area has been.
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
- Group horizons explicitly as `short-term = 1 day, 3 day, 7 day, 30 day`, `medium-term = 90 day and 1 Year`, and `long-term = 2 Year, 4 Year, 5 Year, 10 Year`.
- Highlight which entry periods are positive or negative.
- Explain what the spread of outcomes says about Bitcoin's short-term volatility and long-term return profile.
- Keep the language data-driven and avoid overstating precision.

Return a section titled:

Bitcoin ROI Profile

It should include:
- short-term entry outcomes
- medium-term entry outcomes
- long-term entry outcomes
- what the distribution says about risk and reward across holding periods
```

### 9. Difficulty / Hash Rate / Network Security Snapshot

```text
Generate the Difficulty / Hash Rate / Network Security Snapshot using `summary_table.csv` and `master_metrics_data.csv`.

Rules:
- Use the summary table as a labeled market-context table with `Metric`, `Value`, and `Category` columns.
- Use `master_metrics_data.csv` for the latest difficulty, difficulty_adjustment, hash_rate, miner-relevant context, and any recent trend visible in the latest rows.
- Frame the section as a current network-security snapshot, not as a dedicated difficulty-period report.
- Connect network conditions to market structure and miner incentives where the available data supports that framing.
- Do not imply interval-over-interval precision unless the source data directly supports it.

Return a section titled:

Difficulty / Hash Rate / Network Security Snapshot

It should include:
- current difficulty
- latest difficulty adjustment context
- current hash rate and recent direction
- what the data suggests about network security and miner conditions
```

### 10. YTD Return Comparison

```text
Generate the Year-to-Date Return Comparison section using `ytd_return_comparison.csv`.

Rules:
- Use the current year row and the Median Projection row.
- Compare current YTD return to the historical median path for the same point in the year.
- Use the median projection to discuss context, not certainty.
- Keep the output concise and analytical.

Return a section titled:

Bitcoin Year-to-Date Return Comparison

It should include:
- YTD Performance Snapshot
- Year End Price Scenario Analysis
- Observations and Outlook
```

### 11. Model Valuation

```text
Generate the Model Valuation section using `eoy_model_data.csv` and `master_metrics_data.csv`.

Rules:
- Use the latest available values for `price_close`, `realised_price`, `thermocap_price`, `200_day_ma_price_close`, and `Lagged_Energy_Value` when helpful.
- Use model multiples such as `200_day_multiple`, `thermocap_multiple`, and `mvrv_ratio` if they support the valuation framing.
- Focus on current model context rather than on any missing legacy valuation table.
- Explain what each model contributes without pretending any single model is definitive.

Return a section titled:

Bitcoin Model Valuation

It should include:
- current spot price versus key model prices
- model multiple context
- areas of discount or premium
- a concise valuation takeaway
```

### 12. On-Chain Valuation

```text
Generate the On-Chain Valuation section using `master_metrics_data.csv`.

Rules:
- Use the latest available values for on-chain valuation fields such as `nvt_price`, `nvt_price_adj`, `realised_price`, `thermocap_price`, and relevant multiples.
- Explain what each metric measures before interpreting it.
- Keep the analysis Bitcoin-only and data-grounded.
- Do not refer to a missing standalone on-chain valuation CSV.

Return a section titled:

Bitcoin On-Chain Valuation

It should include:
- key on-chain valuation benchmarks
- what those benchmarks imply about current market structure
- areas where spot price is above or below the relevant models
- a concise synthesis of the on-chain valuation picture
```

### 13. Relative Valuation

```text
Generate the Relative Valuation section using `relative_value_comparison.csv`.

Rules:
- Use the assets listed in the current file.
- Re-rank or restate assets only if required by the current data ordering.
- Explain which assets Bitcoin has already surpassed, which ones it is approaching, and which larger targets remain aspirational.
- Use the current "Market Cap BTC Price" and "BTC % Move to Marketcap BTC Price" fields to frame parity analysis.
- Do not introduce assets that are not in the CSV.

Return a section titled:

Bitcoin Relative Valuation Analysis

It should include:
- a compact parity table
- surpassed assets
- approaching valuation levels
- aspirational targets
```

### 14. Conclusion

```text
Write the conclusion for the Weekly Bitcoin Recap using the completed report sections below.

Rules:
- Summarize the major findings from all prior sections.
- Keep the conclusion forward-looking but grounded in the data already discussed.
- Do not introduce new facts.
- Maintain a professional Bitcoin-only tone.

Sections to synthesize:
- Weekly Bitcoin Recap Summary
- News Section
- Historical Performance
- Monthly Heat Map
- MTD Return Comparison
- Weekly BTC/USD OHLC Analysis
- Trading Range Analysis
- ROI Profile
- Difficulty / Hash Rate / Network Security Snapshot
- YTD Return Comparison
- Model Valuation
- On-Chain Valuation
- Relative Valuation

Return one concise conclusion paragraph.
```

## Final Assembly Notes

- Start with the News Section if news is provided.
- Follow the section order above.
- This recap now absorbs the trading-range, ROI, network-security, and valuation sections that previously lived in narrower standalone report variants.
- Keep headings stable from week to week.
- End with the conclusion only. No execution code, UI text, or marketing copy is part of this migrated spec.
