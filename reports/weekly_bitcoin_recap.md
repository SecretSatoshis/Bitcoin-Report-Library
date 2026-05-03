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
4. Bitcoin Price Analysis
5. Bitcoin Price Outlook
6. Relative Valuation
7. Conclusion

## Data Inputs

| Section | Input Type | Local Source | Published Source |
|---|---|---|---|
| News Section | Web | NA | `https://bitcoinmagazine.com/.rss/full/`, `https://www.coindesk.com/arc/outboundfeeds/rss/`, `https://feeds.bloomberg.com/markets/news.rss`, `https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114`, `https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664`, `https://www.ft.com/rss/home`, `https://finance.yahoo.com/news/rssindex` |
| Weekly Bitcoin Recap Summary | CSV | `csv/summary_table.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/summary_table.csv` |
| Historical Performance | CSV | `csv/performance_table.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/performance_table.csv` |
| Bitcoin Price Analysis | CSV | `csv/ohlc_data.csv`, `csv/1k_bucket_table.csv`, `csv/5k_bucket_table.csv`, `csv/onchain_price_models.csv`, `csv/price_outlook.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/ohlc_data.csv`, `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/1k_bucket_table.csv`, `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/5k_bucket_table.csv`, `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/onchain_price_models.csv`, `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/price_outlook.csv` |
| Bitcoin Price Outlook | CSV | `csv/monthly_heatmap_data.csv`, `csv/mtd_return_comparison.csv`, `csv/ytd_return_comparison.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/monthly_heatmap_data.csv`, `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/mtd_return_comparison.csv`, `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/ytd_return_comparison.csv` |
| Relative Valuation | CSV | `csv/relative_value_comparison.csv` | `https://secretsatoshis.github.io/Bitcoin-Report-Library/csv/relative_value_comparison.csv` |

## Global Execution Rules

- Use one derived report date from the latest available mapped report data.
- For each data-backed section, use the latest row or value at or before the report date.
- When a section maps multiple CSV inputs, use all mapped files named for that section and keep each table's role distinct.
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

Major Financial Outlets:
4. Bloomberg Markets: https://feeds.bloomberg.com/markets/news.rss
5. Financial Times Home: https://www.ft.com/rss/home

Rules:
- For Major Financial Outlets, filter for Bitcoin/crypto/macro
- Use only stories published or posted within the report week.
- Include broader crypto, macro, regulatory, or market-structure stories only if they materially affect Bitcoin or the broader digital asset market.
- Select one Top News Story Of The Week.
- Then select 3 to 5 Bitcoin News stories
- Then select 2 to 5 Market News stories: macro, regulatory, or financial-system stories that are materially Bitcoin-relevant.
- Do not pad the section with weak stories.
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
[One concise synthesis two to three sentences covering why the selected stories matter, the themes connecting them, and their combined relevance to Bitcoin adoption, positioning, sentiment, and current market narrative. No price forecasts.]
```

### 2. Weekly Bitcoin Recap Summary

```text
Generate the Weekly Bitcoin Recap Summary using `summary_table.csv`.

Rules:
- Use only values present in the summary table.
- Read `summary_table.csv` as a labeled table with `Metric`, `Value`, and `Category` columns.
- Use the `Category` column to group metrics into Market Data, On-chain Data, and Investor Sentiment.
- Format summary values using the global formatting rules.
- Keep this section as a concise current-state snapshot; do not introduce technical levels, news events, or performance comparisons from other sections.
- If a metric is missing, omit it rather than inventing a substitute.

Return the section using this repeatable scaffold. Replace every bracketed field with the mapped data or a concise data-backed interpretation.

Current State Of The Bitcoin Market

Bitcoin is trading at [Bitcoin Price USD], with a market capitalization of [Bitcoin Marketcap]. At the current price, one dollar buys approximately [Sats Per Dollar] sats.

Current Bitcoin supply is [Bitcoin Supply], with miners generating [Bitcoin Miner Revenue] in revenue and the network facilitating [Bitcoin Transaction Volume] in transaction volume.

At its current market capitalization, Bitcoin holds [Bitcoin Dominance] of the total crypto market. Current market sentiment is [Bitcoin Market Sentiment], while the valuation reading is [Bitcoin Valuation].

```

### 3. Historical Performance

```text
Generate the Historical Performance section using `performance_table.csv`.

Rules:
- Use the current combined performance table.
- Compare Bitcoin with the asset rows present in the file.
- Use `7 Day Return` as the primary weekly ranking metric.
- Use MTD Return, YTD Return, 90 Day Return, and 90 Day BTC Correlation as secondary context where helpful.
- Interpret all return columns as percentage-point values. For example, `-4.21` means `-4.21%`, not `-0.0421%`.
- Use the `Category` column to group assets into Equity Market Indexes, Sectors, Macro Asset Classes, and Bitcoin Industry Performance.
- Format `90 Day BTC Correlation` as a coefficient from `-1.00` to `1.00`, not as a percentage.
- Do not refer to assets or categories that are not in the current table.
- Keep the section focused on relative weekly performance and only use longer-period returns or correlation when they add useful context.

Return the section using this repeatable scaffold. Replace every bracketed field with the mapped data or a concise data-backed interpretation.

Historical Performance

Bitcoin Performance Snapshot
Bitcoin returned [Bitcoin 7 Day Return] over the past week, compared with [Bitcoin MTD Return] month-to-date, [Bitcoin YTD Return] year-to-date, and [Bitcoin 90 Day Return] over the past 90 days.

[One sentence explaining whether Bitcoin's weekly move shows strength, weakness, or stabilization relative to its own month-to-date, year-to-date, and 90-day performance.]

Equity Market Indexes
Among equity indexes, [Top Equity Index] led the group with a [Top Equity Index 7 Day Return] weekly return, while [Lagging Equity Index] was the weakest at [Lagging Equity Index 7 Day Return]. 

[One to two sentences explaining whether Bitcoin outperformed or lagged broad equity benchmarks and whether equity correlation suggests risk-on/risk-off alignment.]

Sector Performance
Among sectors, [Top Sector] led with a [Top Sector 7 Day Return] weekly return, while [Lagging Sector] lagged at [Lagging Sector 7 Day Return].

[One sentence explaining whether Bitcoin's weekly performance was closer to higher-beta sectors, defensive sectors, or broad market behavior.]

Macro Asset Classes
Across macro assets, [Top Macro Asset] returned [Top Macro Asset 7 Day Return], while [Lagging Macro Asset] returned [Lagging Macro Asset 7 Day Return]. Bitcoin's 90-day correlation to key macro assets ranged from [Lowest Macro BTC Correlation] to [Highest Macro BTC Correlation].

[One to two sentences explaining how Bitcoin performed relative to dollar, gold, bond, or commodity proxies available in the table.]

Bitcoin Industry Performance
Within Bitcoin-related equities and industry proxies, [Top Bitcoin Industry Asset] returned [Top Bitcoin Industry Asset 7 Day Return], while [Lagging Bitcoin Industry Asset] returned [Lagging Bitcoin Industry Asset 7 Day Return].

[One to two sentences explaining whether Bitcoin-related equities confirmed, amplified, or diverged from Bitcoin's weekly move.]

Cross-Asset Summary
[One to two sentence synthesis explaining where Bitcoin stood in the broader cross-asset performance table this week. Mention the top overall weekly performer if it was not Bitcoin, and keep the analysis tied to the assets and categories present in `performance_table.csv`.]
```

### 4. Bitcoin Price Analysis

```text
Generate the Bitcoin Price Analysis section using `ohlc_data.csv`, `1k_bucket_table.csv`, `5k_bucket_table.csv`, `onchain_price_models.csv`, and `price_outlook.csv`.

Rules:
- Use the latest completed weekly OHLC row from `ohlc_data.csv`; treat `Time` as the week-start timestamp and calculate weekly return and range from the row.
- Use the current 1K and 5K ranges from the bucket tables as background context for the current price zone, not as the main subject.
- Use `price_outlook.csv` for explicit support, resistance, and bull/base/bear scenario anchors.
- Use the latest on-chain model row at or before the report date for relevant model context, but include only the models that improve the analysis.
- Keep the section analytical and conditional: do not invent levels, do not force every input into the prose, and do not frame scenario anchors as forecasts.

Return the section using this repeatable scaffold. Replace every bracketed field with the mapped data or a concise data-backed interpretation.

Bitcoin Price Analysis

[Week Start Date] to [Week End Date]: Bitcoin opened at [Open], traded between [Low] and [High], and closed at [Close]. The weekly return was [Weekly Return %], with a weekly trading range of [Weekly Range $] ([Weekly Range %]).

[One sentence interpreting the weekly candle: close versus open, high/low behavior, whether the week showed upside follow-through, downside pressure, consolidation, or range-bound trading.]

Bitcoin is currently trading around [Current Price], placing it in the [Current 1K Bucket] short-term range and the broader [Current 5K Bucket] range.

[One to two sentences using the 1K and 5K bucket counts as background context for how familiar or notable the current price zone has been historically. Do not quote both bucket counts unless they materially improve the analysis. Do not describe bucket counts as volume, liquidity, probability, or forecast odds.]

Nearest support levels below current price are [Nearest Support 1], [Nearest Support 2], and [Nearest Support 3 if available]. Nearest resistance levels above current price are [Nearest Resistance 1], [Nearest Resistance 2], and [Nearest Resistance 3 if available].

[One to two sentences explaining how these levels frame the current setup. Use explicit support/resistance rows from `price_outlook.csv`; do not invent levels.]

Relevant on-chain model levels are [Relevant On-Chain Model 1], [Relevant On-Chain Model 2 if useful], and [Relevant On-Chain Model 3 if useful].

[One to two sentences explaining what the relevant on-chain models say about current price. Use `Electricity Cost` and `Realized Price` for deep value/support context, `STH Realized Price` for newer-entrant cost basis, and `3x Realized Price` for upper-range valuation context. Do not force every model into the response if it is not close to price or otherwise relevant.]

[Two to three sentence synthesis summarizing weekly price action, trading-range context, support/resistance, relevant on-chain level context, and where current price stands relative to the year-end bull/base/bear scenario anchors from `price_outlook.csv`. Integrate [Bull Case Price], [Base Case Price], and [Bear Case Price] naturally into the outlook instead of listing them. Explain what would need to change for price to move toward the bull case, what conditions would keep the base case intact, and what breakdown would make the bear case more relevant. Keep this as a coherent review and conditional outlook.]
```

### 5. Bitcoin Price Outlook

```text
Generate the Bitcoin Price Outlook section using `monthly_heatmap_data.csv`, `mtd_return_comparison.csv`, and `ytd_return_comparison.csv`.

Rules:
- Use `monthly_heatmap_data.csv` for historical monthly return context; interpret monthly heatmap values as decimal returns.
- Use the latest nonblank current-month value in the current year row, then compare it with the same month's `Average`, `Median`, and `4-Year Average` rows.
- Use `mtd_return_comparison.csv` to compare the current month-to-date path with historical median and average paths when available, including implied month-end benchmarks.
- Use `ytd_return_comparison.csv` to compare the current year-to-date path with historical median and average paths when available, including implied year-end benchmarks.
- Treat `Return (%)` and `Report Date Return (%)` in the MTD/YTD comparison files as percentage-point values.
- Frame median, average, and historical path values as context and scenario benchmarks, not forecasts.
- Keep the section focused on seasonal/historical return context, monthly trajectory, and longer-term yearly tracking.

Return the section using this repeatable scaffold. Replace every bracketed field with the mapped data or a concise data-backed interpretation.

Bitcoin Price Outlook

Historical Monthly Return Analysis
Bitcoin's current [Current Month] return is [Current Month Return], compared with a historical average of [Current Month Average Return], a historical median of [Current Month Median Return], and a 4-year average of [Current Month 4-Year Average Return] for the same month.

[One to two sentences explaining how the current month is tracking versus its historical monthly return profile. Focus on whether performance is above, below, or near historical average/median/4-year context, and what that says about current seasonality without forecasting.]

Month-to-Date Path
Bitcoin is currently [Current MTD Return] month-to-date, compared with the historical median path of [Median MTD Report Date Return] and, if available, the historical average path of [Average MTD Report Date Return] at this point in the month. The historical median full-month path would imply a month-end return of [Median Full-Month Return] and a month-end price benchmark of [Median Month-End Price].

[One to two sentences explaining whether Bitcoin is running ahead of, behind, or near the historical month-to-date path. Tie the current return to what historical median and average paths imply for month-end performance, while keeping the analysis conditional.]

Year-to-Date Path
Bitcoin is currently [Current YTD Return] year-to-date, compared with the historical median year-to-date path of [Median YTD Report Date Return] and, if available, the historical average path of [Average YTD Report Date Return] at this point in the year. The historical median full-year path would imply a year-end return of [Median Full-Year Return] and a year-end price benchmark of [Median Year-End Price].

[One to two sentences explaining how Bitcoin is tracking on a longer-term trajectory versus historical yearly paths. Discuss whether current performance is ahead of, behind, or near the median/average path and what that means for the broader yearly outlook.]

Outlook Summary
[Two to three sentence synthesis connecting monthly seasonality, the current month-to-date path, and the year-to-date trajectory. Explain whether near-term monthly performance is reinforcing or diverging from the longer-term yearly path. Keep the tone analytical and scenario-based, not predictive.]
```

### 6. Relative Valuation

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

Return the section using this repeatable scaffold. Replace every bracketed field with the mapped data or a concise data-backed interpretation.

Bitcoin Relative Valuation Analysis

Bitcoin's current valuation is [Bitcoin Market Cap], which corresponds to a BTC price of [Bitcoin Market Cap BTC Price]. In the relative valuation table, Bitcoin sits between [Nearest Larger Asset] at [Nearest Larger Asset Market Cap] and [Nearest Smaller Asset] at [Nearest Smaller Asset Market Cap].

[One sentences explaining where Bitcoin currently ranks in the table and what that says about its scale versus the listed assets and monetary benchmarks.]

Bitcoin has already surpassed [Surpassed Asset 1], [Surpassed Asset 2], and [Surpassed Asset 3 if available] by market capitalization.

[One sentences explaining the significance of the surpassed assets. Use negative `BTC % Move to Marketcap BTC Price` values as evidence that Bitcoin is already above those parity levels.]

The nearest larger valuation benchmarks are [Approaching Asset 1], [Approaching Asset 2], and [Approaching Asset 3 if available]. Reaching those parity levels would imply BTC prices of [Approaching Asset 1 BTC Price], [Approaching Asset 2 BTC Price], and [Approaching Asset 3 BTC Price], requiring moves of [Approaching Asset 1 BTC Move], [Approaching Asset 2 BTC Move], and [Approaching Asset 3 BTC Move].

[One to two sentences explaining which larger benchmarks are closest and how far Bitcoin is from those parity levels. Frame them as valuation comparisons, not targets.]

Relative Valuation Summary
[One to two sentence synthesis explaining what Bitcoin has already overtaken, what it is closest to approaching, and which larger benchmarks remain distant. Keep the analysis comparative and valuation-based, not predictive.]

```

### 7. Conclusion

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
- Bitcoin Price Analysis
- Bitcoin Price Outlook
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
