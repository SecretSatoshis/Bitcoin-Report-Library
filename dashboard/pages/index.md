---
# Evidence's preprocessor builds <title>, description, og: and twitter: tags
# from this block. Without a title here it emits <title>Evidence</title>, and a
# layout <svelte:head> cannot override it — the page renders last and wins.
# hide_title suppresses Evidence's own <h1> so the branded hero is the only one.
title: Bitcoin Market Dashboard | Secret Satoshis
description: Current Bitcoin market data, valuation models, on-chain conditions and cycle context, updated daily from an open pipeline.
hide_title: true
og:
  image: https://secretsatoshis.com/assets/images/social-card.jpg
---

<section class="dashboard-hero" aria-labelledby="dashboardHeroTitle" data-dashboard-date={data_date?.[0]?.date_iso ?? ''}>
  <div class="dashboard-hero-grid">
    <div class="dashboard-hero-copy">
      <p class="dashboard-eyebrow"><span class="brand-accent">//</span> Market Intelligence</p>
      <h1 id="dashboardHeroTitle">Bitcoin Market Dashboard<span class="brand-accent">.</span></h1>
      <p class="dashboard-hero-sub">A daily view of Bitcoin market performance, on-chain conditions, valuation, and network activity.</p>
      <p class="dashboard-hero-source">Explore daily Bitcoin market data and the models behind the analysis. <a href="https://secretsatoshis.github.io/Bitcoin-Report-Library/">View the source data.</a></p>
      <noscript>
        <p class="dashboard-hero-source">The tables and charts on this page are rendered in the browser and stay empty without JavaScript. Every figure they show is computed from an <a href="https://secretsatoshis.github.io/Bitcoin-Report-Library/">open Bitcoin data release</a>, published daily as CSV and readable directly.</p>
      </noscript>
    </div>
    <dl class="dashboard-hero-stats" aria-label="Dashboard status">
      <div>
        <dt>Latest data</dt>
        <dd><Value data={data_date} column=date_label /></dd>
      </div>
      <div>
        <dt>Refresh cadence</dt>
        <dd>Daily</dd>
      </div>
    </dl>
  </div>
</section>

## Bitcoin Snapshot

_Headline metrics — market, on-chain, and sentiment._

<div class="bitcoin-snapshot-cards">

<div class="newsletter-visual" data-newsletter-visual="bitcoin-snapshot-market-data">

### Market Data

<script>
  import PriceOutlookChart from '$lib/PriceOutlookChart.svelte';
  import {
    buildColorMap,
    buildLatestPoints,
    buildSeriesOptions,
    currentYearFrom,
    fmtUsd,
    withAggregates,
    yearCols,
  } from '$lib/seasonalChart.js';
  // Sparkline colour: green if the metric grew over the window, red if it shrank. Rows
  // are newest first, so row 0's pct_change covers the whole window.
  const POS = '#00FF88';
  const NEG = '#FF3B30';
  const FALLBACK = '#F7931A';
  // Heatmap cells: red losses, near-black flat, green gains.
  const HEATMAP_SCALE = ['#FF3B30', '#0A0A0A', '#00FF88'];
  $: priceColor     = btc_price?.length         ? (btc_price[0].pct_change         >= 0 ? POS : NEG) : FALLBACK;
  $: marketcapColor = btc_marketcap?.length     ? (btc_marketcap[0].pct_change     >= 0 ? POS : NEG) : FALLBACK;
  $: satsColor      = sats_per_dollar?.length   ? (sats_per_dollar[0].pct_change   <= 0 ? POS : NEG) : FALLBACK;
  $: supplyColor    = btc_supply?.length        ? (btc_supply[0].pct_change        >= 0 ? POS : NEG) : FALLBACK;
  $: revenueColor   = btc_miner_revenue?.length ? (btc_miner_revenue[0].pct_change >= 0 ? POS : NEG) : FALLBACK;
  $: volumeColor    = btc_tx_volume?.length     ? (btc_tx_volume[0].pct_change     >= 0 ? POS : NEG) : FALLBACK;
  // ─── Seasonal returns chart helpers ──────────────────────────────────
  // Dates and years come from the data, never the viewer's clock, which can be ahead of
  // the latest release.
  $: dataMonthName = data_date?.[0]?.month_name ?? '';
  $: dataYearLabel = data_date?.[0]?.year_label ?? '';

  $: mtdPlot = withAggregates(mtd_history, 'day');
  $: ytdPlot = withAggregates(ytd_history, 'day_of_year');

  $: mtdCurrentYear = currentYearFrom(mtd_history, 'day');
  $: ytdCurrentYear = currentYearFrom(ytd_history, 'day_of_year');

  $: mtdYears = [...yearCols(mtd_history, 'day'), 'Median', 'Average'];
  $: ytdYears = [...yearCols(ytd_history, 'day_of_year'), 'Median', 'Average'];
  $: mtdSeriesColors = buildColorMap(mtdYears, mtdCurrentYear);
  $: ytdSeriesColors = buildColorMap(ytdYears, ytdCurrentYear);

  $: mtdEchartsOptions = buildSeriesOptions(mtdYears, mtdCurrentYear);
  $: ytdEchartsOptions = buildSeriesOptions(ytdYears, ytdCurrentYear);

  $: mtdLatest = buildLatestPoints(mtdPlot, 'day', mtdCurrentYear);
  $: ytdLatest = buildLatestPoints(ytdPlot, 'day_of_year', ytdCurrentYear);

  // Label the forecast with its own outlook_year, so last year's levels cannot pass as current.
  $: outlookYear = price_outlook?.length ? String(price_outlook[0].outlook_year) : '';
  $: outlookYearMismatch = Boolean(outlookYear && dataYearLabel && outlookYear !== dataYearLabel);

  $: outlookCaseLevels = (price_outlook || [])
    .filter(level => level.type === 'case')
    .slice()
    .sort((a, b) => Number(a.price) - Number(b.price));
</script>

<Grid cols=3 gapSize=lg>
  <BigValue
    data={btc_price}
    value=price
    title="Bitcoin Price"
    fmt=usd0
    sparkline=date
    sparklineType=area
    sparklineYScale=true
    sparklineColor={priceColor}
    comparison=pct_change
    comparisonTitle="vs 30d ago"
    comparisonFmt=pct1
    description="BTC spot price (USD)."
  />
  <BigValue
    data={btc_marketcap}
    value=marketcap
    title="Bitcoin Market Cap"
    fmt='$#,##0.00"T"'
    sparkline=date
    sparklineType=area
    sparklineYScale=true
    sparklineColor={marketcapColor}
    comparison=pct_change
    comparisonTitle="vs 30d ago"
    comparisonFmt=pct1
    description="Supply × price, in trillions USD."
  />
  <BigValue
    data={sats_per_dollar}
    value=sats
    title="Sats Per Dollar"
    fmt=num0
    sparkline=date
    sparklineType=area
    sparklineYScale=true
    sparklineColor={satsColor}
    comparison=pct_change
    comparisonTitle="vs 30d ago"
    comparisonFmt=pct1
    downIsGood=true
    description="Satoshis per USD."
  />
</Grid>

</div>

---

### On-chain Data

<Grid cols=3 gapSize=lg>
  <BigValue
    data={btc_supply}
    value=supply
    title="Bitcoin Supply"
    fmt=num0
    sparkline=date
    sparklineType=area
    sparklineYScale=true
    sparklineColor={supplyColor}
    comparison=pct_change
    comparisonTitle="vs 30d ago"
    comparisonFmt=pct2
    description="BTC in circulation (cap 21M)."
  />
  <BigValue
    data={btc_miner_revenue}
    value=revenue
    title="Bitcoin Miner Revenue"
    fmt='$#,##0.00"M"'
    sparkline=date
    sparklineType=area
    sparklineYScale=true
    sparklineColor={revenueColor}
    comparison=pct_change
    comparisonTitle="vs 30d ago"
    comparisonFmt=pct1
    description="Miner rewards (24h, USD)."
  />
  <BigValue
    data={btc_tx_volume}
    value=volume
    title="Bitcoin Transaction Volume"
    fmt='$#,##0.00"B"'
    sparkline=date
    sparklineType=area
    sparklineYScale=true
    sparklineColor={volumeColor}
    comparison=pct_change
    comparisonTitle="vs 30d ago"
    comparisonFmt=pct1
    description="On-chain transfer volume (24h, USD)."
  />
</Grid>

---

### Investor Sentiment

<Grid cols=3 gapSize=lg>
  <BigValue
    data={btc_supply_in_profit}
    value=supply_in_profit
    title="Supply in Profit"
    fmt='#,##0.0"%"'
    description="Share of all bitcoin whose price today is above the price it last moved at."
  />
  <BigValue
    data={btc_sentiment}
    value=sentiment
    title="Fear & Greed"
    description="NUPL zone, from Capitulation to Euphoria / Greed, based on the 7-day average of holders' unrealized profit or loss."
  />
  <BigValue
    data={btc_valuation}
    value=valuation
    title="Bitcoin Valuation"
    description="Price against the power-law fair value: Undervalued, Below Fair Value, Above Fair Value, Overvalued or Extremely Overvalued."
  />
</Grid>

</div>

## Performance

_Compare Bitcoin and other assets’ returns across the same periods._

<div class="newsletter-visual" data-quarterly-visual="performance-indexes">

### Stock Market Index Performance

<DataTable data={equity_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
</DataTable>

</div>

<div class="newsletter-visual" data-quarterly-visual="performance-sectors">

### Sector Performance

<DataTable data={sector_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
</DataTable>

</div>

<div class="newsletter-visual" data-quarterly-visual="performance-macro">

### Macro Asset Class Performance

<DataTable data={macro_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
</DataTable>

</div>

<div class="newsletter-visual" data-quarterly-visual="performance-bitcoin">

### Bitcoin Industry Performance

<DataTable data={bitcoin_industry_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
</DataTable>

</div>

<div class="newsletter-visual" data-newsletter-visual="bitcoin-price">

## Bitcoin Price

_Price vs on-chain valuation models and moving averages._

<!-- HTML heading for a stable #price-outlook anchor; a markdown heading's id would be
     slugged from the unevaluated {outlookYear} expression. -->
<h3 class="markdown" id="price-outlook">Secret Satoshis {outlookYear} Price Outlook</h3>

{#if outlookYearMismatch}
<p class="price-chart-methodology" role="note">These levels are the {outlookYear} outlook; the data runs through {dataYearLabel}.</p>
{/if}

<div class="price-outlook-cases">
{#each outlookCaseLevels as c (c.name)}
  <div class="price-outlook-case" style="--case-color: {c.color}">
    <span class="case-label">{c.name}</span>
    <strong class="case-price">{fmtUsd(c.price)}</strong>
  </div>
{/each}
</div>

<PriceOutlookChart rows={btc_with_models} candles={price_candles} cases={outlookCaseLevels} outlookYear={outlookYear} reportDate={data_date?.[0]?.date_iso ?? ''} />

<p class="price-chart-methodology">Simple moving averages · 3-month = 90 daily closes · 1-year = 52 weeks / 364 daily closes · 200-week = 1,400 daily closes</p>

</div>

<div class="newsletter-visual" data-newsletter-visual="monthly-return-heatmap">

## Monthly Bitcoin Price Return Heatmap

_Monthly returns by year._

### Statistical Reference

<div class="monthly-heatmap-table">

<DataTable data={monthly_returns_agg} rows=all compact=true rowShading=false>
  <Column id=time title="Period" width=120 align=center />
  <Column id=Jan title="Jan" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Feb title="Feb" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Mar title="Mar" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Apr title="Apr" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=May title="May" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Jun title="Jun" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Jul title="Jul" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Aug title="Aug" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Sep title="Sep" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Oct title="Oct" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Nov title="Nov" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Dec title="Dec" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Yearly title="Yearly" fmt='#,##0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-80} colorMid={0} colorMax={150} align=center />
</DataTable>

</div>

### Historical Returns by Year

<div class="monthly-heatmap-table heatmap-historical">

<DataTable data={monthly_returns_years} rows=all compact=true rowShading=false>
  <Column id=time title="Year" width=120 align=center />
  <Column id=Jan title="Jan" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Feb title="Feb" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Mar title="Mar" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Apr title="Apr" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=May title="May" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Jun title="Jun" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Jul title="Jul" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Aug title="Aug" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Sep title="Sep" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Oct title="Oct" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Nov title="Nov" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Dec title="Dec" fmt='#,##0.0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Yearly title="Yearly" fmt='#,##0"%"' contentType=colorscale colorScale={HEATMAP_SCALE} colorMin={-80} colorMid={0} colorMax={150} align=center />
</DataTable>

</div>

</div>

## Seasonal Returns

_Compare this month’s and this year’s price paths with historical years._

<div class="newsletter-visual" data-newsletter-visual="seasonal-mtd">

<h3 class="markdown" id="mtd-returns-comparison">Bitcoin {dataMonthName} MTD Returns Comparison</h3>

<LineChart
  data={mtdPlot}
  x=day
  y={mtdYears}
  xAxisTitle="Day of Month"
  yAxisTitle="Indexed to Month Start ($)"
  yFmt=usd0
  lineWidth=1
  seriesColors={mtdSeriesColors}
  echartsOptions={mtdEchartsOptions}
  yGridlines=true
  xGridlines=false
  markers=false
  legend=true
  yScale=true
  chartAreaHeight={420}
>
  <ReferencePoint data={mtdLatest.current} x=x y=y label=label labelPosition=right symbolSize=4 fontSize=11 color="#F7931A" labelColor="#F7931A" symbolColor="#F7931A" />
  <ReferencePoint data={mtdLatest.average} x=x y=y label=label labelPosition=right symbolSize=4 fontSize=11 color="#00FF88" labelColor="#00FF88" symbolColor="#00FF88" />
</LineChart>

</div>

<div class="newsletter-visual" data-newsletter-visual="seasonal-ytd">

<h3 class="markdown" id="ytd-returns-comparison">Bitcoin {dataYearLabel} YTD Returns Comparison</h3>

<LineChart
  data={ytdPlot}
  x=day_of_year
  y={ytdYears}
  xAxisTitle="Day of Year"
  yAxisTitle="Indexed to Year Start ($)"
  yFmt=usd0
  lineWidth=1
  seriesColors={ytdSeriesColors}
  echartsOptions={ytdEchartsOptions}
  yGridlines=true
  xGridlines=false
  markers=false
  legend=true
  yScale=true
  chartAreaHeight={420}
>
  <ReferencePoint data={ytdLatest.current} x=x y=y label=label labelPosition=right symbolSize=4 fontSize=11 color="#F7931A" labelColor="#F7931A" symbolColor="#F7931A" />
  <ReferencePoint data={ytdLatest.average} x=x y=y label=label labelPosition=right symbolSize=4 fontSize=11 color="#00FF88" labelColor="#00FF88" symbolColor="#00FF88" />
</LineChart>

</div>

<div class="newsletter-visual" data-quarterly-visual="relative-valuation">

## Relative Valuation

_Bitcoin’s hypothetical price if its market cap matched each reference asset. These are comparison scenarios, not forecasts._

<DataTable data={rel_val} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=market_cap title="Market Cap (USD)" fmt='$#,##0.00"T"' align=center />
  <Column id=implied_price title="Hypothetical BTC Price (USD)" fmt=usd0 align=center />
  <Column
    id=implied_return
    title="Change from Current BTC Price (%)"
    fmt='#,##0"%"'
    contentType=delta
    chip=true
    align=center
  />
  <Column
    id=btc_pct_of_mcap
    title="BTC / Asset Market Cap (%)"
    contentType=bar
    fmt='#,##0.0"%"'
    barColor="#F7931A"
    align=right
  />
</DataTable>

</div>

## Network Fundamentals

_Network health, security & on-chain economics._

<DataTable data={fundamentals} rows=all rowShading=true groupBy=Category groupType=section subtotals=false groupNamePosition=top>
  <Column id=Category title="Category" />
  <Column id=Metric title="Metric" />
  <Column id="Current Value" title="Current Value" align=right />
  <Column id="7 Days Ago" title="7 Days Ago" align=right />
  <Column
    id="7 Day Change (%)"
    title="7d Change"
    fmt='#,##0.00"%"'
    contentType=delta
    chip=true
    align=right
  />
  <Column id="52W Low" title="52W Low" align=right />
  <Column id="52W High" title="52W High" align=right />
</DataTable>

## Bitcoin ROI by Time Frame

_Returns by holding period._

<DataTable data={roi_data} rows=all rowShading=true>
  <Column id=time_frame title="Period" />
  <Column
    id=roi_pct
    title="ROI"
    fmt='#,##0.0"%"'
    contentType=delta
    chip=true
    align=right
  />
  <Column id=start_price title="Start Price" fmt=usd0 align=right />
</DataTable>


<!-- ─────────────────────────────────────────────────────────────────────────
     Queries. Evidence resolves them wherever they sit in the file.
     ───────────────────────────────────────────────────────────────────────── -->

```sql data_date
-- Month and year labels come from the data, so headings match what the charts show.
select
  strftime(max(cast(date as date)), '%b %-d, %Y') as date_label,
  strftime(max(cast(date as date)), '%Y-%m-%d') as date_iso,
  strftime(max(cast(date as date)), '%B') as month_name,
  strftime(max(cast(date as date)), '%Y') as year_label
from bitcoin_report_library.summary_history
where Metric = 'Bitcoin Price USD'
```

```sql btc_price
-- Compare each value with the one exactly 30 calendar days earlier; the latest row is shown.
with src as (
  select cast(date as date) as date, Value as price
  from bitcoin_report_library.summary_history
  where Metric = 'Bitcoin Price USD'
)
select
  cur.date,
  cur.price,
  (cur.price - prior.price) / nullif(prior.price, 0) as pct_change
from src cur
left join src prior on prior.date = cur.date - interval 30 day
order by cur.date desc
```

```sql btc_marketcap
with src as (
  select cast(date as date) as date, Value / 1e12 as marketcap
  from bitcoin_report_library.summary_history
  where Metric = 'Bitcoin Marketcap'
)
select
  cur.date,
  cur.marketcap,
  (cur.marketcap - prior.marketcap) / nullif(prior.marketcap, 0) as pct_change
from src cur
left join src prior on prior.date = cur.date - interval 30 day
order by cur.date desc
```

```sql sats_per_dollar
with src as (
  select cast(date as date) as date, Value as sats
  from bitcoin_report_library.summary_history
  where Metric = 'Sats Per Dollar'
)
select
  cur.date,
  cur.sats,
  (cur.sats - prior.sats) / nullif(prior.sats, 0) as pct_change
from src cur
left join src prior on prior.date = cur.date - interval 30 day
order by cur.date desc
```

```sql btc_supply
with src as (
  select cast(date as date) as date, Value as supply
  from bitcoin_report_library.summary_history
  where Metric = 'Bitcoin Supply'
)
select
  cur.date,
  cur.supply,
  (cur.supply - prior.supply) / nullif(prior.supply, 0) as pct_change
from src cur
left join src prior on prior.date = cur.date - interval 30 day
order by cur.date desc
```

```sql btc_miner_revenue
with src as (
  select cast(date as date) as date, Value / 1e6 as revenue
  from bitcoin_report_library.summary_history
  where Metric = 'Bitcoin Miner Revenue'
)
select
  cur.date,
  cur.revenue,
  (cur.revenue - prior.revenue) / nullif(prior.revenue, 0) as pct_change
from src cur
left join src prior on prior.date = cur.date - interval 30 day
order by cur.date desc
```

```sql btc_tx_volume
with src as (
  select cast(date as date) as date, Value / 1e9 as volume
  from bitcoin_report_library.summary_history
  where Metric = 'Bitcoin Transaction Volume'
)
select
  cur.date,
  cur.volume,
  (cur.volume - prior.volume) / nullif(prior.volume, 0) as pct_change
from src cur
left join src prior on prior.date = cur.date - interval 30 day
order by cur.date desc
```

```sql btc_supply_in_profit
select CAST(Value AS DOUBLE) as supply_in_profit
from bitcoin_report_library.summary_table
where Metric = 'Bitcoin Supply in Profit'
```

```sql btc_sentiment
select Value as sentiment
from bitcoin_report_library.summary_table
where Metric = 'Bitcoin Market Sentiment'
```

```sql btc_valuation
select Value as valuation
from bitcoin_report_library.summary_table
where Metric = 'Bitcoin Valuation'
```

```sql equity_perf
select
  case
    when Asset = 'Bitcoin - [BTC]' then '<span style="color:#F7931A;font-weight:700;">Bitcoin - [BTC]</span>'
    else Asset
  end as Asset,
  Price as price,
  "7 Day Return (%)" as return_7d,
  "MTD Return (%)" as return_mtd,
  "YTD Return (%)" as return_ytd,
  "90 Day Return (%)" as return_90d
from bitcoin_report_library.performance_table
where Category = 'Equity Market Indexes'
   or Asset = 'Bitcoin - [BTC]'
order by
  case when Asset = 'Bitcoin - [BTC]' then 0 else 1 end,
  return_7d desc nulls last
```

```sql sector_perf
select
  case
    when Asset = 'Bitcoin - [BTC]' then '<span style="color:#F7931A;font-weight:700;">Bitcoin - [BTC]</span>'
    else Asset
  end as Asset,
  Price as price,
  "7 Day Return (%)" as return_7d,
  "MTD Return (%)" as return_mtd,
  "YTD Return (%)" as return_ytd,
  "90 Day Return (%)" as return_90d
from bitcoin_report_library.performance_table
where Category = 'Sectors'
   or Asset = 'Bitcoin - [BTC]'
order by
  case when Asset = 'Bitcoin - [BTC]' then 0 else 1 end,
  return_7d desc nulls last
```

```sql macro_perf
select
  case
    when Asset = 'Bitcoin - [BTC]' then '<span style="color:#F7931A;font-weight:700;">Bitcoin - [BTC]</span>'
    else Asset
  end as Asset,
  Price as price,
  "7 Day Return (%)" as return_7d,
  "MTD Return (%)" as return_mtd,
  "YTD Return (%)" as return_ytd,
  "90 Day Return (%)" as return_90d
from bitcoin_report_library.performance_table
where Category = 'Macro Asset Classes'
   or Asset = 'Bitcoin - [BTC]'
order by
  case when Asset = 'Bitcoin - [BTC]' then 0 else 1 end,
  return_7d desc nulls last
```

```sql bitcoin_industry_perf
select
  case
    when Asset = 'Bitcoin - [BTC]' then '<span style="color:#F7931A;font-weight:700;">Bitcoin - [BTC]</span>'
    else Asset
  end as Asset,
  Price as price,
  "7 Day Return (%)" as return_7d,
  "MTD Return (%)" as return_mtd,
  "YTD Return (%)" as return_ytd,
  "90 Day Return (%)" as return_90d
from bitcoin_report_library.performance_table
where Category = 'Bitcoin Industry Performance'
   or Asset = 'Bitcoin - [BTC]'
order by
  case when Asset = 'Bitcoin - [BTC]' then 0 else 1 end,
  return_7d desc nulls last
```

```sql btc_with_models
-- Moving averages are precomputed by the Report Library, so the chart and newsletter match.
select
  cast(date as date) as date,
  "BTC Price",
  "Realized Price",
  "STH Realized Price",
  "3x Realized Price",
  "3-month MA",
  "1-year MA",
  "200-week MA"
from bitcoin_report_library.onchain_price_models
order by date
```

```sql price_candles
select * from bitcoin_report_library.bitcoin_candles
order by interval, period_start
```

```sql price_outlook
select
  label as name,
  label || ' - $' || format('{:,.0f}', cast(price as double)) as label,
  cast(price as double) as price,
  type,
  color,
  outlook_year
from bitcoin_report_library.price_outlook
```

```sql rel_val
with parsed as (
  select
    Asset,
    "Market Cap (USD)" / 1e12 as market_cap,
    "Market Cap BTC Price" as implied_price,
    "BTC % Move to Marketcap BTC Price" as implied_return
  from bitcoin_report_library.relative_value_comparison
),
btc_mcap as (
  select market_cap as mc from parsed where Asset = 'Bitcoin'
)
select
  case
    when Asset = 'Bitcoin'
      then '<strong style="color:#F7931A;">Bitcoin</strong>'
    else Asset
  end as Asset,
  market_cap,
  implied_price,
  implied_return,
  case
    when Asset = 'Bitcoin' then null
    else (select mc from btc_mcap) / nullif(market_cap, 0) * 100
  end as btc_pct_of_mcap
from parsed
order by market_cap desc
```

```sql monthly_returns_agg
select
  case when time = '4-Year Average' then '4 Year Avg' else time end as time,
  Jan, Feb, Mar, Apr, May, Jun,
  Jul, Aug, Sep, Oct, Nov, Dec,
  Yearly
from bitcoin_report_library.monthly_heatmap_data
where time in ('Average', 'Median', '4-Year Average')
order by
  case time
    when 'Average' then 0
    when 'Median' then 1
    when '4-Year Average' then 2
  end
```

```sql monthly_returns_years
-- Label the newest year as partial only while its report cutoff precedes December 31.
with years as (
  select *, try_cast(time as integer) as yr
  from bitcoin_report_library.monthly_heatmap_data
  where try_cast(time as integer) is not null
)
select
  case when yr = (select max(yr) from years) and (select strftime(max(cast(date as date)), '%m-%d') from bitcoin_report_library.summary_history) <> '12-31' then time || ' (YTD)' else time end as time,
  Jan, Feb, Mar, Apr, May, Jun,
  Jul, Aug, Sep, Oct, Nov, Dec,
  Yearly
from years
order by yr desc
```

```sql roi_data
select
  "Time Frame" as time_frame,
  "ROI (%)" as roi_pct,
  "BTC Price" as start_price
from bitcoin_report_library.roi_table
```

```sql fundamentals
select
  Section as Category,
  Metric,
  "Current Value",
  "7 Days Ago",
  "7 Day Change (%)",
  Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday,
  "52W Low",
  "52W High"
from bitcoin_report_library.fundamentals_table
```

```sql mtd_history
-- See ytd_history.
select * exclude ("Median", "Average")
from bitcoin_report_library.mtd_returns_history
order by day
```

```sql ytd_history
-- The CSV's Median/Average include hidden and current years, so they are dropped here and
-- recomputed by the chart (components/seasonalChart.js) over the visible past years.
select * exclude ("Median", "Average")
from bitcoin_report_library.ytd_returns_history
order by day_of_year
```
