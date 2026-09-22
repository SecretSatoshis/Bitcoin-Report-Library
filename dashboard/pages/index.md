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
  // Trend-based sparkline colors: green if metric grew over the window, red if it shrank.
  // Data is sorted DESC so row 0 is "today" — pct_change there reflects the full window.
  const POS = '#00FF88';
  const NEG = '#FF3B30';
  const FALLBACK = '#F7931A';
  $: priceColor     = btc_price?.length         ? (btc_price[0].pct_change         >= 0 ? POS : NEG) : FALLBACK;
  $: marketcapColor = btc_marketcap?.length     ? (btc_marketcap[0].pct_change     >= 0 ? POS : NEG) : FALLBACK;
  $: satsColor      = sats_per_dollar?.length   ? (sats_per_dollar[0].pct_change   <= 0 ? POS : NEG) : FALLBACK;
  $: supplyColor    = btc_supply?.length        ? (btc_supply[0].pct_change        >= 0 ? POS : NEG) : FALLBACK;
  $: revenueColor   = btc_miner_revenue?.length ? (btc_miner_revenue[0].pct_change >= 0 ? POS : NEG) : FALLBACK;
  $: volumeColor    = btc_tx_volume?.length     ? (btc_tx_volume[0].pct_change     >= 0 ? POS : NEG) : FALLBACK;
  // ─── Seasonal returns chart helpers ──────────────────────────────────
  // Every date label and series list is derived from the data, never from the
  // viewer's clock. A browser-clock year flips on Jan 1 before the pipeline has
  // published a column for it, which silently drops the current-year styling and
  // annotation; a browser-clock month is wrong for anyone whose local date is
  // ahead of the 00:30 UTC refresh.
  $: dataMonthName = data_date?.[0]?.month_name ?? '';
  $: dataYearLabel = data_date?.[0]?.year_label ?? '';

  // Years hidden from the seasonal charts. 2017's magnitude compresses every other
  // year into a flat band at the bottom of the plot. It is excluded from the chart
  // only — it stays in the CSV, and the Median/Average lines are recomputed below
  // over the visible historical years. The current year is plotted separately but
  // excluded from those reference aggregates while it is still incomplete.
  const HIDDEN_YEARS = ['2017'];

  const _isYearCol = (c) => /^\d{4}$/.test(c);

  // Off-white for Median, cypherpunk green for Average, Bitcoin-orange for current
  // year. Those three are reserved, so the historical palette must avoid orange and
  // green entirely or a past year reads as this year.
  const _medianColor = '#e4e4ef';                         // brand text (legible on dark)
  const _averageColor = '#00FF88';
  const _currentColor = '#F7931A';

  // Historical years use one cool-blue recency ramp: the oldest visible year is
  // darkest and the newest is brightest. This keeps every trajectory on the chart
  // without suggesting that each year is a separate category. Exact year identity
  // comes from the interactive legend and hover focus, so shade is not the only cue.
  function _historicalShade(index, total) {
    const t = total <= 1 ? 1 : index / (total - 1);
    const saturation = Math.round(44 + (t * 28));
    const lightness = Math.round(34 + (t * 40));
    return `hsl(214, ${saturation}%, ${lightness}%)`;
  }

  function _yearCols(rows, xKey, { includeHidden = false } = {}) {
    if (!rows?.length) return [];
    return Object.keys(rows[0])
      .filter(c => c !== xKey && _isYearCol(c))
      .filter(c => includeHidden || !HIDDEN_YEARS.includes(c))
      .sort();
  }

  // The current year is the newest year column present in the data — including a
  // hidden one, so the label stays honest even if the newest year were hidden.
  function _currentYearFrom(rows, xKey) {
    const all = _yearCols(rows, xKey, { includeHidden: true });
    return all.length ? all[all.length - 1] : '';
  }

  // Recompute Median/Average across the visible *historical* years. The newest year
  // remains plotted, but is excluded from the aggregates while it is incomplete.
  // The CSV ships precomputed columns covering every year including hidden/current
  // ones, so those aggregate columns are deliberately ignored here.
  function _withAggregates(rows, xKey) {
    const plottedYears = _yearCols(rows, xKey);
    if (!rows?.length || !plottedYears.length) return [];
    const currentYear = _currentYearFrom(rows, xKey);
    const historicalYears = plottedYears.filter(y => y !== currentYear);
    return rows.map(r => {
      const out = { [xKey]: r[xKey] };
      for (const y of plottedYears) out[y] = r[y];
      const vals = historicalYears
        .map(y => r[y])
        .filter(v => v != null && !isNaN(v))
        .map(Number)
        .sort((a, b) => a - b);
      if (vals.length) {
        const mid = Math.floor(vals.length / 2);
        out.Median = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
        out.Average = vals.reduce((s, v) => s + v, 0) / vals.length;
      } else {
        out.Median = null;
        out.Average = null;
      }
      return out;
    });
  }

  function _buildColorMap(years, currentYear) {
    // Colour follows the year, not its position in the list: the newest historical
    // years take the first palette slots, so a year keeps its hue across both charts
    // and does not repaint when the series count changes.
    const historical = years
      .filter(n => _isYearCol(n) && n !== currentYear)
      .sort();
    const shades = new Map(
      historical.map((yr, i) => [yr, _historicalShade(i, historical.length)])
    );
    return Object.fromEntries(
      years.map(name => [
        name,
        name === 'Median'  ? _medianColor :
        name === 'Average' ? _averageColor :
        name === currentYear ? _currentColor :
        shades.get(name) ?? _medianColor
      ])
    );
  }

  // Day 366 exists only in leap years, so its Median/Average would be computed from
  // leap years alone — a phantom spike at the right edge. Cap the YTD chart at 365.
  $: mtdPlot = _withAggregates(mtd_history, 'day');
  $: ytdPlot = _withAggregates((ytd_history || []).filter(r => r.day_of_year <= 365), 'day_of_year');

  $: mtdCurrentYear = _currentYearFrom(mtd_history, 'day');
  $: ytdCurrentYear = _currentYearFrom(ytd_history, 'day_of_year');

  $: mtdYears = [..._yearCols(mtd_history, 'day'), 'Median', 'Average'];
  $: ytdYears = [..._yearCols(ytd_history, 'day_of_year'), 'Median', 'Average'];
  $: mtdSeriesColors = _buildColorMap(mtdYears, mtdCurrentYear);
  $: ytdSeriesColors = _buildColorMap(ytdYears, ytdCurrentYear);

  // Per-series presentation: recent historical years gain a little weight and
  // opacity, while current year + Median + Average remain the strongest references.
  // Hover restores a historical line to full opacity and reveals its year at the
  // endpoint; the scrollable legend still names every year and can toggle any line.
  function _buildSeriesOptions(years, currentYear) {
    const historical = years
      .filter(name => _isYearCol(name) && name !== currentYear)
      .sort();
    const historicalRank = new Map(historical.map((yr, i) => [yr, i]));
    const legendData = [
      currentYear,
      ...historical.slice().reverse(),
      'Median',
      'Average'
    ].filter(Boolean);

    return {
      legend: {
        show: true,
        type: 'scroll',
        data: legendData,
        top: 4,
        left: 'center',
        right: 20,
        itemWidth: 18,
        itemHeight: 3,
        itemGap: 14,
        textStyle: {
          color: '#9090a8',
          fontFamily: 'JetBrains Mono',
          fontSize: 10
        },
        pageIconColor: '#F7931A',
        pageIconInactiveColor: '#3a3a50',
        pageTextStyle: { color: '#9090a8' }
      },
      grid: { top: 62, containLabel: true },
      tooltip: { trigger: 'axis', confine: true },
      series: years.map(name => {
        const wide = (name === 'Median' || name === 'Average' || name === currentYear);
        const rank = historicalRank.get(name);
        const isHistorical = rank != null;
        const recency = rank == null || historical.length <= 1
          ? 1
          : rank / (historical.length - 1);
        const historicalOpacity = 0.76 + (recency * 0.16);
        const historicalColor = isHistorical
          ? _historicalShade(rank, historical.length)
          : undefined;
        const lineStyle = {
          width: wide ? 2.5 : 1 + (recency * 0.7),
          opacity: isHistorical ? historicalOpacity : 1
        };
        if (name === 'Median') lineStyle.type = 'dashed';
        return {
          z: wide ? 3 : 1,
          lineStyle,
          triggerEvent: isHistorical ? 'line' : false,
          endLabel: isHistorical ? { show: false } : undefined,
          emphasis: {
            focus: 'series',
            lineStyle: { width: wide ? 3.5 : 2.5, opacity: 1 },
            endLabel: isHistorical ? {
              show: true,
              formatter: '{a}',
              color: historicalColor,
              backgroundColor: 'rgba(8, 8, 12, 0.9)',
              borderRadius: 3,
              padding: [3, 5],
              align: 'right',
              distance: 4,
              fontFamily: 'JetBrains Mono',
              fontSize: 10,
              fontWeight: 600
            } : undefined
          },
          blur: { lineStyle: { opacity: 0.12 } }
        };
      })
    };
  }
  $: mtdEchartsOptions = _buildSeriesOptions(mtdYears, mtdCurrentYear);
  $: ytdEchartsOptions = _buildSeriesOptions(ytdYears, ytdCurrentYear);

  function _fmtUsd(n) {
    if (n == null || isNaN(n)) return '';
    return '$' + Math.round(Number(n)).toLocaleString();
  }
  function _buildLatestPoints(rows, xKey, currentYear) {
    if (!rows?.length || !currentYear) return { current: [], average: [] };
    // last row where current year col is not null (= today)
    let currentRow = null;
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i][currentYear] != null) { currentRow = rows[i]; break; }
    }
    // last row of the full Average series (end of month / end of year)
    let endRow = null;
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i]['Average'] != null) { endRow = rows[i]; break; }
    }
    return {
      current: currentRow ? [{ x: currentRow[xKey], y: currentRow[currentYear], label: `${currentYear} · ${_fmtUsd(currentRow[currentYear])}` }] : [],
      average: endRow     ? [{ x: endRow[xKey],     y: endRow['Average'], label: `Average · ${_fmtUsd(endRow['Average'])}` }]  : [],
    };
  }
  $: mtdLatest = _buildLatestPoints(mtdPlot, 'day', mtdCurrentYear);
  $: ytdLatest = _buildLatestPoints(ytdPlot, 'day_of_year', ytdCurrentYear);

  // Bitcoin price history is calculated in SQL before selecting a display range,
  // so moving averages keep their full lookback at the left edge of every view.
  let priceChartYears = 4;
  $: priceChartXMax = dataYearLabel
    ? new Date(`${dataYearLabel}-12-31T00:00:00Z`)
    : undefined;
  $: priceChartXMin = (() => {
    if (!data_date?.[0]?.date_iso) return undefined;
    const cutoff = new Date(`${data_date[0].date_iso}T00:00:00Z`);
    cutoff.setUTCFullYear(cutoff.getUTCFullYear() - priceChartYears);
    return cutoff;
  })();
  $: priceChartRows = (btc_with_models || []).filter(
    row => !priceChartXMin || new Date(row.date) >= priceChartXMin
  );
  $: outlookCaseLevels = (price_outlook || [])
    .filter(level => level.type === 'case')
    .slice()
    .sort((a, b) => Number(a.price) - Number(b.price));

  const _modelMeta = {
    'BTC Price':          { color: '#F7931A', label: 'BTC Price' },
    'Realized Price':     { color: '#2962FF', label: 'Realized' },
    'STH Realized Price': { color: '#E040FB', label: 'STH Realized' },
    '3x Realized Price':  { color: '#8B5E34', label: '3× Realized' },
    '3-month MA':         { color: '#7FDF83', label: '3-month MA' },
    '1-year MA':          { color: '#FF8DA1', label: '1-year MA' },
    '200-week MA':        { color: '#B3A4FF', label: '200-week MA' },
  };
  const priceModelKeys = Object.keys(_modelMeta);
  let priceModelSelected = Object.fromEntries(priceModelKeys.map(key => [key, true]));
  function togglePriceModel(key) {
    priceModelSelected = { ...priceModelSelected, [key]: !priceModelSelected[key] };
  }
  $: priceChartOptions = {
    xAxis: { min: priceChartXMin?.getTime(), max: priceChartXMax?.getTime() },
    legend: { show: false, data: priceModelKeys, selected: priceModelSelected },
    series: priceModelKeys.map((key, index) => ({
      name: key,
      itemStyle: { color: _modelMeta[key].color },
      lineStyle: { color: _modelMeta[key].color, width: index === 0 ? 3 : 1.5, type: 'solid' },
      emphasis: { lineStyle: { width: index === 0 ? 4 : 2.5 } },
      z: index === 0 ? 5 : 2,
    })),
  };
  $: modelStrip = priceModelKeys.flatMap(key => {
    const row = (btc_models_latest || []).find(row => row.series === key && row.y != null);
    return row ? [{ key, ..._modelMeta[key], value: row.label }] : [];
  });
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
    data={btc_dominance}
    value=dominance
    title="Bitcoin Dominance"
    fmt='#,##0.00"%"'
    description="BTC share of total crypto market cap."
  />
  <BigValue
    data={btc_sentiment}
    value=sentiment
    title="Fear & Greed"
    description="Fear & Greed classification (0–100)."
  />
  <BigValue
    data={btc_valuation}
    value=valuation
    title="Bitcoin Valuation"
    description="Undervalued / Fair / Overvalued."
  />
</Grid>

</div>

## Performance

_Compare Bitcoin and other assets’ returns across the same periods._

### Stock Market Index Performance

<DataTable data={equity_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
</DataTable>

### Sector Performance

<DataTable data={sector_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
</DataTable>

### Macro Asset Class Performance

<DataTable data={macro_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
</DataTable>

### Bitcoin Industry Performance

<DataTable data={bitcoin_industry_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt='#,##0.00"%"' contentType=delta chip=true align=center />
</DataTable>

<div class="newsletter-visual" data-newsletter-visual="bitcoin-price">

## Bitcoin Price

_Price vs on-chain valuation models and moving averages._

### Secret Satoshis {dataYearLabel} Price Outlook

<div class="price-outlook-cases">
{#each outlookCaseLevels as c (c.name)}
  <div class="price-outlook-case" style="--case-color: {c.color}">
    <span class="case-label">{c.name}</span>
    <strong class="case-price">{_fmtUsd(c.price)}</strong>
  </div>
{/each}
</div>

<div class="price-chart-controls">
  <span>Click a series below to show or hide it.</span>
  <div class="price-chart-ranges" role="group" aria-label="Bitcoin price time range">
    {#each [1, 4, 10] as years}
      <button type="button" class:active={priceChartYears === years} aria-pressed={priceChartYears === years} on:click={() => priceChartYears = years}>{years}Y</button>
    {/each}
  </div>
</div>

<div class="model-values-strip">
{#each modelStrip as m (m.key)}
  <button type="button" class="model-value" class:off={!priceModelSelected[m.key]} aria-pressed={priceModelSelected[m.key]} on:click={() => togglePriceModel(m.key)} style="--c: {m.color}"><span class="dot"></span><span class="lbl">{m.label}</span><span class="val">{m.value}</span></button>
{/each}
</div>

<LineChart
  data={priceChartRows}
  x=date
  y={priceModelKeys}
  xFmt="mmm yyyy"
  yAxisTitle="Price (USD)"
  yFmt=usd0
  yMin={0}
  xType=time
  xMax={priceChartXMax}
  lineWidth=1
  echartsOptions={priceChartOptions}
  yGridlines=true
  xGridlines=false
  markers=false
  chartAreaHeight={500}
  legend=false
>
  {#each outlookCaseLevels as c (c.name)}
  <ReferenceLine data={[c]} y=price label=label hideValue=true labelPosition=aboveStart labelColor={c.color} lineColor={c.color} lineType=dashed lineWidth=1.5 />
  {/each}
</LineChart>

<p class="price-chart-methodology">Simple moving averages · 3-month = 90 daily closes · 1-year = 52 weeks / 364 daily closes · 200-week = 1,400 daily closes</p>

</div>

## Trading Range

_Days spent at each price level._

<Grid cols=2 gapSize=lg>

<Group>

### Days at Price ($1K Buckets)

<BarChart
  data={bucket_1k}
  x="Price Range ($)"
  y={['Current', 'Other']}
  swapXY=true
  seriesColors={{ Current: '#F7931A', Other: '#2a2a42' }}
  xAxisTitle=""
  yAxisTitle="Days"
  sort=false
  legend=false
  labels=true
  labelPosition=outside
  stackTotalLabel=false
/>

</Group>

<Group>

### Days at Price ($5K Buckets)

<BarChart
  data={bucket_5k}
  x="Price Range ($)"
  y={['Current', 'Other']}
  swapXY=true
  seriesColors={{ Current: '#F7931A', Other: '#2a2a42' }}
  xAxisTitle=""
  yAxisTitle="Days"
  sort=false
  legend=false
  labels=true
  labelPosition=outside
  stackTotalLabel=false
/>

</Group>

</Grid>

<div class="newsletter-visual" data-newsletter-visual="monthly-return-heatmap">

## Monthly Bitcoin Price Return Heatmap

_Monthly returns by year._

### Statistical Reference

<div class="monthly-heatmap-table">

<DataTable data={monthly_returns_agg} rows=all compact=true rowShading=false>
  <Column id=time title="Period" width=120 align=center />
  <Column id=Jan title="Jan" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Feb title="Feb" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Mar title="Mar" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Apr title="Apr" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=May title="May" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Jun title="Jun" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Jul title="Jul" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Aug title="Aug" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Sep title="Sep" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Oct title="Oct" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Nov title="Nov" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Dec title="Dec" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Yearly title="Yearly" fmt='#,##0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-80} colorMid={0} colorMax={150} align=center />
</DataTable>

</div>

### Historical Returns by Year

<div class="monthly-heatmap-table heatmap-historical">

<DataTable data={monthly_returns_years} rows=all compact=true rowShading=false>
  <Column id=time title="Year" width=120 align=center />
  <Column id=Jan title="Jan" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Feb title="Feb" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Mar title="Mar" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Apr title="Apr" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=May title="May" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Jun title="Jun" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Jul title="Jul" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Aug title="Aug" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Sep title="Sep" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Oct title="Oct" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Nov title="Nov" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Dec title="Dec" fmt='#,##0.0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-40} colorMid={0} colorMax={40} align=center />
  <Column id=Yearly title="Yearly" fmt='#,##0"%"' contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-80} colorMid={0} colorMax={150} align=center />
</DataTable>

</div>

</div>

## Seasonal Returns

_Compare this month’s and this year’s price paths with historical years._

<div class="newsletter-visual" data-newsletter-visual="seasonal-mtd">

### Bitcoin {dataMonthName} MTD Returns Comparison

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

### Bitcoin {dataYearLabel} YTD Returns Comparison

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
     Queries — placed at the bottom so they don't break up the dashboard view.
     Evidence resolves them regardless of position in the file.
     ───────────────────────────────────────────────────────────────────────── -->

```sql data_date
-- Month and year labels come from the data, not the viewer's clock, so headings can
-- never name a period the chart isn't showing.
select
  strftime(max(cast(date as date)), '%b %-d, %Y') as date_label,
  strftime(max(cast(date as date)), '%Y-%m-%d') as date_iso,
  strftime(max(cast(date as date)), '%B') as month_name,
  strftime(max(cast(date as date)), '%Y') as year_label
from bitcoin_report_library.summary_history
where Metric = 'Bitcoin Price USD'
```

```sql btc_price
-- Join observations by date so each comparison is against exactly 30 calendar days
-- earlier, independent of source row count or ordering. The latest row is displayed.
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

```sql btc_dominance
select CAST(Value AS DOUBLE) as dominance
from bitcoin_report_library.summary_table
where Metric = 'Bitcoin Dominance'
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
-- Calculate full calendar-day lookbacks before trimming to the 10-year display
-- history. Incomplete windows (including missing daily closes) remain null.
with model_history as (
  select
    cast(date as date) as date,
    "BTC Price",
    "Realized Price",
    "STH Realized Price",
    "3x Realized Price"
  from bitcoin_report_library.onchain_price_models
),
moving_averages as (
  select *,
    case when count("BTC Price") over quarter_window = 90
      then avg("BTC Price") over quarter_window end as "3-month MA",
    case when count("BTC Price") over year_window = 364
      then avg("BTC Price") over year_window end as "1-year MA",
    case when count("BTC Price") over cycle_window = 1400
      then avg("BTC Price") over cycle_window end as "200-week MA"
  from model_history
  window
    quarter_window as (order by date range between interval '89 days' preceding and current row),
    year_window as (order by date range between interval '363 days' preceding and current row),
    cycle_window as (order by date range between interval '1399 days' preceding and current row)
)
select * from moving_averages
where date >= (select max(date) from model_history) - interval '10 years'
order by date
```

```sql btc_models_latest
-- Latest value of each price model for the interactive legend
with latest as (
  select *
  from ${btc_with_models}
  where "BTC Price" is not null
  order by date desc
  limit 1
)
select 'BTC Price' as series, date as x, "BTC Price" as y, '$' || format('{:,.0f}', "BTC Price") as label from latest
union all select 'Realized Price', date, "Realized Price", '$' || format('{:,.0f}', "Realized Price") from latest
union all select 'STH Realized Price', date, "STH Realized Price", '$' || format('{:,.0f}', "STH Realized Price") from latest
union all select '3x Realized Price', date, "3x Realized Price", '$' || format('{:,.0f}', "3x Realized Price") from latest
union all select '3-month MA', date, "3-month MA", '$' || format('{:,.0f}', "3-month MA") from latest
union all select '1-year MA', date, "1-year MA", '$' || format('{:,.0f}', "1-year MA") from latest
union all select '200-week MA', date, "200-week MA", '$' || format('{:,.0f}', "200-week MA") from latest
```

```sql price_outlook
select
  label as name,
  label || ' - $' || format('{:,.0f}', cast(price as double)) as label,
  cast(price as double) as price,
  type,
  color
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

```sql bucket_1k
-- $1K buckets within ±$12K of the current Bitcoin price.
-- Parse the lower bound from labels like "$77K-$78K" by stripping $/K from
-- the first half of the string, then filter to the window around current price.
with parsed as (
  select
    "Price Range ($)",
    Count,
    "Current Price",
    cast(replace(replace(split_part("Price Range ($)", '-', 1), '$', ''), 'K', '') as integer) * 1000 as low_bound
  from bitcoin_report_library."1k_bucket_table"
)
select
  "Price Range ($)",
  case when "Current Price" >= low_bound and "Current Price" < low_bound + 1000 then Count end as Current,
  case when "Current Price" >= low_bound and "Current Price" < low_bound + 1000 then null else Count end as Other
from parsed
where low_bound between "Current Price" - 12000 and "Current Price" + 12000
```

```sql bucket_5k
-- All $5K price buckets in order from low to high.
-- Excludes the $0K-$5K and $5K-$10K buckets (BTC's early years) since their huge count
-- compresses all other bars and ruins the visual scale.
with parsed as (
  select
    "Price Range ($)",
    Count,
    "Current Price",
    cast(replace(replace(split_part("Price Range ($)", '-', 1), '$', ''), 'K', '') as integer) * 1000 as low_bound
  from bitcoin_report_library."5k_bucket_table"
  where "Price Range ($)" not in ('$0K-$5K', '$5K-$10K')
)
select
  "Price Range ($)",
  case when "Current Price" >= low_bound and "Current Price" < low_bound + 5000 then Count end as Current,
  case when "Current Price" >= low_bound and "Current Price" < low_bound + 5000 then null else Count end as Other
from parsed
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
-- See ytd_history: aggregates are recomputed chart-side over visible historical years.
select * exclude ("Median", "Average")
from bitcoin_report_library.mtd_returns_history
order by day
```

```sql ytd_history
-- Every year column is selected, including outliers. The chart decides which years to
-- draw (see HIDDEN_YEARS), keeps the current year visible, and recomputes
-- Median/Average over visible historical years only. The CSV's own aggregate columns
-- cover hidden/current years and are deliberately not used here.
select * exclude ("Median", "Average")
from bitcoin_report_library.ytd_returns_history
order by day_of_year
```
