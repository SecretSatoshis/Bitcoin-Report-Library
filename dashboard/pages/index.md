---
title: Weekly Bitcoin Recap Dashboard
---

<section class="dashboard-hero" aria-labelledby="dashboardHeroTitle">
  <div class="dashboard-hero-grid">
    <div class="dashboard-hero-copy">
      <p class="dashboard-eyebrow"><span class="brand-accent">//</span> Market Intelligence</p>
      <h1 id="dashboardHeroTitle">Weekly Bitcoin Recap Dashboard<span class="brand-accent">.</span></h1>
      <p class="dashboard-hero-sub">A live view of Bitcoin market performance, on-chain conditions, valuation, and network activity.</p>
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
  // ahead of the 16:00 UTC refresh.
  $: dataMonthName = data_date?.[0]?.month_name ?? '';
  $: dataYearLabel = data_date?.[0]?.year_label ?? '';

  // Years hidden from the seasonal charts. 2017's magnitude compresses every other
  // year into a flat band at the bottom of the plot. It is excluded from the chart
  // only — it stays in the CSV, and the Median/Average lines are recomputed below
  // over the visible historical years. The current year is plotted separately but
  // excluded from those reference aggregates while it is still incomplete.
  const HIDDEN_YEARS = ['2017'];

  const _isYearCol = (c) => /^\d{4}$/.test(c);

  // Faded muted-gray for historical years (semi-transparent so they recede),
  // off-white for Median, cypherpunk green for Average, Bitcoin-orange for current year.
  const _historicalColor = 'rgba(110, 110, 138, 0.45)';  // brand text-dim @ 45%
  const _medianColor = '#e4e4ef';                         // brand text (legible on dark)
  const _averageColor = '#00FF88';
  const _currentColor = '#F7931A';

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
    return Object.fromEntries(
      years.map(name => [
        name,
        name === 'Median'  ? _medianColor :
        name === 'Average' ? _averageColor :
        name === currentYear ? _currentColor : _historicalColor
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

  // Per-series line widths: emphasize current year + Median + Average.
  // Median uses a dashed line so it visually separates from the white axis labels.
  function _buildSeriesWidths(years, currentYear) {
    return {
      series: years.map(name => {
        const wide = (name === 'Median' || name === 'Average' || name === currentYear);
        const lineStyle = { width: wide ? 2.5 : 1 };
        if (name === 'Median') lineStyle.type = 'dashed';
        return { lineStyle };
      })
    };
  }
  $: mtdEchartsOptions = _buildSeriesWidths(mtdYears, mtdCurrentYear);
  $: ytdEchartsOptions = _buildSeriesWidths(ytdYears, ytdCurrentYear);

  function _fmtUsd(n) {
    if (n == null || isNaN(n)) return '';
    return '$' + Math.round(Number(n)).toLocaleString();
  }
  function _buildLatestPoints(rows, xKey, currentYear) {
    if (!rows?.length || !currentYear) return { current: [], median: [], average: [] };
    // last row where current year col is not null (= today)
    let currentRow = null;
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i][currentYear] != null) { currentRow = rows[i]; break; }
    }
    // last row of full series for Median/Average (end of month / end of year)
    let endRow = null;
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i]['Median'] != null || rows[i]['Average'] != null) { endRow = rows[i]; break; }
    }
    return {
      current: currentRow ? [{ x: currentRow[xKey], y: currentRow[currentYear], label: _fmtUsd(currentRow[currentYear]) }] : [],
      median:  endRow     ? [{ x: endRow[xKey],     y: endRow['Median'],  label: _fmtUsd(endRow['Median']) }]   : [],
      average: endRow     ? [{ x: endRow[xKey],     y: endRow['Average'], label: _fmtUsd(endRow['Average']) }]  : [],
    };
  }
  $: mtdLatest = _buildLatestPoints(mtdPlot, 'day', mtdCurrentYear);
  $: ytdLatest = _buildLatestPoints(ytdPlot, 'day_of_year', ytdCurrentYear);

  // ─── Bitcoin Price chart axis + outlook cases ────────────────────────
  // Extend the x-axis to the end of the data's own year so there is space to the right
  // of the last price point for the case annotations. Derived from the data, so on
  // Jan 1 the axis rolls forward on its own instead of clamping new points off-canvas.
  $: priceChartXMax = dataYearLabel
    ? new Date(`${dataYearLabel}-12-31T00:00:00`)
    : undefined;

  // Keep the chart focused on the three scenario cases. Support and resistance levels
  // remain available in price_outlook.csv, but their clustered labels obscure the models.
  $: outlookCaseLevels = (price_outlook || [])
    .filter(level => level.type === 'case')
    .slice()
    .sort((a, b) => Number(b.price) - Number(a.price));

  // ─── Bitcoin Price chart current-values strip ────────────────────────
  // BTC Price always first; remaining models sorted by ascending value.
  const _modelMeta = {
    'BTC Price':           { color: '#F7931A', label: 'BTC Price' },
    'Realized Price':      { color: '#2962FF', label: 'Realized' },
    'STH Realized Price':  { color: '#E040FB', label: 'STH Realized' },
    '3x Realized Price':   { color: '#8B5E34', label: '3× Realized' },
    'Power Expense ($0.05/kWh)': { color: '#8A8D91', label: 'Power Expense · $0.05/kWh' },
  };
  $: modelStrip = (() => {
    const rows = (btc_models_latest || []).filter(r => r.y != null);
    if (!rows.length) return [];
    const btc = rows.find(r => r.series === 'BTC Price');
    const others = rows
      .filter(r => r.series !== 'BTC Price')
      .sort((a, b) => Number(a.y) - Number(b.y));
    const ordered = btc ? [btc, ...others] : others;
    return ordered.map(r => ({
      key: r.series,
      label: _modelMeta[r.series]?.label ?? r.series,
      color: _modelMeta[r.series]?.color ?? '#ffffff',
      value: r.label,
    }));
  })();
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

_Returns vs Bitcoin across asset classes._

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

## Bitcoin Price

_Price vs on-chain valuation models._

### Secret Satoshis {dataYearLabel} Price Outlook

<div class="price-outlook-cases">
  <Grid cols=3 gapSize=lg>
  <div class="case-bear">
    <BigValue data={price_outlook_cases} value=bear title="Bear Case" fmt=usd0 />
  </div>
  <div class="case-base">
    <BigValue data={price_outlook_cases} value=base title="Base Case" fmt=usd0 />
  </div>
  <div class="case-bull">
    <BigValue data={price_outlook_cases} value=bull title="Bull Case" fmt=usd0 />
  </div>
  </Grid>
</div>

<div class="model-values-strip">
{#each modelStrip as m (m.key)}
  <div class="model-value" style="--c: {m.color}"><span class="dot"></span><span class="lbl">{m.label}</span><span class="val">{m.value}</span></div>
{/each}
</div>

<LineChart
  data={btc_with_models}
  x=date
  y={['BTC Price', 'Realized Price', 'STH Realized Price', 'Power Expense ($0.05/kWh)', '3x Realized Price']}
  xFmt="mmm yyyy"
  yAxisTitle="Price (USD)"
  yFmt=usd0
  yMin={0}
  xType=time
  xMax={priceChartXMax}
  lineWidth=1
  seriesColors={{
    'BTC Price': '#F7931A',
    'Btc Price': '#F7931A',
    'Realized Price': '#2962FF',
    'STH Realized Price': '#E040FB',
    'Sth Realized Price': '#E040FB',
    'Power Expense ($0.05/kWh)': '#8A8D91',
    '3x Realized Price': '#8B5E34'
  }}
  echartsOptions={{
    series: [
      { lineStyle: { width: 3 }, emphasis: { lineStyle: { width: 4 } } },
      { lineStyle: { width: 1 }, emphasis: { lineStyle: { width: 2 } } },
      { lineStyle: { width: 1 }, emphasis: { lineStyle: { width: 2 } } },
      { lineStyle: { width: 1 }, emphasis: { lineStyle: { width: 2 } } },
      { lineStyle: { width: 1 }, emphasis: { lineStyle: { width: 2 } } }
    ]
  }}
  yGridlines=true
  xGridlines=false
  markers=false
  chartAreaHeight={500}
  legend=false
>
  {#each outlookCaseLevels as c (c.name)}
  <ReferenceLine data={[c]} y=price label=label hideValue=true labelPosition=aboveEnd lineColor={c.color} lineType=dashed lineWidth=2 />
  {/each}
</LineChart>

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

## Seasonal Returns

_Current MTD & YTD vs historical years._

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
  legend=false
  yScale=true
  chartAreaHeight={420}
>
  <ReferencePoint data={mtdLatest.current} x=x y=y label=label labelPosition=right symbolSize=4 fontSize=11 color="#F7931A" labelColor="#F7931A" symbolColor="#F7931A" />
  <ReferencePoint data={mtdLatest.median} x=x y=y label=label labelPosition=right symbolSize=4 fontSize=11 color="#e4e4ef" labelColor="#e4e4ef" symbolColor="#e4e4ef" />
  <ReferencePoint data={mtdLatest.average} x=x y=y label=label labelPosition=right symbolSize=4 fontSize=11 color="#00FF88" labelColor="#00FF88" symbolColor="#00FF88" />
</LineChart>

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
  legend=false
  yScale=true
  chartAreaHeight={420}
>
  <ReferencePoint data={ytdLatest.current} x=x y=y label=label labelPosition=right symbolSize=4 fontSize=11 color="#F7931A" labelColor="#F7931A" symbolColor="#F7931A" />
  <ReferencePoint data={ytdLatest.median} x=x y=y label=label labelPosition=right symbolSize=4 fontSize=11 color="#e4e4ef" labelColor="#e4e4ef" symbolColor="#e4e4ef" />
  <ReferencePoint data={ytdLatest.average} x=x y=y label=label labelPosition=right symbolSize=4 fontSize=11 color="#00FF88" labelColor="#00FF88" symbolColor="#00FF88" />
</LineChart>

## Relative Valuation

_Implied BTC price by asset market cap._

<DataTable data={rel_val} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=market_cap title="Market Cap (USD)" fmt='$#,##0.00"T"' align=center />
  <Column id=implied_price title="Implied (USD) Bitcoin Price" fmt=usd0 align=center />
  <Column
    id=implied_return
    title="Implied (%) Return"
    fmt='#,##0"%"'
    contentType=delta
    chip=true
    align=center
  />
  <Column
    id=btc_pct_of_mcap
    title="BTC % Of Asset Market Cap"
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
  <Column id=Monday title="Mon" align=right />
  <Column id=Tuesday title="Tue" align=right />
  <Column id=Wednesday title="Wed" align=right />
  <Column id=Thursday title="Thu" align=right />
  <Column id=Friday title="Fri" align=right />
  <Column id=Saturday title="Sat" align=right />
  <Column id=Sunday title="Sun" align=right />
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
  strftime(max(cast(date as date)), '%B %d, %Y') as date_label,
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
-- Canonical daily Bitcoin price joined with on-chain valuation models.
-- This uses the same BTC price source as snapshot/performance tables.
-- A trailing all-null ghost row at the end of the data's own year extends the x-axis
-- past the last price point so the case annotations have empty space on the right.
-- Derived from the data rather than hardcoded, so it rolls forward every January.
with model_history as (
  select
    cast(date as date) as date,
    "BTC Price",
    "Electricity Cost" as "Power Expense ($0.05/kWh)",
    "STH Realized Price",
    "Realized Price",
    "3x Realized Price"
  from bitcoin_report_library.onchain_price_models
),
bounds as (
  select
    max(date) as max_date,
    date_trunc('year', max(date)) + interval '1 year' - interval '1 day' as year_end
  from model_history
)
select * from model_history
where date >= (select max_date from bounds) - interval '4' year
union all
select (select year_end from bounds), null, null, null, null, null
order by date
```

```sql btc_models_latest
-- Latest value of each price model for end-of-line annotations
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
union all select 'Power Expense ($0.05/kWh)', date, "Power Expense ($0.05/kWh)", '$' || format('{:,.0f}', "Power Expense ($0.05/kWh)") from latest
union all select '3x Realized Price', date, "3x Realized Price", '$' || format('{:,.0f}', "3x Realized Price") from latest
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

```sql price_outlook_cases
select
  max(case when label = 'Bear Case' then price end) as bear,
  max(case when label = 'Base Case' then price end) as base,
  max(case when label = 'Bull Case' then price end) as bull
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
-- The newest year is still in progress, so its Yearly figure is a partial-year return
-- sitting in a column of completed years. Label it so it can't be read as a full year.
with years as (
  select *, try_cast(time as integer) as yr
  from bitcoin_report_library.monthly_heatmap_data
  where try_cast(time as integer) is not null
)
select
  case when yr = (select max(yr) from years) then time || ' (YTD)' else time end as time,
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
  case when "Current Price" between low_bound and low_bound + 1000 then Count end as Current,
  case when "Current Price" between low_bound and low_bound + 1000 then null else Count end as Other
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
  case when "Current Price" between low_bound and low_bound + 5000 then Count end as Current,
  case when "Current Price" between low_bound and low_bound + 5000 then null else Count end as Other
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
