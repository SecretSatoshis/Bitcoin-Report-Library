---
title: Weekly Bitcoin Recap Dashboard
---

<Alert status=info>
<strong>Data as of:</strong> <Value data={data_date} column=date_label />
</Alert>

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
  $: satsColor      = sats_per_dollar?.length   ? (sats_per_dollar[0].pct_change   >= 0 ? POS : NEG) : FALLBACK;
  $: supplyColor    = btc_supply?.length        ? (btc_supply[0].pct_change        >= 0 ? POS : NEG) : FALLBACK;
  $: revenueColor   = btc_miner_revenue?.length ? (btc_miner_revenue[0].pct_change >= 0 ? POS : NEG) : FALLBACK;
  $: volumeColor    = btc_tx_volume?.length     ? (btc_tx_volume[0].pct_change     >= 0 ? POS : NEG) : FALLBACK;
  // Current month column index for heatmap highlighting.
  // Heatmap columns: 1=Period | 2=Jan | 3=Feb | ... | 13=Dec | 14=Yearly
  // So Jan=2, Feb=3, ..., Dec=13.
  const _now = new Date();
  const currentMonthCol = _now.getMonth() + 2;

  // ─── Seasonal returns chart helpers ──────────────────────────────────
  // Build the year-column list dynamically from the data so we never hard-code
  // a year and the chart auto-extends as the dataset grows. Special-case
  // Median, Average, and the current year for distinct styling.
  const _MONTHS = ['January','February','March','April','May','June',
                   'July','August','September','October','November','December'];
  const currentMonthName = _MONTHS[_now.getMonth()];
  const currentYearLabel = String(_now.getFullYear());

  // Faded muted-gray for historical years (semi-transparent so they recede),
  // off-white for Median, cypherpunk green for Average, Bitcoin-orange for current year.
  const _historicalColor = 'rgba(110, 110, 138, 0.45)';  // brand text-dim @ 45%
  const _medianColor = '#e4e4ef';                         // brand text (legible on dark)
  const _averageColor = '#00FF88';
  const _currentColor = '#F7931A';

  function _buildYearList(rows) {
    if (!rows?.length) return [];
    const cols = Object.keys(rows[0]).filter(c => c !== 'day' && c !== 'day_of_year');
    return cols;
  }
  function _buildColorMap(years) {
    return Object.fromEntries(
      years.map(name => [
        name,
        name === 'Median'  ? _medianColor :
        name === 'Average' ? _averageColor :
        name === currentYearLabel ? _currentColor : _historicalColor
      ])
    );
  }

  $: mtdYears = _buildYearList(mtd_history);
  $: ytdYears = _buildYearList(ytd_history);
  $: mtdSeriesColors = _buildColorMap(mtdYears);
  $: ytdSeriesColors = _buildColorMap(ytdYears);

  // Per-series line widths: emphasize current year + Median + Average.
  // Median uses a dashed line so it visually separates from the white axis labels.
  function _buildSeriesWidths(years) {
    return {
      series: years.map(name => {
        const wide = (name === 'Median' || name === 'Average' || name === currentYearLabel);
        const lineStyle = { width: wide ? 2.5 : 1 };
        if (name === 'Median') lineStyle.type = 'dashed';
        return { lineStyle };
      })
    };
  }
  $: mtdEchartsOptions = _buildSeriesWidths(mtdYears);
  $: ytdEchartsOptions = _buildSeriesWidths(ytdYears);

  function _fmtUsd(n) {
    if (n == null || isNaN(n)) return '';
    return '$' + Math.round(Number(n)).toLocaleString();
  }
  function _buildLatestPoints(rows, xKey) {
    if (!rows?.length) return { current: [], median: [], average: [] };
    const cy = currentYearLabel;
    // last row where current year col is not null (= today)
    let currentRow = null;
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i][cy] != null) { currentRow = rows[i]; break; }
    }
    // last row of full series for Median/Average (end of month / end of year)
    let endRow = null;
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i]['Median'] != null || rows[i]['Average'] != null) { endRow = rows[i]; break; }
    }
    return {
      current: currentRow ? [{ x: currentRow[xKey], y: currentRow[cy], label: _fmtUsd(currentRow[cy]) }] : [],
      median:  endRow     ? [{ x: endRow[xKey],     y: endRow['Median'],  label: _fmtUsd(endRow['Median']) }]   : [],
      average: endRow     ? [{ x: endRow[xKey],     y: endRow['Average'], label: _fmtUsd(endRow['Average']) }]  : [],
    };
  }
  $: mtdLatest = _buildLatestPoints(mtd_history, 'day');
  $: ytdLatest = _buildLatestPoints((ytd_history || []).filter(r => r.day_of_year <= 365), 'day_of_year');

  // ─── Bitcoin Price chart current-values strip ────────────────────────
  // BTC Price always first; remaining models sorted by ascending value.
  const _modelMeta = {
    'BTC Price':           { color: '#F7931A', label: 'BTC Price' },
    'Realized Price':      { color: '#2962FF', label: 'Realized' },
    'STH Realized Price':  { color: '#E040FB', label: 'STH Realized' },
    '3x Realized Price':   { color: '#8B5E34', label: '3× Realized' },
    'Electricity Cost':    { color: '#8A8D91', label: 'Electricity Cost' },
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
    fmt=pct2
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
  <Column id=return_7d title="7 Day Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt=pct2 contentType=delta chip=true align=center />
</DataTable>

### Sector Performance

<DataTable data={sector_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt=pct2 contentType=delta chip=true align=center />
</DataTable>

### Macro Asset Class Performance

<DataTable data={macro_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt=pct2 contentType=delta chip=true align=center />
</DataTable>

### Bitcoin Industry Performance

<DataTable data={bitcoin_industry_perf} rows=all rowShading=true>
  <Column id=Asset title="Asset" contentType=html />
  <Column id=price title="Price" fmt=usd0 align=center />
  <Column id=return_7d title="7 Day Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_mtd title="MTD Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_ytd title="YTD Return" fmt=pct2 contentType=delta chip=true align=center />
  <Column id=return_90d title="90 Day Return" fmt=pct2 contentType=delta chip=true align=center />
</DataTable>

## Bitcoin Price

_Price vs on-chain valuation models._

### Secret Satoshis 2026 Price Outlook

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
  y={['BTC Price', 'Realized Price', 'STH Realized Price', 'Electricity Cost', '3x Realized Price']}
  xFmt="mmm yyyy"
  yAxisTitle="Price (USD)"
  yFmt=usd0
  yMin={0}
  xType=time
  xMax={new Date('2026-12-31')}
  lineWidth=1
  seriesColors={{
    'BTC Price': '#F7931A',
    'Btc Price': '#F7931A',
    'Realized Price': '#2962FF',
    'STH Realized Price': '#E040FB',
    'Sth Realized Price': '#E040FB',
    'Electricity Cost': '#8A8D91',
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
  <ReferenceLine x="2026-01-01" label="2026 Start" hideValue=true labelPosition=aboveStart labelColor="#6e6e8a" lineColor="#6e6e8a" lineType=dashed lineWidth=1 bold=false />
  <ReferenceLine data={price_outlook.filter(d => d.name === 'Bull Case')} y=price label=label hideValue=true labelPosition=aboveEnd lineColor="#00FF88" lineType=dashed lineWidth=2 />
  <ReferenceLine data={price_outlook.filter(d => d.name === 'Base Case')} y=price label=label hideValue=true labelPosition=aboveEnd lineColor="#FFD700" lineType=dashed lineWidth=2 />
  <ReferenceLine data={price_outlook.filter(d => d.name === 'Bear Case')} y=price label=label hideValue=true labelPosition=aboveEnd lineColor="#FF3B30" lineType=dashed lineWidth=2 />
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
  <Column id=Jan title="Jan" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Feb title="Feb" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Mar title="Mar" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Apr title="Apr" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=May title="May" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Jun title="Jun" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Jul title="Jul" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Aug title="Aug" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Sep title="Sep" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Oct title="Oct" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Nov title="Nov" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Dec title="Dec" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Yearly title="Yearly" fmt=pct0 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.8} colorMid={0} colorMax={1.5} align=center />
</DataTable>

</div>

### Historical Returns by Year

<div class="monthly-heatmap-table heatmap-historical" data-current-col={currentMonthCol}>

<DataTable data={monthly_returns_years} rows=all compact=true rowShading=false>
  <Column id=time title="Year" width=120 align=center />
  <Column id=Jan title="Jan" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Feb title="Feb" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Mar title="Mar" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Apr title="Apr" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=May title="May" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Jun title="Jun" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Jul title="Jul" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Aug title="Aug" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Sep title="Sep" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Oct title="Oct" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Nov title="Nov" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Dec title="Dec" fmt=pct1 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.4} colorMid={0} colorMax={0.4} align=center />
  <Column id=Yearly title="Yearly" fmt=pct0 contentType=colorscale colorScale={['#FF3B30', '#0A0A0A', '#00FF88']} colorMin={-0.8} colorMid={0} colorMax={1.5} align=center />
</DataTable>

</div>

## Seasonal Returns

_Current MTD & YTD vs historical years._

### Bitcoin {currentMonthName} MTD Returns Comparison

<LineChart
  data={mtd_history}
  x=day
  y={mtdYears}
  xAxisTitle="Day of Month"
  yAxisTitle="Indexed to Current Year Start ($)"
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

### Bitcoin {currentYearLabel} YTD Returns Comparison

<LineChart
  data={ytd_history}
  x=day_of_year
  y={ytdYears}
  xAxisTitle="Day of Year"
  yAxisTitle="Indexed to Current Year Start ($)"
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
    fmt=pct0
    contentType=delta
    chip=true
    align=center
  />
  <Column
    id=btc_pct_of_mcap
    title="BTC % Of Asset Market Cap"
    contentType=bar
    fmt=pct1
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
    id="7 Day Avg % Change"
    title="7d Avg % Change"
    fmt=pct2
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
    fmt=pct1
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
select strftime(max(cast(date as date)), '%B %d, %Y') as date_label
from bitcoin_report_library.summary_history
where Metric = 'Bitcoin Price USD'
```

```sql btc_price
-- summary_history is generated by main.py as the last 30 daily rows per metric.
-- pct_change here = (today - first row) / first row, which equals "vs 30 days ago"
-- ONLY because the source window is 30 rows. Snapshot BigValues display this as
-- "vs 30d ago". If main.py's tail(30) ever changes, update the comparisonTitle.
with src as (
  select cast(date as date) as date, Value as price
  from bitcoin_report_library.summary_history
  where Metric = 'Bitcoin Price USD'
)
select
  date,
  price,
  (price - first_value(price) over (order by date))
    / first_value(price) over (order by date) as pct_change
from src
order by date desc
```

```sql btc_marketcap
with src as (
  select cast(date as date) as date, Value / 1e12 as marketcap
  from bitcoin_report_library.summary_history
  where Metric = 'Bitcoin Marketcap'
)
select
  date,
  marketcap,
  (marketcap - first_value(marketcap) over (order by date))
    / first_value(marketcap) over (order by date) as pct_change
from src
order by date desc
```

```sql sats_per_dollar
with src as (
  select cast(date as date) as date, Value as sats
  from bitcoin_report_library.summary_history
  where Metric = 'Sats Per Dollar'
)
select
  date,
  sats,
  (sats - first_value(sats) over (order by date))
    / first_value(sats) over (order by date) as pct_change
from src
order by date desc
```

```sql btc_supply
with src as (
  select cast(date as date) as date, Value as supply
  from bitcoin_report_library.summary_history
  where Metric = 'Bitcoin Supply'
)
select
  date,
  supply,
  (supply - first_value(supply) over (order by date))
    / first_value(supply) over (order by date) as pct_change
from src
order by date desc
```

```sql btc_miner_revenue
with src as (
  select cast(date as date) as date, Value / 1e6 as revenue
  from bitcoin_report_library.summary_history
  where Metric = 'Bitcoin Miner Revenue'
)
select
  date,
  revenue,
  (revenue - first_value(revenue) over (order by date))
    / first_value(revenue) over (order by date) as pct_change
from src
order by date desc
```

```sql btc_tx_volume
with src as (
  select cast(date as date) as date, Value / 1e9 as volume
  from bitcoin_report_library.summary_history
  where Metric = 'Bitcoin Transaction Volume'
)
select
  date,
  volume,
  (volume - first_value(volume) over (order by date))
    / first_value(volume) over (order by date) as pct_change
from src
order by date desc
```

```sql btc_dominance
select CAST(Value AS DOUBLE) / 100 as dominance
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
  "7 Day Return" / 100 as return_7d,
  "MTD Return" / 100 as return_mtd,
  "YTD Return" / 100 as return_ytd,
  "90 Day Return" / 100 as return_90d
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
  "7 Day Return" / 100 as return_7d,
  "MTD Return" / 100 as return_mtd,
  "YTD Return" / 100 as return_ytd,
  "90 Day Return" / 100 as return_90d
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
  "7 Day Return" / 100 as return_7d,
  "MTD Return" / 100 as return_mtd,
  "YTD Return" / 100 as return_ytd,
  "90 Day Return" / 100 as return_90d
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
  "7 Day Return" / 100 as return_7d,
  "MTD Return" / 100 as return_mtd,
  "YTD Return" / 100 as return_ytd,
  "90 Day Return" / 100 as return_90d
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
-- A trailing ghost row at 2026-12-31 extends the x-axis past the data so
-- forecast/reference annotations have empty space on the right.
select
  cast(date as date) as date,
  "BTC Price",
  "Electricity Cost",
  "STH Realized Price",
  "Realized Price",
  "3x Realized Price"
from bitcoin_report_library.onchain_price_models
where cast(date as date) >= current_date - interval '4' year
union all
select date '2026-12-31', null, null, null, null, null
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
union all select 'Electricity Cost', date, "Electricity Cost", '$' || format('{:,.0f}', "Electricity Cost") from latest
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
    cast(replace(replace("Market Cap (USD)", '$', ''), ',', '') as double) / 1e12 as market_cap,
    cast(replace(replace("Market Cap BTC Price", '$', ''), ',', '') as double) as implied_price,
    cast(replace("BTC % Move to Marketcap BTC Price", '%', '') as double) / 100 as implied_return
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
    else least((select mc from btc_mcap) / market_cap, 1.0)
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
select
  time,
  Jan, Feb, Mar, Apr, May, Jun,
  Jul, Aug, Sep, Oct, Nov, Dec,
  Yearly
from bitcoin_report_library.monthly_heatmap_data
where try_cast(time as integer) is not null
order by try_cast(time as integer) desc
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
  "ROI (%)" / 100 as roi_pct,
  "BTC Price" as start_price
from bitcoin_report_library.roi_table
```

```sql fundamentals
select
  Section as Category,
  Metric,
  "Current Value",
  "7 Days Ago",
  "7 Day Avg % Change",
  Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday,
  "52W Low",
  "52W High"
from bitcoin_report_library.fundamentals_table
```

```sql mtd_history
select * from bitcoin_report_library.mtd_returns_history
order by day
```

```sql ytd_history
select * exclude ("2017")
from bitcoin_report_library.ytd_returns_history
where day_of_year <= 365
order by day_of_year
```
