import { HIDDEN_SEASONAL_YEARS, correlationGroups, correlationLabel, correlationName, correlationPeriods, num, type DashboardData, type Row, type Metric } from "./data";
export const escape = (v: unknown) =>
  String(v ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
export function format(value: unknown, kind = "number", digits = 0): string {
  const n = num(value);
  if (n === null) return "Unavailable";
  const scale =
    kind === "trillion"
      ? 1e12
      : kind === "billion"
        ? 1e9
        : kind === "million"
          ? 1e6
          : 1;
  const suffix =
    kind === "trillion"
      ? "T"
      : kind === "billion"
        ? "B"
        : kind === "million"
          ? "M"
          : kind === "percent"
            ? "%"
            : "";
  const prefix = ["usd", "trillion", "billion", "million"].includes(kind)
    ? "$"
    : "";
  return (
    prefix +
    (n / scale).toLocaleString("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }) +
    suffix
  );
}
function delta(value: unknown, digits = 2): string {
  const n = num(value);
  return n === null
    ? '<span class="muted">Unavailable</span>'
    : `<span class="delta ${n > 0 ? "positive" : n < 0 ? "negative" : "flat"}">${format(n, "percent", digits)} <span aria-hidden="true">${n > 0 ? "▲" : n < 0 ? "▼" : "–"}</span></span>`;
}
function tooltip(text: string): string {
  return `<span class="help"><button type="button" class="help-button" aria-label="About this metric" aria-expanded="false">i</button><span class="help-text" role="tooltip">${escape(text)}</span></span>`;
}
function sparkline(metric: Metric): string {
  const values = metric.history
    .map((r) => r.value)
    .filter((v): v is number => v !== null);
  if (values.length !== metric.history.length) return "";
  const lo = Math.min(...values),
    hi = Math.max(...values),
    range = hi - lo || 1;
  const points = values
    .map((v, i) => `${(i * 100) / (values.length - 1)},${22 - ((v - lo) / range) * 20}`)
    .join(" ");
  const good = metric.downIsGood
    ? (metric.change ?? 0) <= 0
    : (metric.change ?? 0) >= 0;
  return `<svg class="sparkline ${good ? "positive" : "negative"}" viewBox="0 0 100 24" preserveAspectRatio="none" role="img" aria-label="${escape(metric.title)} over 30 days"><polyline points="${points} 100,24 0,24" fill="currentColor" fill-opacity="0.22" stroke="none"/><polyline points="${points}" fill="none" stroke="currentColor" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>`;
}
function card(m: Metric): string {
  const change =
    m.change === null
      ? "Unavailable"
      : format(m.change * 100, "percent", m.title === "Bitcoin Supply" ? 2 : 1);
  const up = (m.change ?? 0) >= 0;
  const good = m.downIsGood ? (m.change ?? 0) <= 0 : up;
  const arrow = m.change === null ? "" : up ? "▲ " : "▼ ";
  return `<article class="metric-card"><div class="metric-label">${escape(m.title)}${tooltip(m.description)}</div><div class="metric-row"><strong class="metric-value">${format(m.value, m.format, ["million", "billion", "trillion"].includes(m.format) ? 2 : 0)}</strong>${sparkline(m)}</div><p class="metric-comparison"><span class="${good ? "positive" : "negative"}">${arrow}${change}</span> <span class="muted">vs 30d ago</span></p></article>`;
}
function table(
  headings: string[],
  rows: string[][],
  label: string,
  classes = "",
): string {
  return `<div class="table-scroll" role="region" aria-label="${escape(label)}" tabindex="0"><table class="${classes}"><thead><tr>${headings.map((h) => `<th scope="col">${escape(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c, i) => (i === 0 ? `<th scope="row">${c}</th>` : `<td>${c}</td>`)).join("")}</tr>`).join("")}</tbody></table></div>`;
}
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Yearly"];
// Diverging scale: red losses, soft grey near zero, green gains. Fixed bounds keep
// colours comparable across years.
function heatColor(n: number, yearly: boolean): { background: string; text: string } {
  const bound = yearly ? (n < 0 ? 80 : 150) : 40;
  return divergingColor(n, bound);
}
function divergingColor(n: number, bound: number): { background: string; text: string } {
  const weight = Math.min(1, Math.abs(n) / bound);
  const neutral = [170, 172, 182],
    target = n < 0 ? [214, 72, 66] : [26, 142, 78];
  const rgb = neutral.map((c, i) => Math.round(c + weight * (target[i] - c)));
  const luminance = rgb
    .map((channel) => {
      const s = channel / 255;
      return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    })
    .reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0);
  return { background: `rgb(${rgb.join(",")})`, text: luminance > 0.3 ? "#08080c" : "#ffffff" };
}
export function correlationMatrix(rows: Row[] | null, period = 90): string {
  if (!rows) return '<p class="methodology">The 90-day correlation matrix is unavailable in this data release.</p>';
  const tickers = correlationGroups.flatMap((g) => g.tickers);
  const starts = new Set(correlationGroups.map((g) => g.tickers[0]));
  const groups = correlationGroups.map((group, index) => {
    const body = group.tickers.map((ticker, i) => {
      const row = rows.find((r) => r.Ticker === ticker)!;
      return `<tr>${i === 0 ? `<th id="corr-${period}-group-${index}" scope="rowgroup" rowspan="${group.tickers.length}" class="correlation-group">${escape(group.title)}</th>` : ""}<th id="corr-${period}-row-${escape(ticker)}" scope="row" class="correlation-ticker" title="${escape(row.Asset)}"><span class="correlation-row-label"><span class="correlation-name">${escape(correlationName(row))}</span><span class="correlation-symbol">[${escape(correlationLabel(ticker))}]</span></span></th>${tickers.map((other) => {
        if (tickers.indexOf(other) > tickers.indexOf(ticker))
          return '<td class="correlation-unused" aria-hidden="true"></td>';
        const n = num(row[other]);
        const classes = [starts.has(other) ? "correlation-boundary" : "", ticker === other ? "correlation-diagonal" : ""].filter(Boolean).join(" ");
        const headers = `corr-${period}-group-${index} corr-${period}-row-${escape(ticker)} corr-${period}-col-${escape(other)}`;
        const title = `${correlationLabel(ticker)} / ${correlationLabel(other)}: ${n === null ? "Unavailable" : n.toFixed(2)}`;
        if (n === null) return `<td class="empty ${classes}" headers="${headers}" title="${escape(title)}">—</td>`;
        const { background, text } = divergingColor(n, 1);
        return `<td class="${classes}" headers="${headers}" title="${escape(title)}" style="background:${background};color:${text}">${n.toFixed(2)}</td>`;
      }).join("")}</tr>`;
    }).join("");
    return `<tbody>${body}</tbody>`;
  }).join("");
  return `<div class="table-scroll correlation-scroll" role="region" aria-label="${period}-day asset correlation matrix" tabindex="0"><table class="heatmap correlation-matrix"><caption class="sr-only">${period}-day Pearson return correlations for Bitcoin and 16 assets, grouped on both axes. Each pair appears once in the lower triangle.</caption><colgroup><col class="correlation-group-column"><col class="correlation-ticker-column"></colgroup>${correlationGroups.map((g) => `<colgroup span="${g.tickers.length}"></colgroup>`).join("")}<thead><tr><th rowspan="2" scope="col" class="correlation-group">Asset group</th><th rowspan="2" scope="col" class="correlation-ticker">Asset</th>${correlationGroups.map((g) => `<th colspan="${g.tickers.length}" scope="colgroup" class="correlation-boundary">${escape(g.title)}</th>`).join("")}</tr><tr>${tickers.map((t) => `<th id="corr-${period}-col-${escape(t)}" scope="col" class="${starts.has(t) ? "correlation-boundary" : ""}" title="${escape(rows.find((r) => r.Ticker === t)!.Asset)}">${escape(correlationLabel(t))}</th>`).join("")}</tr></thead>${groups}</table></div><div class="correlation-legend" aria-label="Correlation color scale"><span class="correlation-scale" aria-hidden="true"></span><span>−1 <span class="muted">Opposite</span></span><span>0 <span class="muted">Uncorrelated</span></span><span>+1 <span class="muted">Together</span></span></div><p class="methodology">Pearson return correlations over ${period} calendar days, using shared observation dates. Each pair appears once; — indicates insufficient data. DXY = DX-Y.NYB · GSCI = ^SPGSCI.</p>`;
}
function correlationSection(d: DashboardData): string {
  if (!d.correlations) return correlationMatrix(null);
  return `<div class="correlation-tools"><h3 id="correlation-matrix-title">90-Day Correlation Matrix</h3><div class="correlation-periods" role="tablist" aria-label="Correlation time frame">${correlationPeriods.map((period) => `<button type="button" role="tab" id="correlation-tab-${period}" data-correlation-period="${period}" aria-controls="correlation-panel-${period}" aria-selected="${period === 90}" tabindex="${period === 90 ? 0 : -1}">${period} days</button>`).join("")}</div></div>${correlationPeriods.map((period) => `<div role="tabpanel" id="correlation-panel-${period}" aria-labelledby="correlation-tab-${period}" ${period === 90 ? "" : "hidden"}>${correlationMatrix(d.correlations![period], period)}</div>`).join("")}`;
}
function heatmap(rows: Row[], d: DashboardData, reference = false): string {
  const body = rows.map((r) => {
    let year = r.Year === "4-Year Average" ? "4 Year Avg" : r.Year;
    if (year === d.year && !d.reportDate.endsWith("12-31")) year += " (YTD)";
    return `<tr><th scope="row">${escape(year)}</th>${MONTHS.map((key) => {
      const n = num(r[key]);
      if (n === null) return `<td class="empty">—</td>`;
      const { background, text } = heatColor(n, key === "Yearly");
      return `<td style="background:${background};color:${text}">${format(n, "percent", key === "Yearly" ? 0 : 1)}</td>`;
    }).join("")}</tr>`;
  });
  return `<div class="table-scroll" role="region" aria-label="${reference ? "Statistical reference" : "Historical monthly returns"}" tabindex="0"><table class="heatmap"><thead><tr><th scope="col">${reference ? "Period" : "Year"}</th>${MONTHS.map((c) => `<th scope="col">${c}</th>`).join("")}</tr></thead><tbody>${body.join("")}</tbody></table></div>`;
}
function chart(id: string, title: string): string {
  const file =
    id === "dashboard-price-outlook"
      ? "frame.html"
      : id.endsWith("mtd")
        ? "seasonal-mtd.html"
        : "seasonal-ytd.html";
  return `<div class="chart-container"><p class="chart-error" role="alert" data-chart-error="${id}" hidden></p><iframe src="/shared-chart/${file}" title="${escape(title)}" data-chart-frame="${id}" data-chart-src="/data/charts/${id}.json" ${id === "dashboard-price-outlook" ? "data-price-outlook-frame" : ""} class="shared-chart" loading="eager"></iframe><noscript><p>Interactive charts require JavaScript. <a href="https://secretsatoshis.github.io/Bitcoin-Report-Library/">Download the source data.</a></p></noscript></div>`;
}
function sectionHeading(title: string, subtitle: string): string {
  return `<div class="section-heading"><h2>${title}</h2><p>${subtitle}</p></div>`;
}
const NAV_LINKS = [
  ["https://newsletter.secretsatoshis.com/p/start-here", "Start Here"],
  ["https://newsletter.secretsatoshis.com/", "Newsletter"],
  ["https://chatgpt.com/g/g-BZXtVdU6M-agent-21", "Agent 21"],
  ["https://charts.secretsatoshis.com/", "Charts"],
  ["https://dashboard.secretsatoshis.com/", "Dashboard"],
];
const SECTIONS = [
  ["bitcoin-snapshot", "Snapshot"],
  ["bitcoin-price", "Price"],
  ["performance", "Performance"],
  ["correlation", "Correlation"],
  ["monthly-bitcoin-price-return-heatmap", "Returns"],
  ["seasonal-returns", "Seasonality"],
  ["relative-valuation", "Valuation"],
  ["network-fundamentals", "Network"],
  ["bitcoin-roi-by-time-frame", "ROI"],
];
function performanceSection(d: DashboardData): string {
  return d.performance
    .map(
      (g) =>
        `<div class="performance-group" data-quarterly-visual="${g.id}"><h3 id="${g.title.toLowerCase().replaceAll(" ", "-")}">${escape(g.title)}</h3>${table(
          ["Asset", "Price", "7 Day Return", "MTD Return", "YTD Return", "90 Day Return"],
          g.rows.map((r) => [
            r.Asset === "Bitcoin - [BTC]"
              ? `<span class="accent">${escape(r.Asset)}</span>`
              : escape(r.Asset),
            format(r.Price, "usd"),
            ...["7 Day Return (%)", "MTD Return (%)", "YTD Return (%)", "90 Day Return (%)"].map((k) => delta(r[k])),
          ]),
          g.title,
          "performance",
        )}</div>`,
    )
    .join("");
}
function fundamentalsTable(d: DashboardData): string {
  const rows = d.fundamentals
    .map((r, i) => {
      const first = i === 0 || d.fundamentals[i - 1].Section !== r.Section;
      const span = d.fundamentals.filter((x) => x.Section === r.Section).length;
      const category = first
        ? `<th scope="rowgroup" rowspan="${span}" class="group">${escape(r.Section)}</th>`
        : "";
      return `<tr class="${first ? "group-start" : ""}">${category}<td class="metric-name">${escape(r.Metric)}</td><td>${escape(r["Current Value"] || "Unavailable")}</td><td>${escape(r["7 Days Ago"] || "Unavailable")}</td><td>${delta(r["7 Day Change (%)"])}</td><td>${escape(r["52W Low"] || "Unavailable")}</td><td>${escape(r["52W High"] || "Unavailable")}</td></tr>`;
    })
    .join("");
  return `<div class="table-scroll" role="region" aria-label="Network fundamentals" tabindex="0"><table class="fundamentals"><thead><tr>${["Category", "Metric", "Current Value", "7 Days Ago", "7d Change", "52W Low", "52W High"].map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
}
function relativeTable(d: DashboardData): string {
  const shares = d.relative.map((r) => num(r.btcShare)).filter((n): n is number => n !== null);
  const widest = Math.max(100, ...shares);
  return table(
    ["Asset", "Market Cap (USD)", "Hypothetical BTC Price (USD)", "Change from Current BTC Price (%)", "BTC / Asset Market Cap (%)"],
    d.relative.map((r) => [
      r.Asset === "Bitcoin" ? '<span class="accent">Bitcoin</span>' : escape(r.Asset),
      format(r["Market Cap (USD)"], "trillion", 2),
      format(r["BTC Price at Market Cap"], "usd"),
      delta(r["Move Needed (%)"], 0),
      r.btcShare
        ? `<span class="share"><span class="share-bar" style="width:${((num(r.btcShare) ?? 0) / widest) * 100}%"></span><span class="share-value">${format(r.btcShare, "percent", 1)}</span></span>`
        : "—",
    ]),
    "Relative valuation",
    "relative",
  );
}
export function renderDashboard(d: DashboardData): string {
  const cases = d.cases
    .map(
      (c) =>
        `<div class="case" style="--case-color:${escape(c.color)}"><span>${escape(c.label)}</span><strong>${format(c.price, "usd")}</strong></div>`,
    )
    .join("");
  const sentiment = [
    ["Supply in Profit", format(d.sentiment.supplyInProfit, "percent", 1), "Share of bitcoin whose current price exceeds its last-moved price."],
    ["Fear & Greed", d.sentiment.sentiment, "NUPL zone of the trailing seven-day average; derived from holder unrealized profits and losses."],
    ["Bitcoin Valuation", d.sentiment.valuation, "Bitcoin price divided by fitted power-law fair value."],
  ]
    .map(
      ([label, value, description]) =>
        `<article class="metric-card"><div class="metric-label">${label}${tooltip(description)}</div><strong class="sentiment-value">${escape(value)}</strong></article>`,
    )
    .join("");
  const footerGroup = (title: string, links: string[][]) =>
    `<div><h2>${title}</h2><ul>${links.map(([url, label]) => `<li><a href="${url}">${label}</a></li>`).join("")}</ul></div>`;
  const hidden = HIDDEN_SEASONAL_YEARS.join(", ");
  return `<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bitcoin Market Dashboard | Secret Satoshis</title><meta name="description" content="Current Bitcoin market data, valuation models, on-chain conditions and cycle context, updated daily from an open pipeline."><link rel="canonical" href="https://dashboard.secretsatoshis.com/"><meta name="robots" content="index, follow, max-image-preview:large"><meta name="theme-color" content="#08080c">
<meta property="og:type" content="website"><meta property="og:site_name" content="Secret Satoshis"><meta property="og:title" content="Bitcoin Market Dashboard | Secret Satoshis"><meta property="og:description" content="Daily Bitcoin market intelligence from a verified open data release."><meta property="og:url" content="https://dashboard.secretsatoshis.com/"><meta property="og:image" content="https://secretsatoshis.com/assets/images/social-card.jpg"><meta property="og:image:alt" content="Secret Satoshis — AI-Native Bitcoin Market Intelligence"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:site" content="@SecretSatoshis"><meta name="twitter:title" content="Bitcoin Market Dashboard | Secret Satoshis"><meta name="twitter:description" content="Daily Bitcoin market intelligence from a verified open data release."><meta name="twitter:image" content="https://secretsatoshis.com/assets/images/social-card.jpg">
<link rel="icon" href="/favicon.ico"><link rel="stylesheet" href="../src/styles.css"><script type="module" src="../src/main.ts"></script></head><body>
<a class="skip-link" href="#bitcoin-snapshot">Skip to dashboard</a><header class="site-nav"><a href="https://secretsatoshis.com/" class="brand"><span class="accent">//</span> SECRET SATOSHIS</a><button id="nav-toggle" type="button" aria-label="Open menu" aria-controls="site-links" aria-expanded="false">≡</button><nav id="site-links" aria-label="Primary navigation">${NAV_LINKS.map(([url, label]) => `<a href="${url}" ${label === "Dashboard" ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</nav></header>
<section class="hero" data-dashboard-date="${d.reportDate}" aria-labelledby="dashboard-title"><div class="hero-grid"><div><p class="eyebrow"><span class="accent">//</span> Market Intelligence</p><h1 id="dashboard-title">Bitcoin Market Dashboard<span class="accent">.</span></h1><p class="hero-description">A daily view of Bitcoin market performance, on-chain conditions, valuation, and network activity.</p><p class="hero-source">Explore daily Bitcoin market data and the models behind the analysis. <a href="https://secretsatoshis.github.io/Bitcoin-Report-Library/">View the source data.</a></p></div><dl class="release-status" aria-label="Dashboard status"><div><dt>Latest data</dt><dd>${d.dateLabel}</dd></div><div><dt>Refresh cadence</dt><dd>Daily</dd></div></dl></div></section>
<nav class="section-nav" aria-label="Dashboard sections"><div>${SECTIONS.map(([id, label]) => `<a href="#${id}">${label}</a>`).join("")}</div></nav>
<main>
<section id="bitcoin-snapshot" class="dashboard-section">${sectionHeading("Bitcoin Snapshot", "Headline metrics — market, on-chain, and sentiment.")}<div class="visual-block" data-newsletter-visual="bitcoin-snapshot-market-data"><h3 id="market-data">Market Data</h3><div class="metric-grid">${d.metrics.slice(0, 3).map(card).join("")}</div></div><h3 id="on-chain-data">On-chain Data</h3><div class="metric-grid">${d.metrics.slice(3).map(card).join("")}</div><h3 id="investor-sentiment">Investor Sentiment</h3><div class="metric-grid sentiment-grid">${sentiment}</div></section>
<section id="bitcoin-price" class="dashboard-section" data-newsletter-visual="bitcoin-price">${sectionHeading("Bitcoin Price", "Price vs on-chain valuation models and moving averages.")}<h3 id="price-outlook">Secret Satoshis ${d.year} Price Outlook</h3><div class="case-grid">${cases}</div>${chart(d.charts[0].id, d.charts[0].title)}<p class="methodology">Simple moving averages · 3-month = 90 daily closes · 1-year = 364 daily closes · 200-week = 1,400 daily closes. Annual cases are maintained outlook levels.</p></section>
<section id="performance" class="dashboard-section">${sectionHeading("Performance", "Compare Bitcoin and other assets’ returns across the same periods.")}${performanceSection(d)}</section>
<section id="correlation" class="dashboard-section">${sectionHeading("Correlation", "Return correlations across Bitcoin and major markets.")}${correlationSection(d)}</section>
<section id="monthly-bitcoin-price-return-heatmap" class="dashboard-section" data-newsletter-visual="monthly-return-heatmap">${sectionHeading("Monthly Bitcoin Price Return Heatmap", "Monthly returns by year, measured from the previous period’s close.")}<h3 id="statistical-reference">Statistical Reference</h3>${heatmap(d.heatmap.reference, d, true)}<h3 id="historical-returns-by-year">Historical Returns by Year</h3>${heatmap(d.heatmap.years, d)}<p class="methodology">Statistics omit incomplete periods. The four-year average uses the four most recent completed observations for each column.</p></section>
<section id="seasonal-returns" class="dashboard-section">${sectionHeading("Seasonal Returns", "Compare this month’s and this year’s price paths with historical years.")}${d.charts
    .slice(1)
    .map(
      (p, i) =>
        `<div class="seasonal-block" data-newsletter-visual="seasonal-${i === 0 ? "mtd" : "ytd"}"><h3 id="${i === 0 ? "mtd" : "ytd"}-returns-comparison">${escape(p.title)}</h3>${chart(p.id, p.title)}</div>`,
    )
    .join("")}<p class="methodology">Paths start at the current period’s prior close. Average and Median exclude ${hidden} and the current year; February 29 is removed from YTD alignment. These comparisons are not forecasts.</p></section>
<section id="relative-valuation" class="dashboard-section" data-quarterly-visual="relative-valuation">${sectionHeading("Relative Valuation", "Bitcoin’s hypothetical price if its market cap matched each reference asset. These are comparison scenarios, not forecasts.")}${relativeTable(d)}</section>
<section id="network-fundamentals" class="dashboard-section">${sectionHeading("Network Fundamentals", "Network health, security &amp; on-chain economics.")}${fundamentalsTable(d)}</section>
<section id="bitcoin-roi-by-time-frame" class="dashboard-section">${sectionHeading("Bitcoin ROI by Time Frame", "Returns by holding period.")}${table(
    ["Period", "ROI", "Start Price"],
    d.roi.map((r) => [escape(r["Time Frame"]), delta(r["ROI (%)"], 1), format(r["Start Price"], "usd")]),
    "Bitcoin holding-period returns",
    "roi",
  )}</section>
</main>
<footer class="site-footer"><div class="footer-inner"><div class="footer-top"><div><a class="brand" href="https://secretsatoshis.com/"><span class="accent">//</span> SECRET SATOSHIS</a><p>AI-native Bitcoin market intelligence</p></div>${footerGroup("Platform", NAV_LINKS.slice(0, 3))}${footerGroup("Data", [
    ["https://dashboard.secretsatoshis.com/", "Market Dashboard"],
    ["https://charts.secretsatoshis.com/", "Chart Library"],
    ["https://github.com/SecretSatoshis", "GitHub"],
  ])}${footerGroup("Connect", [
    ["https://x.com/SecretSatoshis", "𝕏 @SecretSatoshis"],
    ["https://linkedin.com/company/secretsatoshis/", "LinkedIn"],
    ["https://treybrunson.com/", "TreyBrunson.com"],
  ])}</div><div class="footer-bottom"><span>© ${d.year} Secret Satoshis · Don’t trust. Verify.</span><a href="https://www.tradingview.com/">Charts by TradingView</a><a href="/shared-chart/assets/NOTICE">Chart attribution</a><a href="/shared-chart/assets/LICENSE">Apache 2.0 license</a></div></div></footer>
</body></html>`;
}
