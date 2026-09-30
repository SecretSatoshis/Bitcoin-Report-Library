import Papa from "papaparse";

export type Row = Record<string, string>;
export type Tables = Record<string, Row[]>;
export interface ReleaseManifest {
  schema_version: number;
  report_date: string;
  release_id: string;
  generated_at: string;
  files: Record<string, { sha256: string; size_bytes: number }>;
}
export interface Series {
  id: string;
  name: string;
  axis: string;
  role: string;
  color: string;
  lineWidth: number;
  lineStyle: string;
  opacity: number;
  start: number;
  values: (number | null)[];
}
export interface Candle {
  time: string;
  periodEnd: string;
  observationDate: string;
  complete: boolean;
  open: number;
  high: number;
  low: number;
  close: number;
}
export interface ChartPayload {
  schemaVersion: 2;
  id: string;
  family: string;
  title: string;
  description: string;
  category: string;
  source: string;
  reportDate: string;
  coverage: string;
  axisKind: "time" | "days";
  axes: Record<string, { unit: string; label: string; mode: string }>;
  defaultRange: string;
  x: (string | number)[];
  series: Series[];
  events: { date: string; name: string; originalDate?: string }[];
  unavailable: string[];
  readingPoint: string | number;
  note: string;
  gridlines: boolean;
  defaultPresentation?: string;
  defaultInterval?: string;
  rangeEndDate?: string;
  referenceLines?: { name: string; price: number; color: string }[];
  candleViews?: Record<
    string,
    {
      x: string[];
      series: Series[];
      candles: Candle[];
      events: ChartPayload["events"];
      readingPoint: string;
      interval: string;
      offset?: number;
      ohlc?: number[][];
    }
  >;
  xAxisLabel?: string;
  seriesOrder?: string[];
}
export interface Metric {
  name: string;
  title: string;
  value: number | null;
  change: number | null;
  history: { date: string; value: number | null }[];
  format: string;
  description: string;
  downIsGood?: boolean;
}
export interface DashboardData {
  reportDate: string;
  dateLabel: string;
  month: string;
  year: string;
  metrics: Metric[];
  sentiment: {
    supplyInProfit: number | null;
    sentiment: string;
    valuation: string;
  };
  performance: { id: string; title: string; rows: Row[] }[];
  heatmap: { reference: Row[]; years: Row[] };
  relative: (Row & { btcShare: string })[];
  fundamentals: Row[];
  roi: Row[];
  cases: Row[];
  charts: ChartPayload[];
}

export const inputFiles = [
  "summary_table.csv",
  "summary_history.csv",
  "fundamentals_table.csv",
  "performance_table.csv",
  "monthly_heatmap_data.csv",
  "relative_value_comparison.csv",
  "roi_table.csv",
  "onchain_price_models.csv",
  "mtd_price_paths.csv",
  "ytd_price_paths.csv",
  "price_outlook.csv",
  "bitcoin_candles.csv.gz",
];
export const num = (value: unknown): number | null => {
  if (value == null || String(value).trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};
export function parseCSV(text: string, name: string): Row[] {
  const parsed = Papa.parse<Row>(text, {
    header: true,
    skipEmptyLines: "greedy",
    dynamicTyping: false,
  });
  if (
    parsed.errors.length ||
    (parsed.meta as Papa.ParseMeta & { renamedHeaders?: unknown })
      .renamedHeaders ||
    !parsed.data.length
  )
    throw new Error(`${name}: malformed, duplicate-header or empty CSV`);
  return parsed.data;
}
const fields: Record<string, { required: string[]; numeric: string[] }> = {
  "summary_table.csv": {
    required: ["Metric", "Value", "Label"],
    numeric: ["Value"],
  },
  "summary_history.csv": {
    required: ["date", "Metric", "Value"],
    numeric: ["Value"],
  },
  "fundamentals_table.csv": {
    required: [
      "Section",
      "Metric",
      "Current Value",
      "7 Days Ago",
      "7 Day Change (%)",
      "52W Low",
      "52W High",
    ],
    numeric: ["7 Day Change (%)"],
  },
  "performance_table.csv": {
    required: [
      "Category",
      "Asset",
      "Price",
      "7 Day Return (%)",
      "MTD Return (%)",
      "YTD Return (%)",
      "90 Day Return (%)",
    ],
    numeric: [
      "Price",
      "7 Day Return (%)",
      "MTD Return (%)",
      "YTD Return (%)",
      "90 Day Return (%)",
    ],
  },
  "monthly_heatmap_data.csv": {
    required: [
      "Year",
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
      "Yearly",
    ],
    numeric: [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
      "Yearly",
    ],
  },
  "relative_value_comparison.csv": {
    required: [
      "Asset",
      "Market Cap (USD)",
      "BTC Price at Market Cap",
      "Move Needed (%)",
    ],
    numeric: ["Market Cap (USD)", "BTC Price at Market Cap", "Move Needed (%)"],
  },
  "roi_table.csv": {
    required: ["Time Frame", "ROI (%)", "Start Price"],
    numeric: ["ROI (%)", "Start Price"],
  },
  "onchain_price_models.csv": {
    required: [
      "date",
      "BTC Price",
      "Realized Price",
      "STH Realized Price",
      "3x Realized Price",
      "3-month MA",
      "1-year MA",
      "200-week MA",
    ],
    numeric: [
      "BTC Price",
      "Realized Price",
      "STH Realized Price",
      "3x Realized Price",
      "3-month MA",
      "1-year MA",
      "200-week MA",
    ],
  },
  "price_outlook.csv": {
    required: ["label", "price", "type", "color", "outlook_year"],
    numeric: ["price", "outlook_year"],
  },
  "bitcoin_candles.csv": {
    required: [
      "interval",
      "period_start",
      "period_end",
      "observation_date",
      "complete",
      "Open",
      "High",
      "Low",
      "Close",
    ],
    numeric: ["Open", "High", "Low", "Close"],
  },
};
export function validateTables(t: Tables) {
  for (const name of inputFiles.map((n) => n.replace(".gz", ""))) {
    const rows = t[name];
    if (!rows?.length) throw new Error(`Missing table ${name}`);
    const specification = fields[name] ?? {
      required: [name.startsWith("mtd") ? "day" : "day_of_year"],
      numeric: Object.keys(rows[0]),
    };
    for (const key of specification.required)
      if (!(key in rows[0])) throw new Error(`${name}: missing field ${key}`);
    for (const [i, row] of rows.entries())
      for (const key of specification.numeric)
        if (row[key]?.trim() && num(row[key]) === null)
          throw new Error(`${name}: invalid number in ${key}, row ${i + 2}`);
  }
}
export function iso(value: string): string {
  if (!/^\d{4}-\d{2}-\d{2}(?:[ T].*)?$/.test(value))
    throw new Error(`Invalid date ${value}`);
  const label = value.slice(0, 10);
  const d = new Date(label + "T00:00:00Z");
  if (!Number.isFinite(+d) || d.toISOString().slice(0, 10) !== label)
    throw new Error(`Invalid date ${value}`);
  return label;
}
const needed = (value: unknown, label: string): number => {
  const n = num(value);
  if (n === null) throw new Error(`Missing numeric ${label}`);
  return n;
};
const series = (
  id: string,
  name: string,
  values: (number | null)[],
  color: string,
  role = "normal",
): Series => ({
  id,
  name,
  values,
  color,
  role,
  axis: "right",
  lineWidth: role === "highlight" ? 3 : 2,
  lineStyle: "solid",
  opacity: 1,
  start: 0,
});
function baseChart(id: string, title: string, date: string): ChartPayload {
  return {
    schemaVersion: 2,
    id,
    title,
    description: "",
    category: "Market Intelligence",
    source: "Bitview",
    reportDate: date,
    coverage: date,
    axisKind: "time",
    axes: {
      right: { unit: "USD", label: "Bitcoin Price (USD)", mode: "linear" },
    },
    defaultRange: "ALL",
    x: [],
    series: [],
    events: [],
    unavailable: [],
    readingPoint: date,
    note: "Daily observations.",
    family: "timeseries",
    gridlines: false,
  };
}
function futureCalendar(
  start: string,
  end: string,
  interval = "daily",
): string[] {
  const result: string[] = [],
    d = new Date(start + "T00:00:00Z");
  for (;;) {
    if (interval === "monthly") d.setUTCMonth(d.getUTCMonth() + 1, 1);
    else d.setUTCDate(d.getUTCDate() + (interval === "weekly" ? 7 : 1));
    const label = d.toISOString().slice(0, 10);
    if (label > end) return result;
    result.push(label);
  }
}
export function pricePayload(
  t: Tables,
  date: string,
  colors: Record<string, string>,
  events: { name: string; dates: string[] }[],
): ChartPayload {
  if (iso(t["onchain_price_models.csv"].at(-1)!.date) !== date)
    throw new Error("Price history cutoff differs from the report date");
  const rows = t["onchain_price_models.csv"].filter((r) => iso(r.date) <= date);
  const x = rows.map((r) => iso(r.date));
  if (
    x.at(-1) !== date ||
    new Set(x).size !== x.length ||
    x.some((d, i) => i > 0 && d <= x[i - 1])
  )
    throw new Error(
      "Price history must be unique, ordered and reach the report date",
    );
  const mapping = [
    ["price_close", "BTC Price", "Bitcoin Price"],
    ["realized_price", "Realized Price", "Realized Price"],
    ["sth_realized_price", "STH Realized Price", "STH Realized Price"],
    ["realizedcap_multiple_3", "3x Realized Price", "3× Realized Price"],
    ["90_day_ma_price_close", "3-month MA", "3-month MA"],
    ["364_day_ma_price_close", "1-year MA", "1-year MA"],
    ["200_week_ma_price_close", "200-week MA", "200-week MA"],
  ];
  const p = baseChart(
    "dashboard-price-outlook",
    `Secret Satoshis ${date.slice(0, 4)} Price Outlook`,
    date,
  );
  p.series = mapping.map(([id, key, name]) =>
    series(
      id,
      name,
      rows.map((r) => num(r[key])),
      colors[id],
      id === "price_close" ? "highlight" : "normal",
    ),
  );
  if (p.series.find((s) => s.id === "price_close")!.values.at(-1) === null)
    throw new Error("Bitcoin price must reach the report date");
  p.unavailable = p.series
    .filter((s) => s.values.at(-1) === null)
    .map((s) => s.name);
  p.series.sort((a, b) =>
    a.id === "price_close"
      ? -1
      : b.id === "price_close"
        ? 1
        : (b.values.at(-1) ?? -Infinity) - (a.values.at(-1) ?? -Infinity),
  );
  p.seriesOrder = p.series.map((s) => s.id);
  p.events = [
    ...events.flatMap((e) => e.dates.map((d) => ({ date: d, name: e.name }))),
    { date: date.slice(0, 4) + "-01-01", name: date.slice(0, 4) + " Start" },
  ]
    .filter((e) => e.date >= x[0] && e.date <= date)
    .sort((a, b) => a.date.localeCompare(b.date));
  const end = date.slice(0, 4) + "-12-31";
  p.x = [...x, ...futureCalendar(date, end)];
  p.rangeEndDate = end;
  p.coverage = `${x[0]}/${date}`;
  p.defaultRange = "4Y";
  p.defaultPresentation = "candles";
  p.defaultInterval = "weekly";
  p.referenceLines = t["price_outlook.csv"]
    .filter((r) => r.type === "case")
    .map((r) => ({
      name: r.label,
      price: needed(r.price, "case price"),
      color: r.color,
    }))
    .sort((a, b) => a.price - b.price);
  p.candleViews = {};
  const byDate = new Map(x.map((d, i) => [d, i]));
  for (const interval of ["daily", "weekly", "monthly"]) {
    const source = t["bitcoin_candles.csv"].filter(
      (r) =>
        r.interval === interval &&
        iso(r.period_start) >= x[0] &&
        iso(r.observation_date) <= date,
    );
    const candles = source.map((r) => ({
      time: iso(r.period_start),
      periodEnd: iso(r.period_end),
      observationDate: iso(r.observation_date),
      complete: r.complete.toLowerCase() === "true",
      open: needed(r.Open, "open"),
      high: needed(r.High, "high"),
      low: needed(r.Low, "low"),
      close: needed(r.Close, "close"),
    }));
    if (!candles.length || candles.at(-1)!.observationDate !== date)
      throw new Error(`${interval} candles must reach ${date}`);
    if (
      source.some(
        (r) => !["true", "false"].includes(r.complete.toLowerCase()),
      ) ||
      candles.some(
        (c, i) =>
          !byDate.has(c.observationDate) ||
          c.time > c.observationDate ||
          c.observationDate > c.periodEnd ||
          c.complete !== c.periodEnd <= date ||
          (i > 0 && c.time <= candles[i - 1].time) ||
          c.high < Math.max(c.open, c.close, c.low) ||
          c.low > Math.min(c.open, c.close, c.high),
      )
    )
      throw new Error(`Invalid ${interval} candles`);
    if (
      interval === "daily" &&
      candles.some(
        (c) =>
          Math.abs(
            c.close -
              needed(
                rows[byDate.get(c.observationDate)!]["BTC Price"],
                "daily price",
              ),
          ) > 0.001,
      )
    )
      throw new Error("Candle and model prices disagree");
    const dates = candles.map((c) => c.time);
    p.candleViews[interval] = {
      x: [...dates, ...futureCalendar(dates.at(-1)!, end, interval)],
      candles,
      series: p.series.map((s) => ({
        ...s,
        values: candles.map(
          (c) => s.values[byDate.get(c.observationDate)!] ?? null,
        ),
      })),
      events: p.events.flatMap((e) => {
        const c = candles.find(
          (c) => c.time <= e.date && c.observationDate >= e.date,
        );
        return c ? [{ ...e, date: c.time, originalDate: e.date }] : [];
      }),
      readingPoint: dates.at(-1)!,
      interval,
    };
    if (interval === "daily") {
      p.candleViews[interval].offset = x.indexOf(dates[0]);
      p.candleViews[interval].ohlc = candles.map((c) => [
        c.open,
        c.high,
        c.low,
        c.close,
      ]);
    }
  }
  return p;
}
// Years left off the seasonal charts and their averages; 2017's scale flattens every other year.
export const HIDDEN_SEASONAL_YEARS = ["2017"];
export function seasonalPayload(
  rows: Row[],
  date: string,
  period: "mtd" | "ytd",
): ChartPayload {
  const key = period === "mtd" ? "day" : "day_of_year",
    year = date.slice(0, 4);
  const years = Object.keys(rows[0])
    .filter((k) => /^\d{4}$/.test(k) && !HIDDEN_SEASONAL_YEARS.includes(k))
    .sort();
  if (years.at(-1) !== year)
    throw new Error(`${period} paths do not match the report year`);
  const p = baseChart(
    `dashboard-seasonal-${period}`,
    `Bitcoin ${period === "mtd" ? new Date(date + "T00:00:00Z").toLocaleDateString("en-US", { month: "long", timeZone: "UTC" }) : year} ${period.toUpperCase()} Returns Comparison`,
    date,
  );
  p.family = "seasonal";
  p.axisKind = "days";
  p.x = rows.map((r) => needed(r[key], key));
  if (
    p.x[0] !== 0 ||
    p.x.some((x, i) => i > 0 && Number(x) !== Number(p.x[i - 1]) + 1)
  )
    throw new Error("Seasonal calendar must start at zero and be contiguous");
  const last = rows.findLastIndex((r) => num(r[year]) !== null);
  const d = new Date(date + "T00:00:00Z");
  const ordinal =
    Math.floor((+d - Date.UTC(d.getUTCFullYear(), 0, 1)) / 86400000) + 1;
  const leap =
    d.getUTCFullYear() % 4 === 0 &&
    (d.getUTCFullYear() % 100 !== 0 || d.getUTCFullYear() % 400 === 0);
  const expected =
    period === "mtd"
      ? d.getUTCDate()
      : ordinal - (leap && d.getUTCMonth() >= 2 ? 1 : 0);
  // February 29 has no observation on the shared 365-day calendar.
  const position =
    period === "ytd" && leap && d.getUTCMonth() === 1 && d.getUTCDate() === 29
      ? 59
      : expected;
  if (last !== position)
    throw new Error(`${period} current path must end at the report cutoff`);
  const past = years.filter((y) => y !== year);
  p.series = years.map((y, i) =>
    series(
      y,
      y,
      rows.map((r) => num(r[y])),
      y === year
        ? "#F7931A"
        : `hsl(214, ${Math.round(44 + (i / Math.max(1, past.length - 1)) * 28)}%, ${Math.round(34 + (i / Math.max(1, past.length - 1)) * 40)}%)`,
      y === year ? "highlight" : "historical",
    ),
  );
  const values = rows.map((r) =>
    past
      .map((y) => num(r[y]))
      .filter((n): n is number => n !== null)
      .sort((a, b) => a - b),
  );
  p.series.push(
    series(
      "median",
      "Median",
      values.map((v) =>
        v.length
          ? v.length % 2
            ? v[Math.floor(v.length / 2)]
            : (v[v.length / 2 - 1] + v[v.length / 2]) / 2
          : null,
      ),
      "#e4e4ef",
      "median",
    ),
    series(
      "mean",
      "Average",
      values.map((v) =>
        v.length ? v.reduce((a, b) => a + b, 0) / v.length : null,
      ),
      "#00FF88",
      "mean",
    ),
  );
  p.series.find((s) => s.id === "median")!.lineStyle = "dashed";
  p.seriesOrder = [year, ...past.slice().reverse(), "median", "mean"];
  p.readingPoint = position;
  p.xAxisLabel = period === "mtd" ? "Day of Month" : "Day of Year";
  p.axes.right.label =
    period === "mtd"
      ? "Indexed to Month Start ($)"
      : "Indexed to Year Start ($)";
  p.note =
    `Historical paths are rebased comparisons. Average and Median exclude ${HIDDEN_SEASONAL_YEARS.join(", ")} and the current year. YTD removes February 29.`;
  p.coverage = `2014/${date}`;
  p.gridlines = true;
  return p;
}
export function createDashboard(
  t: Tables,
  release: ReleaseManifest,
  colors: Record<string, string>,
  events: { name: string; dates: string[] }[],
): DashboardData {
  validateTables(t);
  const date = iso(release.report_date),
    year = date.slice(0, 4),
    d = new Date(date + "T00:00:00Z");
  if (release.schema_version !== 1 || release.release_id !== date)
    throw new Error("Invalid release manifest");
  if (t["price_outlook.csv"].some((r) => r.outlook_year !== year))
    throw new Error("Outlook year differs from report year");
  const cases = t["price_outlook.csv"]
    .filter((r) => r.type === "case")
    .sort((a, b) => needed(a.price, "price") - needed(b.price, "price"));
  if (cases.length !== 3 || new Set(cases.map((r) => r.label)).size !== 3)
    throw new Error("Three unique annual cases are required");
  const definitions = [
    ["Bitcoin Price USD", "Bitcoin Price", "usd", "BTC daily close in USD."],
    [
      "Bitcoin Market Cap",
      "Bitcoin Market Cap",
      "trillion",
      "Bitcoin market capitalization in USD.",
    ],
    [
      "Sats Per Dollar",
      "Sats Per Dollar",
      "number",
      "Satoshis one US dollar buys.",
    ],
    [
      "Bitcoin Supply",
      "Bitcoin Supply",
      "number",
      "Circulating Bitcoin supply.",
    ],
    [
      "Bitcoin Miner Revenue",
      "Bitcoin Miner Revenue",
      "million",
      "Miner revenue for the report day, USD.",
    ],
    [
      "Bitcoin Transaction Volume",
      "Bitcoin Transaction Volume",
      "billion",
      "On-chain transfer volume for the report day, USD.",
    ],
  ];
  const metrics = definitions.map(([name, title, format, description]) => {
    const history = t["summary_history.csv"]
      .filter((r) => r.Metric === name)
      .map((r) => ({ date: iso(r.date), value: num(r.Value) }))
      .sort((a, b) => a.date.localeCompare(b.date));
    if (
      history.length !== 31 ||
      history.some(
        (r, i) =>
          r.date !==
          new Date(+d - (30 - i) * 86400000).toISOString().slice(0, 10),
      )
    )
      throw new Error(
        `${name}: history must contain 31 unique dates through ${date}`,
      );
    const prior = new Date(+d - 30 * 86400000).toISOString().slice(0, 10),
      value = history.at(-1)!.value,
      start = history.find((r) => r.date === prior)?.value;
    const snapshot = t["summary_table.csv"].find((r) => r.Metric === name);
    const snapshotValue = num(snapshot?.Value);
    if (
      !snapshot ||
      (value === null) !== (snapshotValue === null) ||
      (value !== null &&
        snapshotValue !== null &&
        Math.abs(value - snapshotValue) >
          Math.max(1e-5, Math.abs(value) * 1e-8))
    )
      throw new Error(`${name}: snapshot disagreement`);
    return {
      name,
      title,
      format,
      description,
      history,
      value,
      change:
        value !== null && start != null && start !== 0
          ? (value - start) / start
          : null,
      downIsGood: name === "Sats Per Dollar",
    };
  });
  const summary = (name: string) =>
    t["summary_table.csv"].find((r) => r.Metric === name);
  const performance = [
    [
      "performance-indexes",
      "Stock Market Index Performance",
      "Equity Market Indexes",
    ],
    ["performance-sectors", "Sector Performance", "Sectors"],
    [
      "performance-macro",
      "Macro Asset Class Performance",
      "Macro Asset Classes",
    ],
    [
      "performance-bitcoin",
      "Bitcoin Industry Performance",
      "Bitcoin Industry Performance",
    ],
  ].map(([id, title, category]) => ({
    id,
    title,
    rows: t["performance_table.csv"]
      .filter((r) => r.Category === category || r.Asset === "Bitcoin - [BTC]")
      .sort((a, b) =>
        a.Asset === "Bitcoin - [BTC]"
          ? -1
          : b.Asset === "Bitcoin - [BTC]"
            ? 1
            : (num(b["7 Day Return (%)"]) ?? -Infinity) -
              (num(a["7 Day Return (%)"]) ?? -Infinity),
      ),
  }));
  const heat = t["monthly_heatmap_data.csv"];
  const btcCap = num(
    t["relative_value_comparison.csv"].find((r) => r.Asset === "Bitcoin")?.[
      "Market Cap (USD)"
    ],
  );
  return {
    reportDate: date,
    year,
    month: d.toLocaleDateString("en-US", { month: "long", timeZone: "UTC" }),
    dateLabel: d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      timeZone: "UTC",
    }),
    metrics,
    sentiment: {
      supplyInProfit: num(summary("Bitcoin Supply in Profit (%)")?.Value),
      sentiment: summary("Bitcoin Market Sentiment")?.Label || "Unavailable",
      valuation: summary("Bitcoin Valuation")?.Label || "Unavailable",
    },
    performance,
    heatmap: {
      reference: ["Average", "Median", "4-Year Average"]
        .map((y) => heat.find((r) => r.Year === y)!)
        .filter(Boolean),
      years: heat
        .filter((r) => /^\d{4}$/.test(r.Year))
        .sort((a, b) => Number(b.Year) - Number(a.Year)),
    },
    relative: t["relative_value_comparison.csv"]
      .map<
        Row & { btcShare: string }
      >((r) => ({ ...r, btcShare: r.Asset === "Bitcoin" || btcCap === null || !num(r["Market Cap (USD)"]) ? "" : String((btcCap / needed(r["Market Cap (USD)"], "cap")) * 100) }))
      .sort(
        (a, b) =>
          (num(b["Market Cap (USD)"]) ?? -Infinity) -
          (num(a["Market Cap (USD)"]) ?? -Infinity),
      ),
    fundamentals: t["fundamentals_table.csv"]
      .slice()
      .sort((a, b) => a.Section.localeCompare(b.Section)),
    roi: t["roi_table.csv"],
    cases,
    charts: [
      pricePayload(t, date, colors, events),
      seasonalPayload(t["mtd_price_paths.csv"], date, "mtd"),
      seasonalPayload(t["ytd_price_paths.csv"], date, "ytd"),
    ],
  };
}
