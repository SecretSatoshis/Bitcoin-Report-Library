import { test, after } from "node:test";
import assert from "node:assert/strict";
import {
  mkdtempSync,
  readFileSync,
  writeFileSync,
  rmSync,
  cpSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";
import { loadVerified, prepareData, hash, root } from "../scripts/prepare-data";
import {
  createDashboard,
  seasonalPayload,
  parseCSV,
  num,
  iso,
  validateTables,
  type Row,
  CORRELATION_FILE,
} from "../src/data";
import { renderDashboard, format, escape } from "../src/render";

const directory = mkdtempSync(resolve(tmpdir(), "ss-dashboard-reference-"));
execFileSync("tar", [
  "-xzf",
  resolve(root, "tests/fixtures/release-2026-09-29.tar.gz"),
  "-C",
  directory,
]);
after(() => rmSync(directory, { recursive: true, force: true }));
const { tables, release } = loadVerified(directory);
const colors = JSON.parse(
  readFileSync(resolve(root, "components/chart-colors.json"), "utf8"),
);
const events = JSON.parse(
  readFileSync(resolve(root, "components/chart-events.json"), "utf8"),
);
const reference = JSON.parse(
  readFileSync(
    resolve(root, "tests/fixtures/reference-2026-09-29.json"),
    "utf8",
  ),
);
const data = () =>
  createDashboard(structuredClone(tables), release, colors, events);

test("the new matrix is loaded only from its release manifest and hash checked", () => {
  const copy = mkdtempSync(resolve(tmpdir(), "ss-correlation-release-"));
  try {
    cpSync(directory, copy, { recursive: true });
    const csv = "Ticker,BTC\nBTC,1\n";
    const manifest = structuredClone(release);
    manifest.files[CORRELATION_FILE] = { sha256: hash(csv), size_bytes: Buffer.byteLength(csv) };
    writeFileSync(resolve(copy, CORRELATION_FILE), csv);
    writeFileSync(resolve(copy, "release_manifest.json"), JSON.stringify(manifest));
    assert.equal(loadVerified(copy).tables[CORRELATION_FILE][0].Ticker, "BTC");
    writeFileSync(resolve(copy, CORRELATION_FILE), "Ticker,BTC\nBTC,0\n");
    assert.throws(() => loadVerified(copy), /manifest hash\/size mismatch/);
  } finally {
    rmSync(copy, { recursive: true, force: true });
  }
});

test("frozen release: all seven price series, ranges, events and candle payloads agree with the reference", () => {
  const p = data().charts[0];
  for (const key of [
    "x",
    "events",
    "axes",
    "referenceLines",
    "defaultInterval",
    "defaultPresentation",
    "defaultRange",
    "readingPoint",
    "rangeEndDate",
  ] as const)
    assert.deepEqual(p[key], reference.price[key], key);
  for (const s of p.series) {
    const expected = reference.price.series.find(
      (e: { id: string }) => e.id === s.id,
    );
    assert.equal(hash(JSON.stringify(s.values)), expected.valuesHash, s.id);
    assert.equal(s.values.length, expected.valuesLength);
  }
  // Key order is part of the frozen payload fingerprint.
  for (const [interval, view] of Object.entries(p.candleViews!)) {
    const ordered = {
      x: view.x,
      series: view.series.map((s) => ({
        id: s.id,
        name: s.name,
        axis: s.axis,
        role: s.role,
        color: reference.price.series.find((e: { id: string }) => e.id === s.id)
          .color,
        lineWidth: s.lineWidth,
        lineStyle: s.lineStyle,
        opacity: s.opacity,
        start: s.start,
        values: s.values,
      })),
      candles: view.candles,
      events: view.events,
      readingPoint: view.readingPoint,
      interval: view.interval,
      ...(view.offset !== undefined
        ? { offset: view.offset, ohlc: view.ohlc }
        : {}),
    };
    assert.equal(
      hash(JSON.stringify(ordered)),
      reference.price.candleViews[interval].hash,
      interval,
    );
  }
});
test("every seasonal observation, average and median agrees with the original aggregator", () => {
  for (const p of data().charts.slice(1)) {
    const period = p.id.endsWith("mtd") ? "mtd" : "ytd",
      expected = reference.seasonal[period + "_price_paths.csv"];
    for (const s of p.series) {
      const name =
        s.id === "mean" ? "Average" : s.id === "median" ? "Median" : s.id;
      assert.deepEqual(
        s.values,
        expected.map((r: Row) => r[name]),
        name,
      );
    }
    assert.equal(p.x[0], 0);
    assert.equal(
      p.series.some((s) => s.id === "2017"),
      false,
    );
  }
});
test("all snapshot values and exact 30-calendar-day comparisons agree with the release", () => {
  for (const m of data().metrics) {
    const current = tables["summary_history.csv"].find(
        (r) => r.Metric === m.name && r.date.startsWith("2026-09-29"),
      )!,
      prior = tables["summary_history.csv"].find(
        (r) => r.Metric === m.name && r.date.startsWith("2026-08-30"),
      )!;
    assert.equal(m.value, Number(current.Value));
    assert.equal(
      m.change,
      (Number(current.Value) - Number(prior.Value)) / Number(prior.Value),
    );
    assert.ok(
      reference.snapshot.includes(
        format(
          m.value,
          m.format,
          ["million", "billion", "trillion"].includes(m.format) ? 2 : 0,
        ),
      ),
    );
  }
});
test("CSV parsing preserves nulls, percentages and text; malformed records are rejected", () => {
  assert.equal(num(""), null);
  assert.equal(num("  "), null);
  assert.equal(num(null), null);
  assert.equal(num("0"), 0);
  assert.equal(num("-3.00"), -3);
  assert.deepEqual(
    parseCSV('Asset,Value\n"Gold, spot",\nBTC,1.5\n', "test.csv"),
    [
      { Asset: "Gold, spot", Value: "" },
      { Asset: "BTC", Value: "1.5" },
    ],
  );
  assert.throws(() => parseCSV("A,B\n1,2,3", "test.csv"), /malformed/);
  assert.throws(() => parseCSV("A,A\n1,2", "test.csv"), /duplicate/);
  const broken = structuredClone(tables);
  broken["performance_table.csv"][0].Price = "bad";
  assert.throws(() => validateTables(broken), /invalid number/);
  delete broken["roi_table.csv"];
  assert.throws(() => validateTables(broken));
});
test("seasonal calendars use release cutoffs across month/year boundaries and leap days", () => {
  for (const [date, position] of [
    ["2024-02-28", 59],
    ["2024-02-29", 59],
    ["2024-03-01", 60],
    ["2024-12-31", 365],
    ["2025-01-01", 1],
    ["2025-12-31", 365],
  ] as const) {
    const year = date.slice(0, 4),
      rows = Array.from({ length: 366 }, (_, day) => ({
        day_of_year: String(day),
        "2017": "999999",
        "2020": "100",
        "2021": "120",
        [year]: day <= position ? "500" : "",
      }));
    const p = seasonalPayload(rows, date, "ytd");
    assert.equal(p.readingPoint, position);
    assert.equal(p.series.find((s) => s.id === "mean")!.values[0], 110);
    assert.equal(p.series.find((s) => s.id === "median")!.values[0], 110);
  }
  for (const date of ["2024-02-29", "2025-01-01", "2025-04-30", "2025-12-31"]) {
    const day = Number(date.slice(8)),
      rows = Array.from({ length: day + 1 }, (_, i) => ({
        day: String(i),
        "2020": "100",
        [date.slice(0, 4)]: i <= day ? "200" : "",
      }));
    assert.equal(seasonalPayload(rows, date, "mtd").readingPoint, day);
  }
  assert.throws(() => iso("2025-02-29"), /Invalid date/);
});
test("inconsistent cutoffs, calendar gaps and invalid candles fail preparation", () => {
  const broken = structuredClone(tables);
  broken["summary_history.csv"][0].date = "2000-01-01";
  assert.throws(
    () => createDashboard(broken, release, colors, events),
    /31 unique dates/,
  );
  const candle = structuredClone(tables);
  candle["bitcoin_candles.csv"].at(-1)!.complete = "perhaps";
  assert.throws(
    () => createDashboard(candle, release, colors, events),
    /Invalid monthly candles/,
  );
  const chart = structuredClone(tables);
  chart["onchain_price_models.csv"].at(-1)!.date = "2026-09-30";
  assert.throws(
    () => createDashboard(chart, release, colors, events),
    /cutoff/,
  );
  const missing = structuredClone(tables);
  missing["mtd_price_paths.csv"].at(-2)!["2026"] = "";
  assert.throws(
    () => createDashboard(missing, release, colors, events),
    /cutoff/,
  );
});
test("hash mismatch, missing files and malformed CSVs preserve the previous generated site and build", () => {
  const before = readFileSync(resolve(root, "build/index.html")),
    generated = readFileSync(resolve(root, ".generated/index.html"));
  const broken = mkdtempSync(resolve(tmpdir(), "ss-dashboard-broken-"));
  cpSync(directory, broken, { recursive: true });
  try {
    writeFileSync(resolve(broken, "summary_table.csv"), "changed");
    assert.throws(() => prepareData(broken), /hash\/size mismatch/);
    // Even a matching manifest cannot make syntactically broken CSV acceptable.
    const malformed = 'Metric,Value,Label\n"broken';
    writeFileSync(resolve(broken, "summary_table.csv"), malformed);
    const badManifest = structuredClone(release);
    badManifest.files["summary_table.csv"] = {
      sha256: hash(malformed),
      size_bytes: Buffer.byteLength(malformed),
    };
    writeFileSync(
      resolve(broken, "release_manifest.json"),
      JSON.stringify(badManifest),
    );
    assert.throws(() => prepareData(broken), /malformed/);
    rmSync(resolve(broken, "roi_table.csv"));
    assert.throws(() => loadVerified(broken));
    assert.deepEqual(readFileSync(resolve(root, "build/index.html")), before);
    assert.deepEqual(
      readFileSync(resolve(root, ".generated/index.html")),
      generated,
    );
  } finally {
    rmSync(broken, { recursive: true, force: true });
  }
});
test("missing optional metric observations render as unavailable without inventing zero", () => {
  const missing = structuredClone(tables),
    name = "Bitcoin Miner Revenue";
  missing["summary_table.csv"].find((r) => r.Metric === name)!.Value = "";
  missing["summary_history.csv"].find(
    (r) => r.Metric === name && iso(r.date) === release.report_date,
  )!.Value = "";
  const d = createDashboard(missing, release, colors, events);
  assert.equal(d.metrics.find((m) => m.name === name)!.value, null);
  assert.equal(d.metrics.find((m) => m.name === name)!.change, null);
  assert.ok(renderDashboard(d).includes("Unavailable"));
});
test("semantic HTML includes every section, all rows, null labels, and escaped data", () => {
  const d = data(),
    html = renderDashboard(d);
  assert.equal((html.match(/data-chart-frame=/g) || []).length, 3);
  for (const row of [...d.fundamentals, ...d.roi])
    assert.ok(html.includes(escape(row.Metric || row["Time Frame"])));
  d.roi[0]["ROI (%)"] = "";
  assert.ok(renderDashboard(d).includes("Unavailable"));
  assert.equal(escape('<script>"&'), "&lt;script&gt;&quot;&amp;");
  assert.equal(format(null), "Unavailable");
  assert.equal(format(1.23, "percent", 2), "1.23%");
  assert.ok(html.includes("2026 (YTD)"));
  d.reportDate = "2026-12-31";
  assert.ok(!renderDashboard(d).includes("2026 (YTD)"));
});
