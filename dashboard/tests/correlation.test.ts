import { test } from "node:test";
import assert from "node:assert/strict";
import { CORRELATION_FILE, correlationGroups, correlationPeriods, readCorrelationMatrix, releaseInputFiles, inputFiles, type Row, type ReleaseManifest } from "../src/data";
import { correlationMatrix } from "../src/render";

const date = "2026-09-30";
const tickers = correlationGroups.flatMap((g) => g.tickers);
function rows(): Row[] {
  return correlationPeriods.flatMap((period) => correlationGroups.flatMap((g) => g.tickers.map((ticker) => ({
    "Report Date": date, "Window Days": String(period), Category: g.category, Asset: ticker === "BTC" ? "Bitcoin - [BTC]" : ticker === "COIN" ? "Coinbase - [COIN]" : ticker, Ticker: ticker,
    ...Object.fromEntries(tickers.map((other) => [other, ticker === other ? "1" : period === 30 ? "0.2" : period === 90 ? "0.42" : "0.65"])),
  }))));
}
test("matrix preserves group order and renders the lower triangle with 153 values", () => {
  const ordered = readCorrelationMatrix({ [CORRELATION_FILE]: rows().reverse() }, date)![90];
  assert.deepEqual(ordered.map((r) => r.Ticker), tickers);
  const html = correlationMatrix(ordered);
  assert.equal((html.match(/<td /g) ?? []).length, 289);
  assert.equal((html.match(/class="correlation-unused"/g) ?? []).length, 136);
  assert.equal((html.match(/style="background:rgb/g) ?? []).length, 153);
  assert.equal((html.match(/scope="colgroup"/g) ?? []).length, 5);
  assert.equal((html.match(/scope="rowgroup"/g) ?? []).length, 5);
  assert.ok(!html.includes('title="SPY / QQQ: 0.42"'));
  assert.ok(html.includes('title="QQQ / SPY: 0.42"'));
  assert.ok(html.includes("rgb(26,142,78)"));
  assert.ok(!html.includes("0.42%"));
  assert.ok(html.includes("DXY = DX-Y.NYB"));
  assert.ok(html.includes('>Bitcoin</span><span class="correlation-symbol">[BTC]</span>'));
  assert.ok(html.includes('>Coinbase</span><span class="correlation-symbol">[COIN]</span>'));
});
test("matrix rejects invalid numbers, mismatched windows, duplicates and asymmetric blanks", () => {
  const changes: [(r: Row[]) => void, RegExp][] = [
    [(r) => { r[0].SPY = "1.2"; }, /outside/],
    [(r) => { r[0].SPY = "not-a-number"; }, /invalid correlation number/],
    [(r) => { r[0].BTC = "0.8"; }, /diagonal/],
    [(r) => { r[0].SPY = ""; }, /symmetric/],
    [(r) => { r[0]["Report Date"] = "2026-09-29"; }, /report-date/],
    [(r) => { r[0]["Window Days"] = "15"; }, /30\/90\/365-day/],
    [(r) => { r[1].Ticker = "BTC"; }, /one row per asset/],
    [(r) => { r[1].Category = "Sectors"; }, /asset group/],
  ];
  for (const [change, message] of changes) {
    const matrix = rows(); change(matrix);
    assert.throws(() => readCorrelationMatrix({ [CORRELATION_FILE]: matrix }, date), message);
  }
});
test("unavailable pairs remain blank, while valid negative correlations use heatmap red", () => {
  const matrix = rows();
  for (const period of correlationPeriods) {
    matrix.find((r) => r.Ticker === "BTC" && Number(r["Window Days"]) === period)!.SPY = "-1";
    matrix.find((r) => r.Ticker === "SPY" && Number(r["Window Days"]) === period)!.BTC = "-1";
  }
  for (const row of matrix) row.WGMI = "";
  for (const row of matrix.filter((r) => r.Ticker === "WGMI"))
    for (const ticker of tickers) row[ticker] = "";
  const html = correlationMatrix(readCorrelationMatrix({ [CORRELATION_FILE]: matrix }, date)![90]);
  assert.ok(html.includes("rgb(214,72,66)"));
  assert.ok(html.includes('title="WGMI / WGMI: Unavailable">—'));
});
test("Bitcoin correlations agree with the performance table", () => {
  assert.throws(() => readCorrelationMatrix({
    [CORRELATION_FILE]: rows(),
    "performance_table.csv": [{ Asset: "SPY", "90 Day BTC Correlation": "0.8" }],
  }, date), /differs from performance table/);
});
test("each period keeps its own coefficients and requires all three complete windows", () => {
  const matrix = readCorrelationMatrix({ [CORRELATION_FILE]: rows() }, date)!;
  assert.equal(matrix[30][1].BTC, "0.2");
  assert.equal(matrix[90][1].BTC, "0.42");
  assert.equal(matrix[365][1].BTC, "0.65");
  assert.ok(correlationMatrix(matrix[365], 365).includes('aria-label="365-day asset correlation matrix"'));
  assert.throws(() => readCorrelationMatrix({ [CORRELATION_FILE]: rows().slice(17) }, date), /each 30\/90\/365-day window/);
});
test("old releases remain readable and manifests opt into the new CSV", () => {
  const release = { files: {} } as ReleaseManifest;
  assert.deepEqual(releaseInputFiles(release), inputFiles);
  assert.equal(readCorrelationMatrix({}, date), null);
  release.files[CORRELATION_FILE] = { sha256: "hash", size_bytes: 10 };
  assert.ok(releaseInputFiles(release).includes(CORRELATION_FILE));
});
