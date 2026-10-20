import { test } from "node:test";
import assert from "node:assert/strict";
import {
  RELATIVE_RANK_FILE, readRelativeRankChanges, relativeRankBaselines,
  relativeRankPeriods, releaseInputFiles, inputFiles, type Row, type ReleaseManifest,
} from "../src/data";
import { rankChip } from "../src/render";

const date = "2026-10-11";
const assets = [
  { Asset: "Apple", "Market Cap (USD)": "30" },
  { Asset: "Bitcoin", "Market Cap (USD)": "20" },
  { Asset: "NVIDIA", "Market Cap (USD)": "20" },
  { Asset: "SpaceX", "Market Cap (USD)": "10" },
  { Asset: "Unavailable", "Market Cap (USD)": "" },
];
function tables(reportDate = date) {
  const baselines = relativeRankBaselines(reportDate);
  const rows: Row[] = relativeRankPeriods.flatMap((period) => assets.map((asset, i) => ({
    Asset: asset.Asset, Timeframe: period, "Current Date": reportDate, "Baseline Date": baselines[period],
    "Current Rank": ["1", "2", "2", "4", ""][i],
    "Previous Rank": ["2", "1", "2", "", "4"][i],
    Change: ["1", "-1", "0", "", ""][i],
    Status: i === 3 ? "missing_baseline" : i === 4 ? "missing_current" : "available",
  })));
  return { "relative_value_comparison.csv": structuredClone(assets), [RELATIVE_RANK_FILE]: rows };
}
test("rank changes retain competition ranks, separate periods and missing data", () => {
  const t = tables();
  t[RELATIVE_RANK_FILE].find((r) => r.Asset === "Apple" && r.Timeframe === "ytd")!["Previous Rank"] = "4";
  t[RELATIVE_RANK_FILE].find((r) => r.Asset === "Apple" && r.Timeframe === "ytd")!.Change = "3";
  const result = readRelativeRankChanges(t, date)!;
  assert.equal(result.wtd.Apple.change, 1);
  assert.equal(result.ytd.Apple.change, 3);
  assert.equal(result.wtd.Bitcoin.change, -1);
  assert.equal(result.wtd.NVIDIA.change, 0);
  assert.equal(result.wtd.SpaceX.change, null);
  assert.equal(result.wtd.Unavailable.status, "missing_current");
});
test("invalid dates, ranks, availability, duplicates and incomplete periods are rejected", () => {
  const mutations: [(r: Row[]) => void, RegExp][] = [
    [(r) => { r[0]["Current Date"] = "2032-01-01"; }, /dates/],
    [(r) => { r[0]["Baseline Date"] = "2026-10-10"; r[0].Timeframe = "wtd"; }, /dates/],
    [(r) => { r[0]["Current Rank"] = "2"; }, /disagrees/],
    [(r) => { r[0]["Previous Rank"] = "1.5"; }, /invalid Previous Rank/],
    [(r) => { r[0]["Previous Rank"] = "6"; }, /invalid Previous Rank/],
    [(r) => { r[0].Change = "0"; }, /rank change/],
    [(r) => { r[0].Change = "bad"; }, /rank change/],
    [(r) => { r[3].Status = "available"; }, /availability/],
    [(r) => { r[3].Change = "0"; }, /rank change/],
    [(r) => { r[1] = { ...r[0] }; }, /duplicate/],
    [(r) => { r[0].Timeframe = "monthly"; }, /timeframe/],
    [(r) => { delete r[0].Status; }, /columns/],
    [(r) => { r.pop(); }, /comparisons/],
  ];
  for (const [mutate, message] of mutations) {
    const t = tables(); mutate(t[RELATIVE_RANK_FILE]);
    assert.throws(() => readRelativeRankChanges(t, date), message);
  }
});
test("calendar baselines handle week, quarter, year and leap-day boundaries", () => {
  for (const [date, expected] of [
    ["2026-10-12", ["2026-10-11", "2026-10-11", "2026-09-30", "2025-12-31", "2026-10-05"]],
    ["2026-10-11", ["2026-10-10", "2026-10-04", "2026-09-30", "2025-12-31", "2026-10-04"]],
    ["2026-10-01", ["2026-09-30", "2026-09-27", "2026-09-30", "2025-12-31", "2026-09-24"]],
    ["2024-01-01", ["2023-12-31", "2023-12-31", "2023-12-31", "2023-12-31", "2023-12-25"]],
    ["2024-03-01", ["2024-02-29", "2024-02-25", "2023-12-31", "2023-12-31", "2024-02-23"]],
    ["2024-04-01", ["2024-03-31", "2024-03-31", "2024-03-31", "2023-12-31", "2024-03-25"]],
  ] as const) assert.deepEqual(Object.values(relativeRankBaselines(date)).slice(0, 5), expected);
});
test("rolling windows use calendar days rather than month, quarter or year boundaries", () => {
  for (const [report, expected] of [
    ["2024-03-01", ["2024-02-23", "2024-01-31", "2023-12-02", "2023-03-02"]],
    ["2026-10-12", ["2026-10-05", "2026-09-12", "2026-07-14", "2025-10-12"]],
  ] as const) {
    const baselines = relativeRankBaselines(report);
    assert.deepEqual([baselines["7d"], baselines["30d"], baselines["90d"], baselines["365d"]], expected);
  }
  const t = tables();
  t[RELATIVE_RANK_FILE].find(row => row.Asset === "Apple" && row.Timeframe === "30d")!.Change = "3";
  t[RELATIVE_RANK_FILE].find(row => row.Asset === "Apple" && row.Timeframe === "30d")!["Previous Rank"] = "4";
  const ranks = readRelativeRankChanges(t, date)!;
  assert.equal(ranks["7d"].Apple.change, 1);
  assert.equal(ranks["30d"].Apple.change, 3);
  assert.equal(ranks["90d"].Apple.change, 1);
  assert.equal(ranks["365d"].Apple.change, 1);
});
test("seven-day history uses its exact baseline and never substitutes legacy WTD data", () => {
  const t = tables("2026-10-12");
  assert.equal(readRelativeRankChanges(t, "2026-10-12")!["7d"].Apple.baselineDate, "2026-10-05");
  const wrong = structuredClone(t);
  wrong[RELATIVE_RANK_FILE].find(row => row.Timeframe === "7d")!["Baseline Date"] = "2026-10-11";
  assert.throws(() => readRelativeRankChanges(wrong, "2026-10-12"), /dates/);
  const intermediate = structuredClone(t);
  intermediate[RELATIVE_RANK_FILE] = intermediate[RELATIVE_RANK_FILE].filter(row => !["30d", "90d", "365d"].includes(row.Timeframe));
  const previous = readRelativeRankChanges(intermediate, "2026-10-12")!;
  assert.equal(previous["7d"].Apple.baselineDate, "2026-10-05");
  assert.deepEqual(Object.keys(previous["30d"]), []);
  t[RELATIVE_RANK_FILE] = t[RELATIVE_RANK_FILE].filter(row => ["daily", "wtd", "qtd", "ytd"].includes(row.Timeframe));
  const legacy = readRelativeRankChanges(t, "2026-10-12")!;
  assert.deepEqual(Object.keys(legacy["7d"]), []);
  assert.equal(legacy.wtd.Apple.baselineDate, "2026-10-11");
  assert.match(rankChip("Apple", "7d", legacy["7d"].Apple), /unavailable.*7 Day comparison.*>N\/A<\/button>/);
});
test("chips show directions, unchanged and unavailable with accessible explanations", () => {
  const ranks = readRelativeRankChanges(tables(), date)!.wtd;
  assert.match(rankChip("Apple", "wtd", ranks.Apple), /positive.*Moved from #2 to #1.*>↑ 1<\/button>/);
  assert.match(rankChip("Bitcoin", "wtd", ranks.Bitcoin), /negative.*>↓ 1<\/button>/);
  assert.match(rankChip("NVIDIA", "wtd", ranks.NVIDIA), /flat.*Unchanged at #2.*>—<\/button>/);
  assert.match(rankChip("SpaceX", "wtd", ranks.SpaceX), /unavailable.*baseline.*>N\/A<\/button>/);
  assert.match(rankChip("<asset>", "wtd"), /aria-label="&lt;asset&gt;: Rank history is unavailable/);
  assert.match(rankChip("Apple", "wtd", ranks.Apple), /hidden>↑ 1/);
  assert.doesNotMatch(rankChip("Apple", "7d", ranks.Apple), /hidden/);
  assert.match(rankChip("Apple", "365d", ranks.Apple), /over the past 365 days.*hidden/);
});
test("older manifests omit rank data and keep the existing inputs readable", () => {
  const release = { files: {} } as ReleaseManifest;
  assert.deepEqual(releaseInputFiles(release), inputFiles);
  assert.equal(readRelativeRankChanges({}, date), null);
  release.files[RELATIVE_RANK_FILE] = { sha256: "hash", size_bytes: 20 };
  assert.ok(releaseInputFiles(release).includes(RELATIVE_RANK_FILE));
});
