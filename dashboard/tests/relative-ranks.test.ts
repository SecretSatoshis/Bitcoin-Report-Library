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
function tables() {
  const baselines = relativeRankBaselines(date);
  const rows: Row[] = relativeRankPeriods.flatMap((period) => assets.map((asset, i) => ({
    Asset: asset.Asset, Timeframe: period, "Current Date": date, "Baseline Date": baselines[period],
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
    [(r) => { r.pop(); }, /four comparisons/],
  ];
  for (const [mutate, message] of mutations) {
    const t = tables(); mutate(t[RELATIVE_RANK_FILE]);
    assert.throws(() => readRelativeRankChanges(t, date), message);
  }
});
test("calendar baselines handle week, quarter, year and leap-day boundaries", () => {
  for (const [date, expected] of [
    ["2026-10-12", ["2026-10-11", "2026-10-11", "2026-09-30", "2025-12-31"]],
    ["2026-10-11", ["2026-10-10", "2026-10-04", "2026-09-30", "2025-12-31"]],
    ["2026-10-01", ["2026-09-30", "2026-09-27", "2026-09-30", "2025-12-31"]],
    ["2024-01-01", ["2023-12-31", "2023-12-31", "2023-12-31", "2023-12-31"]],
    ["2024-03-01", ["2024-02-29", "2024-02-25", "2023-12-31", "2023-12-31"]],
    ["2024-04-01", ["2024-03-31", "2024-03-31", "2024-03-31", "2023-12-31"]],
  ] as const) assert.deepEqual(Object.values(relativeRankBaselines(date)), expected);
});
test("chips show directions, unchanged and unavailable with accessible explanations", () => {
  const ranks = readRelativeRankChanges(tables(), date)!.wtd;
  assert.match(rankChip("Apple", "wtd", ranks.Apple), /positive.*Moved from #2 to #1.*>↑ 1<\/button>/);
  assert.match(rankChip("Bitcoin", "wtd", ranks.Bitcoin), /negative.*>↓ 1<\/button>/);
  assert.match(rankChip("NVIDIA", "wtd", ranks.NVIDIA), /flat.*Unchanged at #2.*>—<\/button>/);
  assert.match(rankChip("SpaceX", "wtd", ranks.SpaceX), /unavailable.*baseline.*>N\/A<\/button>/);
  assert.match(rankChip("<asset>", "wtd"), /aria-label="&lt;asset&gt;: Rank history is unavailable/);
  assert.match(rankChip("Apple", "wtd", ranks.Apple), /hidden>↑ 1/);
  assert.doesNotMatch(rankChip("Apple", "ytd", ranks.Apple), /hidden/);
});
test("older manifests omit rank data and keep the existing inputs readable", () => {
  const release = { files: {} } as ReleaseManifest;
  assert.deepEqual(releaseInputFiles(release), inputFiles);
  assert.equal(readRelativeRankChanges({}, date), null);
  release.files[RELATIVE_RANK_FILE] = { sha256: "hash", size_bytes: 20 };
  assert.ok(releaseInputFiles(release).includes(RELATIVE_RANK_FILE));
});
