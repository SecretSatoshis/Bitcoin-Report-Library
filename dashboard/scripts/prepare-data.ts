import { createHash } from "node:crypto";
import {
  readFileSync,
  mkdirSync,
  writeFileSync,
  renameSync,
  rmSync,
} from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import {
  createDashboard,
  inputFiles,
  parseCSV,
  type ReleaseManifest,
  type Tables,
} from "../src/data";
import { renderDashboard } from "../src/render";

export const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
export const hash = (bytes: string | Buffer) =>
  createHash("sha256").update(bytes).digest("hex");
export function loadVerified(directory: string): {
  tables: Tables;
  release: ReleaseManifest;
} {
  const release = JSON.parse(
    readFileSync(resolve(directory, "release_manifest.json"), "utf8"),
  ) as ReleaseManifest;
  if (release.schema_version !== 1 || !release.generated_at || !release.files)
    throw new Error("Invalid release manifest");
  const tables: Tables = {};
  for (const name of inputFiles) {
    const decoded =
      name === "bitcoin_candles.csv.gz" ? "bitcoin_candles.csv" : name;
    // Candles are read from the verified gzip archive itself.
    const bytes = readFileSync(resolve(directory, name));
    if (
      !release.files[name] ||
      hash(bytes) !== release.files[name].sha256 ||
      bytes.length !== release.files[name].size_bytes
    )
      throw new Error(`${name}: manifest hash/size mismatch`);
    tables[decoded] = parseCSV(
      (name.endsWith(".gz") ? gunzipSync(bytes) : bytes).toString("utf8"),
      name,
    );
  }
  return { tables, release };
}
export function prepareData(
  directory = resolve(root, "sources/bitcoin_report_library"),
) {
  const { tables, release } = loadVerified(directory);
  const colors = JSON.parse(
    readFileSync(resolve(root, "components/chart-colors.json"), "utf8"),
  );
  const events = JSON.parse(
    readFileSync(resolve(root, "components/chart-events.json"), "utf8"),
  );
  const data = createDashboard(tables, release, colors, events);
  const sourceManifest = JSON.parse(
    readFileSync(
      resolve(root, "static/shared-chart/source-manifest.json"),
      "utf8",
    ),
  );
  for (const [asset, expected] of Object.entries(
    sourceManifest.vendored_hashes,
  ) as [string, string][])
    if (
      hash(readFileSync(resolve(root, "static/shared-chart", asset))) !==
      expected
    )
      throw new Error(`Shared renderer hash mismatch: ${asset}`);
  const rendererHashes: Record<string, string> = {};
  for (const asset of Object.values(sourceManifest.assets) as string[])
    rendererHashes[asset] = hash(
      readFileSync(resolve(root, "static/shared-chart", asset)),
    );
  for (const name of [
    "frame.html",
    "seasonal-mtd.html",
    "seasonal-ytd.html",
    "embed-host.js",
    "source-manifest.json",
  ])
    rendererHashes[name] = hash(
      readFileSync(resolve(root, "static/shared-chart", name)),
    );
  // Chart payloads are published as separate files so the page itself stays small.
  const { charts, ...display } = data;
  const chartFiles = charts.map((chart) => ({
    path: `charts/${chart.id}.json`,
    text: JSON.stringify(chart),
  }));
  const manifest = {
    schema_version: 1,
    report_date: data.reportDate,
    release_generated_at: release.generated_at,
    inputs: Object.fromEntries(
      inputFiles.map((name) => [name, release.files[name]]),
    ),
    charts: Object.fromEntries(chartFiles.map((f) => [f.path, hash(f.text)])),
    renderer: rendererHashes,
  };
  const stage = resolve(root, ".prepare-staging");
  rmSync(stage, { recursive: true, force: true });
  mkdirSync(stage, { recursive: true });
  try {
    mkdirSync(resolve(stage, "data/charts"), { recursive: true });
    writeFileSync(resolve(stage, "index.html"), renderDashboard(data));
    for (const file of chartFiles)
      writeFileSync(resolve(stage, "data", file.path), file.text);
    writeFileSync(
      resolve(stage, "data/dashboard.json"),
      JSON.stringify({
        ...display,
        charts: charts.map((c) => ({ id: c.id, title: c.title, file: `charts/${c.id}.json` })),
      }),
    );
    writeFileSync(
      resolve(stage, "data/manifest.json"),
      JSON.stringify(manifest, null, 2) + "\n",
    );
    const current = resolve(root, ".generated"),
      previous = resolve(root, ".prepare-previous");
    rmSync(previous, { recursive: true, force: true });
    try {
      renameSync(current, previous);
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
    }
    try {
      renameSync(stage, current);
    } catch (e) {
      try {
        renameSync(previous, current);
      } catch {}
      throw e;
    }
    rmSync(previous, { recursive: true, force: true });
  } catch (e) {
    rmSync(stage, { recursive: true, force: true });
    throw e;
  }
  return { data, manifest };
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const { data } = prepareData();
  console.log(`Prepared verified dashboard for ${data.reportDate}`);
}
