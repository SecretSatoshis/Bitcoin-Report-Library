/**
 * Data sync script for Bitcoin Report Dashboard (Evidence.dev)
 *
 * Modes:
 *   --local   Copy CSVs directly from ../csv/ (Report Library local output).
 *   (default) Download CSVs from GitHub Pages.
 *
 * Currently scoped to the CSVs the dashboard actually uses.
 * Wide files (master_metrics_data, cagr_data) are intentionally excluded —
 * they cause Evidence's CSV plugin to hang on type inference.
 */

import {
  createWriteStream,
  mkdirSync,
  existsSync,
  copyFileSync,
  renameSync,
  rmSync,
  statSync,
  readdirSync,
} from "node:fs";
import { pipeline } from "node:stream/promises";
import https from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const LOCAL_MODE = process.argv.includes("--local");

const REMOTE_BASE_URL =
  "https://secretsatoshis.github.io/Bitcoin-Report-Library/csv";
const LOCAL_CSV_DIR = path.resolve(__dirname, "../../csv");
const OUT_DIR = path.resolve(__dirname, "../sources/bitcoin_report_library");

// Only the CSVs the dashboard actually queries. ohlc_data.csv and
// report_ohlc_summary.csv were fetched and ingested on every build but are referenced
// nowhere on the page — they belong to the weekly recap workflow, not here.
const CSV_FILES = [
  "summary_table.csv",
  "summary_history.csv",
  "fundamentals_table.csv",
  "performance_table.csv",
  "monthly_heatmap_data.csv",
  "relative_value_comparison.csv",
  "1k_bucket_table.csv",
  "5k_bucket_table.csv",
  "roi_table.csv",
  "onchain_price_models.csv",
  "mtd_returns_history.csv",
  "ytd_returns_history.csv",
  "price_outlook.csv",
];

mkdirSync(OUT_DIR, { recursive: true });

// Drop CSVs left behind by an earlier sync. Removing a file from CSV_FILES only stops
// it being copied — without this it lingers in sources/ and Evidence keeps ingesting it
// into the build forever.
function prune() {
  const keep = new Set(CSV_FILES);
  for (const entry of readdirSync(OUT_DIR)) {
    if (entry.endsWith(".csv") && !keep.has(entry)) {
      rmSync(path.join(OUT_DIR, entry), { force: true });
      console.log(`  ✗ pruned stale ${entry}`);
    }
  }
}

const REQUEST_TIMEOUT_MS = 20000;
const MAX_ATTEMPTS = 3;
const MAX_REDIRECTS = 5;

function httpsGet(url, redirectsLeft = MAX_REDIRECTS) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { timeout: REQUEST_TIMEOUT_MS }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        // Drain the redirect body so the socket can be reused, and cap the depth —
        // an unbounded chain would otherwise recurse until the process dies.
        res.resume();
        if (redirectsLeft <= 0) {
          reject(new Error(`Too many redirects for ${url}`));
          return;
        }
        resolve(httpsGet(new URL(res.headers.location, url).href, redirectsLeft - 1));
        return;
      }
      if (res.statusCode !== 200) {
        res.resume();
        reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        return;
      }
      resolve(res);
    });

    // Without an explicit timeout a hung socket never errors, so the build blocks
    // until the platform kills it rather than failing in seconds.
    req.on("timeout", () => {
      req.destroy(new Error(`Timed out after ${REQUEST_TIMEOUT_MS}ms for ${url}`));
    });
    req.on("error", reject);
  });
}

async function downloadRemote(file) {
  const url = `${REMOTE_BASE_URL}/${file}`;
  const dst = path.join(OUT_DIR, file);
  const tmp = `${dst}.part`;

  let lastError;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const res = await httpsGet(url);
      // Stream to a scratch file and only publish it on success. Writing straight to
      // the destination leaves a truncated CSV behind when a transfer dies mid-stream,
      // and a short file parses cleanly — the dashboard would build and deploy with
      // silently missing rows.
      await pipeline(res, createWriteStream(tmp));

      const bytes = statSync(tmp).size;
      if (bytes === 0) throw new Error("empty response body");

      renameSync(tmp, dst);
      console.log(`  ↓ ${file} (${bytes.toLocaleString()} bytes)`);
      return;
    } catch (err) {
      lastError = err;
      if (existsSync(tmp)) rmSync(tmp, { force: true });
      if (attempt < MAX_ATTEMPTS) {
        const backoff = 500 * 2 ** (attempt - 1);
        console.warn(`  ⟳ ${file} attempt ${attempt} failed (${err.message}) — retrying in ${backoff}ms`);
        await new Promise((r) => setTimeout(r, backoff));
      }
    }
  }
  throw new Error(`${file}: ${lastError.message}`);
}

function copyLocal(file) {
  const src = path.join(LOCAL_CSV_DIR, file);
  const dst = path.join(OUT_DIR, file);
  const tmp = `${dst}.part`;

  try {
    // Match remote publication semantics: fully copy and validate a scratch file,
    // then atomically replace the destination. A failed copy therefore cannot
    // truncate a previously good dashboard source.
    copyFileSync(src, tmp);
    const bytes = statSync(tmp).size;
    if (bytes === 0) throw new Error("empty source file");

    renameSync(tmp, dst);
    console.log(`  ✓ ${file} (${bytes.toLocaleString()} bytes)`);
  } catch (err) {
    if (existsSync(tmp)) rmSync(tmp, { force: true });
    throw new Error(`${file}: ${err.message}`);
  }
}

function validateLocalInputs() {
  const failures = [];

  for (const file of CSV_FILES) {
    const src = path.join(LOCAL_CSV_DIR, file);
    try {
      const stats = statSync(src);
      if (!stats.isFile()) {
        failures.push(`${file}: source is not a regular file`);
      } else if (stats.size === 0) {
        failures.push(`${file}: source file is empty`);
      }
    } catch (err) {
      failures.push(
        err.code === "ENOENT"
          ? `${file}: source file is missing`
          : `${file}: ${err.message}`
      );
    }
  }

  return failures;
}

if (LOCAL_MODE) {
  console.log(`\nSyncing from local Report Library: ${LOCAL_CSV_DIR}\n`);

  // Validate the complete required set before pruning or publishing anything. This
  // keeps the existing dashboard sources intact when a report run is incomplete.
  const failures = validateLocalInputs();
  if (failures.length) {
    console.error(
      `${failures.length} of ${CSV_FILES.length} local source files are invalid:\n` +
        failures.map((f) => `  - ${f}`).join("\n") +
        "\nRefusing to sync incomplete data; existing dashboard sources were left unchanged.\n"
    );
    process.exit(1);
  }

  prune();
  for (const file of CSV_FILES) copyLocal(file);
} else {
  prune();
  console.log(`\nDownloading from GitHub Pages: ${REMOTE_BASE_URL}\n`);
  const failures = [];
  for (const file of CSV_FILES) {
    try {
      await downloadRemote(file);
    } catch (err) {
      failures.push(err.message);
      console.error(`  ✗ ${err.message}`);
    }
  }
  // Fail the build rather than deploying a dashboard with missing datasets. A skipped
  // CSV used to leave the previous build's file in place (or none at all), so the page
  // shipped with stale or empty sections and a green checkmark.
  if (failures.length) {
    console.error(
      `\n${failures.length} of ${CSV_FILES.length} files could not be downloaded:\n` +
        failures.map((f) => `  - ${f}`).join("\n") +
        "\nRefusing to build with incomplete data.\n"
    );
    process.exit(1);
  }
}

console.log("\nDone. Next: npm run sources && npm run dev\n");
