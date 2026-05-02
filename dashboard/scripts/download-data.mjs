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

const CSV_FILES = [
  "summary_table.csv",
  "summary_history.csv",
  "fundamentals_table.csv",
  "performance_table.csv",
  "monthly_heatmap_data.csv",
  "relative_value_comparison.csv",
  "ohlc_data.csv",
  "1k_bucket_table.csv",
  "5k_bucket_table.csv",
  "roi_table.csv",
  "onchain_price_models.csv",
  "mtd_returns_history.csv",
  "ytd_returns_history.csv",
  "price_outlook.csv",
];

mkdirSync(OUT_DIR, { recursive: true });

function httpsGet(url) {
  return new Promise((resolve, reject) => {
    https
      .get(url, (res) => {
        if (
          res.statusCode >= 300 &&
          res.statusCode < 400 &&
          res.headers.location
        ) {
          resolve(httpsGet(res.headers.location));
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} for ${url}`));
          return;
        }
        resolve(res);
      })
      .on("error", reject);
  });
}

async function downloadRemote(file) {
  const url = `${REMOTE_BASE_URL}/${file}`;
  const dst = path.join(OUT_DIR, file);
  const res = await httpsGet(url);
  await pipeline(res, createWriteStream(dst));
  console.log(`  ↓ ${file}`);
}

function copyLocal(file) {
  const src = path.join(LOCAL_CSV_DIR, file);
  const dst = path.join(OUT_DIR, file);
  if (!existsSync(src)) {
    console.warn(`  ⚠ skipped ${file} (not found in ../csv/)`);
    return;
  }
  copyFileSync(src, dst);
  console.log(`  ✓ ${file}`);
}

if (LOCAL_MODE) {
  console.log(`\nSyncing from local Report Library: ${LOCAL_CSV_DIR}\n`);
  for (const file of CSV_FILES) copyLocal(file);
} else {
  console.log(`\nDownloading from GitHub Pages: ${REMOTE_BASE_URL}\n`);
  for (const file of CSV_FILES) {
    try {
      await downloadRemote(file);
    } catch (err) {
      console.warn(`  ⚠ skipped ${file}: ${err.message}`);
    }
  }
}

console.log("\nDone. Next: npm run sources && npm run dev\n");
