/**
 * Sync the dashboard's CSVs from the Report Library.
 *
 *   --local   Copy from ../../csv (a local Report Library run).
 *   (default) Download from GitHub Pages.
 *
 * Each sync stages one complete release, checks every file against its manifest, then
 * swaps the input folder in one step. A failure leaves the current inputs untouched.
 * Only the files the dashboard reads (inputFiles in src/data.ts) are synced.
 */

import {
  createWriteStream,
  mkdirSync,
  existsSync,
  copyFileSync,
  renameSync,
  rmSync,
  statSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { pipeline } from "node:stream/promises";
import https from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { inputFiles } from "../src/data.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const LOCAL_MODE = process.argv.includes("--local");

const REMOTE_BASE_URL =
  "https://secretsatoshis.github.io/Bitcoin-Report-Library/csv";
const LOCAL_CSV_DIR = path.resolve(__dirname, "../../csv");
const OUT_DIR = path.resolve(__dirname, "../sources/bitcoin_report_library");
// Stage outside the published input directory so readers always see one release.
const STAGING_DIR = path.resolve(__dirname, "../.sync-staging");
const PREVIOUS_DIR = path.resolve(__dirname, "../.sync-previous");
const RELEASE_MANIFEST = "release_manifest.json";
// Pages deploys a few minutes after a release commit and caches files for 10 minutes, so
// remote sync waits until Pages serves at least this checkout's release.
const RELEASE_WAIT_MS = 12 * 60 * 1000;
const RELEASE_POLL_MS = 15 * 1000;

const CSV_FILES = inputFiles;

const REQUEST_TIMEOUT_MS = 20000;
const MAX_ATTEMPTS = 3;
const MAX_REDIRECTS = 5;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function httpsGet(url, redirectsLeft = MAX_REDIRECTS) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { timeout: REQUEST_TIMEOUT_MS }, (res) => {
      if (
        res.statusCode >= 300 &&
        res.statusCode < 400 &&
        res.headers.location
      ) {
        // Drain the redirect body so the socket is reused; the depth cap stops a loop.
        res.resume();
        if (redirectsLeft <= 0) {
          reject(new Error(`Too many redirects for ${url}`));
          return;
        }
        resolve(
          httpsGet(new URL(res.headers.location, url).href, redirectsLeft - 1),
        );
        return;
      }
      if (res.statusCode !== 200) {
        res.resume();
        reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        return;
      }
      resolve(res);
    });

    // A hung socket never errors on its own; fail fast instead of stalling the build.
    req.on("timeout", () => {
      req.destroy(
        new Error(`Timed out after ${REQUEST_TIMEOUT_MS}ms for ${url}`),
      );
    });
    req.on("error", reject);
  });
}

async function readRemoteJson(url) {
  const response = await httpsGet(url);
  const chunks = [];
  for await (const chunk of response) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

// A missing or incomplete manifest fails the sync.
function loadManifest(payload) {
  if (
    !payload ||
    payload.schema_version !== 1 ||
    typeof payload.release_id !== "string" ||
    payload.release_id !== payload.report_date ||
    !/^\d{4}-\d{2}-\d{2}$/.test(payload.report_date) ||
    !payload.files ||
    typeof payload.files !== "object"
  ) {
    throw new Error("invalid Report Library release manifest");
  }
  for (const file of CSV_FILES) {
    const record = payload.files[file];
    if (!record || typeof record.sha256 !== "string") {
      throw new Error(`release manifest is missing ${file}`);
    }
  }
  return payload;
}

function readLocalManifest() {
  const manifestPath = path.join(LOCAL_CSV_DIR, RELEASE_MANIFEST);
  try {
    return loadManifest(JSON.parse(readFileSync(manifestPath, "utf8")));
  } catch (err) {
    if (err.code === "ENOENT") {
      throw new Error(`${RELEASE_MANIFEST} is missing from ${LOCAL_CSV_DIR}`);
    }
    throw err;
  }
}

function verifyHash(file, filePath, manifest) {
  const digest = createHash("sha256")
    .update(readFileSync(filePath))
    .digest("hex");
  if (digest !== manifest.files[file].sha256) {
    throw new Error(`${file}: release manifest hash mismatch`);
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
          : `${file}: ${err.message}`,
      );
    }
  }

  return failures;
}

function stageLocal() {
  console.log(`\nSyncing from local Report Library: ${LOCAL_CSV_DIR}\n`);
  const failures = validateLocalInputs();
  if (failures.length) {
    throw new Error(
      `${failures.length} of ${CSV_FILES.length} local source files are invalid:\n` +
        failures.map((f) => `  - ${f}`).join("\n"),
    );
  }
  const manifest = readLocalManifest();
  for (const file of CSV_FILES) {
    const staged = path.join(STAGING_DIR, file);
    copyFileSync(path.join(LOCAL_CSV_DIR, file), staged);
    verifyHash(file, staged, manifest);
    console.log(
      `  ✓ ${file} (${statSync(staged).size.toLocaleString()} bytes)`,
    );
  }
  return manifest;
}

// The release committed in this checkout, if any.
function checkoutRelease() {
  try {
    const manifest = readLocalManifest();
    return {
      report_date: manifest.report_date,
      generated_at: manifest.generated_at ?? "",
    };
  } catch {
    return null;
  }
}

// A rerun can republish the same report date, so ties are broken by generation time.
// Both are ISO strings, which sort chronologically.
function isAtLeast(remote, minimum) {
  if (!minimum) return true;
  if (remote.report_date !== minimum.report_date) {
    return remote.report_date > minimum.report_date;
  }
  return (remote.generated_at ?? "") >= minimum.generated_at;
}

function describeRelease(release) {
  return release
    ? `${release.report_date} (generated ${release.generated_at || "unknown"})`
    : "none";
}

async function waitForRemoteRelease(minimum) {
  const deadline = Date.now() + RELEASE_WAIT_MS;
  for (;;) {
    let manifest = null;
    try {
      // A unique query bypasses the CDN's cached copy of the manifest.
      manifest = loadManifest(
        await readRemoteJson(
          `${REMOTE_BASE_URL}/${RELEASE_MANIFEST}?t=${Date.now()}`,
        ),
      );
    } catch (err) {
      if (Date.now() >= deadline) throw err;
      console.warn(`  ⟳ release manifest unavailable (${err.message})`);
    }
    if (manifest && isAtLeast(manifest, minimum)) {
      return manifest;
    }
    if (Date.now() >= deadline) {
      throw new Error(
        `GitHub Pages still serves release ${describeRelease(manifest)}, but this ` +
          `checkout contains ${describeRelease(minimum)}; refusing to build stale data`,
      );
    }
    if (manifest) {
      console.log(
        `  … Pages serves ${describeRelease(manifest)}; waiting for ${describeRelease(minimum)} to deploy`,
      );
    }
    await sleep(RELEASE_POLL_MS);
  }
}

async function downloadRemote(file, manifest) {
  // Key the request to the release so a CDN copy of an older file cannot be served.
  const url = `${REMOTE_BASE_URL}/${file}?release=${encodeURIComponent(manifest.release_id)}`;
  const dst = path.join(STAGING_DIR, file);
  const tmp = `${dst}.part`;

  let lastError;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const res = await httpsGet(url);
      await pipeline(res, createWriteStream(tmp));
      const bytes = statSync(tmp).size;
      if (bytes === 0) throw new Error("empty response body");
      verifyHash(file, tmp, manifest);
      renameSync(tmp, dst);
      console.log(`  ↓ ${file} (${bytes.toLocaleString()} bytes)`);
      return;
    } catch (err) {
      lastError = err;
      if (existsSync(tmp)) rmSync(tmp, { force: true });
      if (attempt < MAX_ATTEMPTS) {
        const backoff = 500 * 2 ** (attempt - 1);
        console.warn(
          `  ⟳ ${file} attempt ${attempt} failed (${err.message}) — retrying in ${backoff}ms`,
        );
        await sleep(backoff);
      }
    }
  }
  throw new Error(`${file}: ${lastError.message}`);
}

async function stageRemote() {
  console.log(`\nDownloading from GitHub Pages: ${REMOTE_BASE_URL}\n`);
  const manifest = await waitForRemoteRelease(checkoutRelease());
  console.log(`  ✓ release ${manifest.release_id}`);
  const failures = [];
  for (const file of CSV_FILES) {
    try {
      await downloadRemote(file, manifest);
    } catch (err) {
      failures.push(err.message);
      console.error(`  ✗ ${err.message}`);
    }
  }
  if (failures.length) {
    throw new Error(
      `${failures.length} of ${CSV_FILES.length} files could not be downloaded:\n` +
        failures.map((f) => `  - ${f}`).join("\n"),
    );
  }
  return manifest;
}

// Swap the whole input folder at once, so preparation never sees mixed releases and
// files dropped from CSV_FILES disappear.
function publishStaged(manifest) {
  writeFileSync(
    path.join(STAGING_DIR, RELEASE_MANIFEST),
    JSON.stringify(manifest, null, 2) + "\n",
  );
  rmSync(PREVIOUS_DIR, { recursive: true, force: true });
  if (existsSync(OUT_DIR)) renameSync(OUT_DIR, PREVIOUS_DIR);
  renameSync(STAGING_DIR, OUT_DIR);
  rmSync(PREVIOUS_DIR, { recursive: true, force: true });
}

rmSync(STAGING_DIR, { recursive: true, force: true });
mkdirSync(STAGING_DIR, { recursive: true });
try {
  const manifest = LOCAL_MODE ? stageLocal() : await stageRemote();
  publishStaged(manifest);
  console.log(
    `\nSynced release ${manifest.release_id}. Next: npm run dev\n`,
  );
} catch (err) {
  rmSync(STAGING_DIR, { recursive: true, force: true });
  console.error(
    `\n✗ ${err.message}\nRefusing to sync; existing dashboard sources were left unchanged.\n`,
  );
  process.exit(1);
}
