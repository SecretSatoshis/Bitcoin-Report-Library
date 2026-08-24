#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync, linkSync, unlinkSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, join, normalize, relative, resolve, sep } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const DASHBOARD_ROOT = resolve(SCRIPT_DIR, '..');
const REPOSITORY_ROOT = resolve(DASHBOARD_ROOT, '..');
const DEFAULT_BUILD_DIR = join(DASHBOARD_ROOT, 'build');
const MANIFEST_NAME = 'visual-manifest.json';
const VIEWPORT = { width: 1440, height: 1000 };
const DEVICE_SCALE_FACTOR = 2;

const VISUALS = [
  {
    id: 'bitcoin-snapshot-market-data',
    file: 'bitcoin-snapshot.png',
    sourceUrl: 'https://dashboard.secretsatoshis.com/#bitcoin-snapshot',
    alt: 'Bitcoin Snapshot market-data cards showing price, market capitalization, and sats per dollar.',
    requiredText: ['Market Data', 'Bitcoin Price', 'Bitcoin Market Cap', 'Sats Per Dollar'],
    forbiddenText: ['Bitcoin Supply'],
  },
  {
    id: 'bitcoin-price',
    file: 'bitcoin-price.png',
    sourceUrl: 'https://dashboard.secretsatoshis.com/#bitcoin-price',
    alt: 'Bitcoin price with realized-price models and Secret Satoshis bear, base, and bull cases.',
    requiredText: ['Bitcoin Price', 'Bear Case', 'Base Case', 'Bull Case', 'STH Realized', 'LTH Realized', 'Realized'],
    forbiddenText: ['Power Expense', 'Electricity'],
  },
  {
    id: 'monthly-return-heatmap',
    file: 'monthly-return-heatmap.png',
    sourceUrl: 'https://dashboard.secretsatoshis.com/#monthly-bitcoin-price-return-heatmap',
    alt: 'Monthly Bitcoin price-return heatmap with statistical reference rows and historical yearly returns.',
    requiredText: ['Monthly Bitcoin Price Return Heatmap', 'Statistical Reference', 'Historical Returns by Year'],
    forbiddenText: [],
  },
  {
    id: 'seasonal-mtd',
    file: 'seasonal-mtd.png',
    sourceUrl: 'https://dashboard.secretsatoshis.com/#seasonal-returns',
    alt: 'Current Bitcoin month-to-date path compared with historical years and the historical average.',
    requiredText: ['MTD Returns Comparison'],
    forbiddenText: [],
    requireCanvas: true,
  },
  {
    id: 'seasonal-ytd',
    file: 'seasonal-ytd.png',
    sourceUrl: 'https://dashboard.secretsatoshis.com/#seasonal-returns',
    alt: 'Current Bitcoin year-to-date path compared with historical years and the historical average.',
    requiredText: ['YTD Returns Comparison'],
    forbiddenText: [],
    requireCanvas: true,
  },
];

function usage(message) {
  if (message) process.stderr.write(`ERROR: ${message}\n`);
  process.stderr.write(
    'Usage: node scripts/export-newsletter-visuals.mjs --report-date YYYY-MM-DD --output-dir PATH [--build-dir PATH] [--chrome PATH]\n'
  );
  process.exit(2);
}

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) usage(`unexpected argument: ${token}`);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) usage(`${token} requires a value`);
    args[token.slice(2)] = value;
    index += 1;
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(args['report-date'] || '')) {
    usage('--report-date must be YYYY-MM-DD');
  }
  if (!args['output-dir']) usage('--output-dir is required');
  return {
    reportDate: args['report-date'],
    outputDir: resolve(args['output-dir']),
    buildDir: resolve(args['build-dir'] || DEFAULT_BUILD_DIR),
    chrome: args.chrome || process.env.CHROME_EXECUTABLE
      ? resolve(args.chrome || process.env.CHROME_EXECUTABLE)
      : null,
  };
}

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function git(command) {
  const result = spawnSync('git', ['-C', REPOSITORY_ROOT, ...command], {
    encoding: 'utf8',
    timeout: 10_000,
  });
  return result.status === 0 ? result.stdout.trim() : null;
}

function pngDimensions(path) {
  const bytes = readFileSync(path);
  const signature = '89504e470d0a1a0a';
  if (bytes.length < 24 || bytes.subarray(0, 8).toString('hex') !== signature) {
    throw new Error(`${path} is not a valid PNG`);
  }
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

function mimeType(path) {
  return {
    '.css': 'text/css; charset=utf-8',
    '.html': 'text/html; charset=utf-8',
    '.ico': 'image/x-icon',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.wasm': 'application/wasm',
    '.webmanifest': 'application/manifest+json',
  }[extname(path).toLowerCase()] || 'application/octet-stream';
}

function staticServer(buildDir) {
  return createServer((request, response) => {
    try {
      const requestPath = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      const requested = requestPath === '/' ? 'index.html' : requestPath.replace(/^\/+/, '');
      const path = normalize(join(buildDir, requested));
      const relativePath = relative(buildDir, path);
      if (relativePath === '..' || relativePath.startsWith(`..${sep}`) || !existsSync(path)) {
        response.writeHead(404).end('Not found');
        return;
      }
      response.writeHead(200, {
        'Cache-Control': 'no-store',
        'Content-Type': mimeType(path),
      });
      createReadStream(path).pipe(response);
    } catch (error) {
      response.writeHead(500).end(String(error));
    }
  });
}

async function listen(server) {
  await new Promise((resolvePromise, rejectPromise) => {
    server.once('error', rejectPromise);
    server.listen(0, '127.0.0.1', resolvePromise);
  });
  return server.address().port;
}

async function close(server) {
  await new Promise((resolvePromise) => server.close(resolvePromise));
}

function publishFiles(temporaryDir, outputDir, filenames) {
  const published = [];
  try {
    for (const filename of filenames) {
      const destination = join(outputDir, filename);
      if (existsSync(destination)) throw new Error(`refusing to overwrite ${destination}`);
      linkSync(join(temporaryDir, filename), destination);
      published.push(destination);
    }
  } catch (error) {
    for (const path of published.reverse()) unlinkSync(path);
    throw error;
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!existsSync(join(args.buildDir, 'index.html'))) {
    throw new Error(`dashboard build is missing: ${args.buildDir}`);
  }
  if (args.chrome && !existsSync(args.chrome)) {
    throw new Error(`Chrome executable is missing: ${args.chrome}`);
  }
  mkdirSync(args.outputDir, { recursive: true });
  for (const visual of VISUALS) {
    if (existsSync(join(args.outputDir, visual.file))) {
      throw new Error(`refusing to overwrite ${join(args.outputDir, visual.file)}`);
    }
  }
  if (existsSync(join(args.outputDir, MANIFEST_NAME))) {
    throw new Error(`refusing to overwrite ${join(args.outputDir, MANIFEST_NAME)}`);
  }

  const temporaryDir = mkdtempSync(join(args.outputDir, '.newsletter-export-'));
  const server = staticServer(args.buildDir);
  let browser;
  try {
    const port = await listen(server);
    browser = await chromium.launch({
      ...(args.chrome ? { executablePath: args.chrome } : {}),
      headless: true,
      args: ['--force-color-profile=srgb', '--font-render-hinting=none'],
    });
    const context = await browser.newContext({
      viewport: VIEWPORT,
      deviceScaleFactor: DEVICE_SCALE_FACTOR,
      colorScheme: 'dark',
      reducedMotion: 'reduce',
    });
    const page = await context.newPage();
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(String(error)));
    await page.goto(`http://127.0.0.1:${port}/`, {
      waitUntil: 'domcontentloaded',
      timeout: 60_000,
    });
    await page.waitForFunction(
      () => {
        const hero = document.querySelector('[data-dashboard-date]');
        return hero?.getAttribute('data-dashboard-date') &&
          document.querySelectorAll('[data-newsletter-visual]').length === 5;
      },
      { timeout: 120_000 },
    );
    await page.evaluate(async () => {
      await document.fonts.ready;
      document.body.classList.add('newsletter-export-mode');
    });
    await page.addStyleTag({
      content: '*,*::before,*::after{animation-duration:0s!important;animation-delay:0s!important;transition:none!important;caret-color:transparent!important}',
    });
    await page.waitForTimeout(1500);

    const dashboardDate = await page.locator('[data-dashboard-date]').getAttribute('data-dashboard-date');
    if (dashboardDate !== args.reportDate) {
      throw new Error(`dashboard latest-data date ${dashboardDate || 'missing'} does not match ${args.reportDate}`);
    }
    if (pageErrors.length) throw new Error(`dashboard page errors: ${pageErrors.join('; ')}`);

    const files = [];
    for (const visual of VISUALS) {
      const selector = `[data-newsletter-visual="${visual.id}"]`;
      const locator = page.locator(selector);
      const count = await locator.count();
      if (count !== 1) throw new Error(`${selector} expected exactly once, found ${count}`);
      const text = (await locator.innerText()).replace(/\s+/g, ' ').trim();
      for (const required of visual.requiredText) {
        if (!text.toLocaleLowerCase('en-US').includes(required.toLocaleLowerCase('en-US'))) {
          throw new Error(`${visual.id} is missing required text: ${required}`);
        }
      }
      for (const forbidden of visual.forbiddenText) {
        if (text.toLocaleLowerCase('en-US').includes(forbidden.toLocaleLowerCase('en-US'))) {
          throw new Error(`${visual.id} contains forbidden text: ${forbidden}`);
        }
      }
      if (visual.requireCanvas && await locator.locator('canvas').count() !== 1) {
        throw new Error(`${visual.id} must contain exactly one rendered chart canvas`);
      }
      const box = await locator.boundingBox();
      if (!box || box.width < 1000 || box.height < 160) {
        throw new Error(`${visual.id} has an implausible capture box: ${JSON.stringify(box)}`);
      }
      await locator.scrollIntoViewIfNeeded();
      const temporaryPath = join(temporaryDir, visual.file);
      await locator.screenshot({ path: temporaryPath, animations: 'disabled', type: 'png' });
      const dimensions = pngDimensions(temporaryPath);
      const sizeBytes = statSync(temporaryPath).size;
      if (dimensions.width < 2000 || dimensions.height < 300 || sizeBytes < 10_000) {
        throw new Error(`${visual.id} produced an implausible PNG: ${dimensions.width}x${dimensions.height}, ${sizeBytes} bytes`);
      }
      files.push({
        id: visual.id,
        filename: visual.file,
        selector,
        source_url: visual.sourceUrl,
        alt_text: visual.alt,
        sha256: sha256(temporaryPath),
        size_bytes: sizeBytes,
        width_px: dimensions.width,
        height_px: dimensions.height,
        logical_width_px: Math.round(box.width),
        logical_height_px: Math.round(box.height),
      });
    }

    const buildManifestPath = join(args.buildDir, 'data', 'manifest.json');
    const manifest = {
      schema_version: 1,
      workflow: 'Secret Satoshis Weekly Newsletter Dashboard Visual Export',
      report_date: args.reportDate,
      generated_at: new Date().toISOString(),
      status: 'passed',
      dashboard: {
        repository: 'SecretSatoshis/Bitcoin-Report-Library',
        local_root: REPOSITORY_ROOT,
        commit: git(['rev-parse', 'HEAD']),
        dirty: Boolean(git(['status', '--porcelain'])),
        build_index_sha256: sha256(join(args.buildDir, 'index.html')),
        build_data_manifest_sha256: existsSync(buildManifestPath) ? sha256(buildManifestPath) : null,
        latest_data_date: dashboardDate,
        production_url: 'https://dashboard.secretsatoshis.com/',
      },
      render: {
        browser: await browser.version(),
        viewport_css_px: VIEWPORT,
        device_scale_factor: DEVICE_SCALE_FACTOR,
        color_scheme: 'dark',
        color_profile: 'sRGB',
        format: 'PNG',
      },
      files,
    };
    writeFileSync(join(temporaryDir, MANIFEST_NAME), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
    publishFiles(
      temporaryDir,
      args.outputDir,
      [...files.map((item) => item.filename), MANIFEST_NAME],
    );
    process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
    process.stdout.write(`OK: newsletter visuals written to ${args.outputDir}\n`);
  } finally {
    if (browser) await browser.close();
    await close(server);
    rmSync(temporaryDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  process.stderr.write(`FAIL: ${error.message}\n`);
  process.exitCode = 1;
});
