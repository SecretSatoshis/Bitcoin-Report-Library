import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep } from "node:path";
import { chromium } from "playwright-core";

const root = resolve("build"),
  artifacts = resolve("../outputs/dashboard-browser");
mkdirSync(artifacts, { recursive: true });
const data = JSON.parse(readFileSync(resolve(root, "data/dashboard.json")));
const reference = JSON.parse(
  readFileSync("tests/fixtures/reference-2026-09-29.json"),
);
const server = createServer((req, res) => {
  try {
    const url = new URL(req.url, "http://localhost"),
      path = resolve(
        root,
        "." +
          decodeURIComponent(url.pathname) +
          (url.pathname.endsWith("/") ? "index.html" : ""),
      );
    if (!path.startsWith(root + sep)) throw Error("outside root");
    res.setHeader(
      "Content-Type",
      {
        ".html": "text/html",
        ".js": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".woff2": "font/woff2",
        ".ico": "image/x-icon",
      }[extname(path)] || "application/octet-stream",
    );
    res.end(readFileSync(path));
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: true });
const normalized = (s) =>
  s
    .replace(/[▲▼–—]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^-$|^Unavailable$/, "");
try {
  const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
      colorScheme: "dark",
    }),
    page = await context.newPage();
  const errors = [],
    external = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => {
    if (r.url().startsWith("http") && !r.url().startsWith(base))
      external.push(r.url());
  });
  await page.clock.setFixedTime(new Date("2032-01-01T00:00:00Z"));
  await page.goto(base);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(
    () => document.querySelectorAll('[data-chart-ready="true"]').length === 3,
  );
  assert.equal(
    await page
      .locator("[data-dashboard-date]")
      .getAttribute("data-dashboard-date"),
    data.reportDate,
  );
  assert.equal(
    await page.evaluate(
      () =>
        document.fonts.check("700 32px Syne") &&
        document.fonts.check('400 12px "JetBrains Mono"'),
    ),
    true,
  );
  if (data.reportDate === reference.date) {
    const tables = await page
      .locator("table:not(.correlation-matrix)")
      .evaluateAll((tables) =>
        tables.map((t) =>
          [...t.querySelectorAll("tbody tr")].map((r) =>
            [...r.querySelectorAll("th,td")].map((c) => c.innerText.trim()),
          ),
        ),
      );
    const originals = reference.tables.map((t) =>
      t.map((r) => r.map(normalized)),
    );
    const actual = tables.map((t) => t.map((r) => r.map(normalized)));
    // Fundamentals show each category once, as a row group, so compare the metric cells.
    const compare = [
      ...actual.slice(0, 7),
      actual[7].map((r) => r.slice(r.length - 6)),
      actual[8],
    ];
    const expected = [
      ...originals.slice(0, 7),
      originals[7].map((r) => r.slice(1)),
      originals[8],
    ];
    assert.deepEqual(
      compare,
      expected,
      "Every displayed table cell, null and row order agrees with the frozen release",
    );
    for (const text of await page
      .locator(
        ".metric-value,.sentiment-value,.metric-comparison>span:first-child",
      )
      .allTextContents())
      assert.ok(
        reference.snapshot.includes(text.trim()),
        "Snapshot value " + text,
      );
  }
  const frames = [];
  for (const element of await page
    .locator("[data-chart-frame]")
    .elementHandles()) {
    const frame = await element.contentFrame();
    frames.push(frame);
    await frame.evaluate(() => window.SecretSatoshisChart.ready);
    const payload = await frame.evaluate(() =>
      JSON.parse(document.querySelector("#chart-data").textContent),
    );
    assert.deepEqual(
      payload,
      JSON.parse(readFileSync(resolve(root, "data/charts", `${payload.id}.json`))),
    );
    assert.equal(
      await frame.evaluate(() =>
        SecretSatoshisChart.view.entries.every((e) => {
          const a = SecretSatoshisChart,
            s = e.definition,
            expected = s.values.flatMap((value, i) =>
              Number.isFinite(value)
                ? [{ time: a.payload.x[s.start + i], value }]
                : [],
            ),
            actual = e.parts.flatMap((p) => p.api.data());
          return (
            JSON.stringify(
              actual.map((p) => ({ time: p.time, value: p.value ?? p.close })),
            ) === JSON.stringify(expected)
          );
        }),
      ),
      true,
      "Exact plotted observations",
    );
    await frame.locator(".solo").first().click();
    assert.equal(
      await frame.evaluate(() => SecretSatoshisChart.state.visible.size),
      1,
    );
    await frame.locator("#show-all").click();
    assert.equal(
      await frame.evaluate(() => SecretSatoshisChart.state.visible.size),
      payload.series.length,
    );
    await frame.locator("#scale-right").selectOption("log");
    assert.equal(
      await frame.evaluate(() => SecretSatoshisChart.state.modes.right),
      "log",
    );
    await frame.locator("#reset").click();
    assert.equal(
      await frame.evaluate(() => SecretSatoshisChart.state.modes.right),
      "linear",
    );
    const png = await frame.evaluate(() =>
        SecretSatoshisChart.exportImage(false),
      ),
      bytes = Buffer.from(png.split(",")[1], "base64");
    assert.equal(bytes.readUInt32BE(16), 2400);
    assert.equal(bytes.readUInt32BE(20), 1350);
    writeFileSync(resolve(artifacts, payload.id + "-export.png"), bytes);
  }
  const price = frames[0];
  assert.equal(await price.locator("#bitcoin-style").inputValue(), "candles");
  assert.equal(await price.locator("#candle-interval").inputValue(), "weekly");
  for (const interval of ["daily", "monthly", "weekly"]) {
    await price.locator("#candle-interval").selectOption(interval);
    assert.equal(
      await price.evaluate(() => SecretSatoshisChart.state.interval),
      interval,
    );
  }
  for (const range of ["YTD", "1Y", "10Y", "ALL", "4Y"]) {
    await price.locator(`[data-range="${range}"]`).click();
    assert.equal(
      await price
        .locator(`[data-range="${range}"]`)
        .getAttribute("aria-pressed"),
      "true",
    );
  }
  await price.locator("#events").click();
  assert.equal(
    await price.evaluate(() => SecretSatoshisChart.state.events),
    false,
  );
  await price.locator("#reset").click();
  await price.locator("#bitcoin-style").selectOption("line");
  await page.locator("[data-price-outlook-frame]").scrollIntoViewIfNeeded();
  const frameBox = await page
    .locator("[data-price-outlook-frame]")
    .boundingBox();
  const point = await price.evaluate(() => {
    const a = SecretSatoshisChart,
      r = document.querySelector(".chart-pane").getBoundingClientRect(),
      time = a.payload.x[a.payload.x.indexOf(a.payload.reportDate) - 14];
    return {
      x: r.x + a.view.charts[0].timeScale().timeToCoordinate(time),
      y: r.y + 120,
    };
  });
  await page.mouse.move(frameBox.x + point.x, frameBox.y + point.y);
  await page.waitForTimeout(100);
  assert.match(await price.locator("#reading-mode").textContent(), /CURSOR/);
  await price.locator("#reset").click();
  for (const frame of frames.slice(1)) {
    assert.equal(await frame.locator("#ranges button").count(), 1);
    assert.equal(
      await frame.locator("#reading-mode").textContent(),
      "REPORT DAY",
    );
    assert.equal(
      await frame.evaluate(() => SecretSatoshisChart.view.range().from),
      0,
    );
  }
  const button = page.locator(".help-button").first();
  if (data.correlations) {
    for (const period of [30, 365, 90]) {
      const tab = page.getByRole("tab", { name: `${period} days`, exact: true });
      await tab.click();
      assert.equal(await tab.getAttribute("aria-selected"), "true");
      assert.equal(await page.locator(`#correlation-panel-${period}`).isVisible(), true);
      assert.equal(await page.locator("#correlation-matrix-title").textContent(), `${period}-Day Correlation Matrix`);
      const actual = await page.locator(`#correlation-panel-${period} tbody tr`).evaluateAll((rows) =>
        rows.map((r) => [...r.querySelectorAll("td")].map((c) => c.textContent.trim())),
      );
      const tickers = data.correlations[period].map((r) => r.Ticker);
      assert.deepEqual(actual, data.correlations[period].map((r, i) => tickers.map((ticker, j) =>
        j > i ? "" : r[ticker].trim() ? Number(r[ticker]).toFixed(2) : "—",
      )), `${period}-day values must match the release`);
    }
    await page.getByRole("tab", { name: "90 days", exact: true }).press("End");
    assert.equal(await page.getByRole("tab", { name: "365 days", exact: true }).getAttribute("aria-selected"), "true");
    await page.getByRole("tab", { name: "365 days", exact: true }).press("Home");
    await page.getByRole("tab", { name: "30 days", exact: true }).press("ArrowRight");
    assert.equal(await page.getByRole("tab", { name: "90 days", exact: true }).getAttribute("aria-selected"), "true");
  }
  await button.focus();
  await button.press("Enter");
  assert.equal(await button.getAttribute("aria-expanded"), "true");
  assert.equal(await page.locator(".help-text").first().isVisible(), true);
  await button.press("Escape");
  assert.equal(await button.getAttribute("aria-expanded"), "false");
  assert.equal(await page.locator(".help-text").first().isVisible(), false);
  for (const width of [1440, 768, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.waitForTimeout(200);
    assert.equal(
      await page.evaluate(() => document.documentElement.scrollWidth),
      width,
      "Page overflow at " + width,
    );
    for (const frame of frames)
      assert.equal(
        await frame.evaluate(() => document.documentElement.scrollWidth),
        await frame.evaluate(() => innerWidth),
        "Frame overflow",
      );
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({
      path: resolve(artifacts, `dashboard-${width}.png`),
    });
    if (width === 390) {
      await page.locator("#nav-toggle").click();
      assert.equal(
        await page.locator("#nav-toggle").getAttribute("aria-expanded"),
        "true",
      );
      await page.keyboard.press("Escape");
      assert.equal(
        await page.locator("#nav-toggle").getAttribute("aria-expanded"),
        "false",
      );
      assert.equal(
        await page.evaluate(() =>
          [...document.querySelectorAll(".table-scroll")].some(
            (el) => el.scrollWidth > el.clientWidth,
          ),
        ),
        true,
      );
    }
  }
  const disabled = await browser.newContext({
      javaScriptEnabled: false,
      viewport: { width: 390, height: 844 },
    }),
    staticPage = await disabled.newPage();
  await staticPage.goto(base);
  assert.equal(await staticPage.locator(".metric-card").count(), 9);
  assert.equal(await staticPage.locator("table").count(), 9 + (data.correlations ? 3 : 0));
  if (data.correlations) {
    const actual = await staticPage.locator(".correlation-matrix tbody tr").evaluateAll((rows) =>
      rows.map((row) => [...row.querySelectorAll("td")].map((cell) => cell.textContent.trim())),
    );
    const tickers = data.correlations[90].map((row) => row.Ticker);
    const expected = [30, 90, 365].flatMap((period) => data.correlations[period].map((row, i) => tickers.map((ticker, j) =>
      j > i ? "" : row[ticker].trim() ? Number(row[ticker]).toFixed(2) : "—",
    )));
    assert.deepEqual(actual, expected, "Every matrix cell renders from the verified release without JavaScript");
  }
  assert.equal(
    await staticPage.evaluate(() => document.documentElement.scrollWidth),
    390,
  );
  await disabled.close();
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  console.log(
    "PASS: numerical display parity, three independent charts, controls, hover, PNG exports, keyboard, offline fonts, responsive layouts and JavaScript-free data.",
  );
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}
