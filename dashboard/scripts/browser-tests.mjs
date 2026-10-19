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
  await page.addInitScript(() => {
    // Record live and export reference labels separately to verify both rendering paths.
    window.__referenceDraws = {};
    window.__exportReferenceLabels = [];
    const p = CanvasRenderingContext2D.prototype;
    const move = p.moveTo, line = p.lineTo, stroke = p.stroke, fill = p.fillText;
    p.moveTo = function(x, y) { this._referenceFrom = [x, y]; return move.call(this, x, y); };
    p.lineTo = function(x, y) { this._referenceTo = [x, y]; return line.call(this, x, y); };
    p.stroke = function(...args) {
      this._referenceStroke = { from: this._referenceFrom, to: this._referenceTo, width: this.lineWidth, dash: this.getLineDash() };
      return stroke.apply(this, args);
    };
    p.fillText = function(text, x, y, ...args) {
      if (this.canvas.closest("#chart") && / · \$/.test(text))
        window.__referenceDraws[text] = { x, y, align: this.textAlign, ...this._referenceStroke };
      else if (this.canvas.closest('[aria-hidden="true"]') && / · \$/.test(text))
        window.__exportReferenceLabels.push(text);
      return fill.call(this, text, x, y, ...args);
    };
  });
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
  const performancePeriods = ["7 Day", "30 Day", "90 Day", "365 Day", "2 Year", "4 Year", "WTD", "MTD", "YTD"];
  for (const group of data.performance) {
    const table = page.locator(`[data-quarterly-visual="${group.id}"] table`);
    assert.deepEqual(await table.locator("thead th").allTextContents(), ["Asset", "Price", ...performancePeriods]);
    const rows = await table.locator("tbody tr").all();
    for (const [i, row] of rows.entries()) {
      const cells = await row.locator("th,td").allTextContents();
      assert.equal(cells.length, 11);
      for (const [j, period] of performancePeriods.entries()) {
        const source = group.rows[i][period + " Return (%)"], text = cells[j + 2].trim();
        if (source === undefined || source === "") assert.equal(text, "Unavailable");
        else assert.ok(Math.abs(Number(text.replace(/[^\d.+-]/g, "")) - Number(source)) <= 0.0051,
          group.rows[i].Asset + " " + period + " agrees with its exported return");
      }
    }
    for (const boundary of [2, 8]) {
      const spacing = await table.locator("thead th").nth(boundary).evaluate((cell) => ({
        line: getComputedStyle(cell).borderLeftWidth,
        before: getComputedStyle(cell.previousElementSibling).paddingRight,
        after: getComputedStyle(cell).paddingLeft,
      }));
      assert.equal(spacing.line, "1px");
      assert.equal(spacing.before, spacing.after, "Group dividers have equal space on both sides");
    }
  }
  if (data.reportDate === reference.date) {
    const tables = await page
      .locator("table:not(.correlation-matrix)")
      .evaluateAll((tables) =>
        tables.map((t) =>
          [...t.querySelectorAll("tbody tr")].map((r) =>
            [...r.querySelectorAll("th,td")].filter((c) => !c.classList.contains("rank-change")).map((c) => c.innerText.trim()),
          ),
        ),
      );
    const originals = reference.tables.map((t) =>
      t.map((r) => r.map((cell) => normalized(cell === "Berkshire Hathaway Class B" ? "Berkshire Hathaway" : cell))),
    );
    const actual = tables.map((t) => t.map((r) => r.map(normalized)));
    // Fundamentals show each category once, as a row group, so compare the metric cells.
    const compare = [
      ...actual.slice(0, 4).map((table) => table.map((row) => [0, 1, 2, 9, 10, 4].map((i) => row[i]))),
      ...actual.slice(4, 7),
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
  async function desktopLegend(frame) {
    return frame.evaluate(async () => {
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const legend = document.querySelector("#legend");
      const rows = [...legend.querySelectorAll(".indicator-row")].map((row) => {
        const box = row.getBoundingClientRect();
        return {id: row.querySelector(".indicator").id.slice(7), left: box.left, top: box.top};
      });
      const a = SecretSatoshisChart, p = a.payload;
      const pinned = ["price_close", p.reportDate.slice(0, 4), "median", "mean"]
        .filter((id) => p.series.some((series) => series.id === id));
      const values = p.series.filter((series) => !pinned.includes(series.id)).map((series) => ({
        id: series.id, value: series.values[p.x.indexOf(a.readingPoint) - series.start] ?? -Infinity,
      }));
      const expected = [...pinned, ...values.sort((a, b) => b.value - a.value).map((series) => series.id)];
      return {
        two: document.body.classList.contains("legend-two-columns"),
        overflow: legend.scrollHeight > legend.clientHeight + 1,
        horizontalOverflow: legend.scrollWidth > legend.clientWidth + 1,
        expected,
        domOrder: rows.map((row) => row.id),
        visualOrder: rows.sort((a, b) => a.top - b.top || a.left - b.left).map((row) => row.id),
        columns: new Set(rows.map((row) => row.left)).size,
      };
    });
  }
  for (const frame of frames) {
    const legend = await desktopLegend(frame);
    assert.equal(legend.two, true, "Long dashboard legends use two columns");
    assert.equal(legend.columns, 2);
    assert.equal(legend.overflow, false, "Current dashboard series fit without scrolling");
    assert.equal(legend.horizontalOverflow, false);
    assert.deepEqual(legend.domOrder, legend.expected);
    assert.deepEqual(legend.visualOrder, legend.expected, "Pinned series, then descending values, across each row");
  }
  // The ranking follows the displayed candle/day, rather than a saved latest-date order.
  for (const [index, frame] of frames.entries()) {
    const element = page.locator("[data-chart-frame]").nth(index);
    await element.scrollIntoViewIfNeeded();
    const target = await frame.evaluate(() => {
      const a = SecretSatoshisChart, p = a.payload;
      const current = p.x.indexOf(a.readingPoint);
      const time = p.x[Math.max(1, Math.floor(current * (p.axisKind === "days" ? 0.5 : 0.97)))];
      return {time, x: a.view.charts[0].timeScale().timeToCoordinate(time), y: 120};
    });
    await frame.locator(".chart-pane").hover({position: {x: target.x, y: target.y}});
    await frame.waitForFunction(() => SecretSatoshisChart.readingPoint !== SecretSatoshisChart.payload.readingPoint, null, {timeout: 5000});
    const legend = await desktopLegend(frame);
    assert.deepEqual(legend.domOrder, legend.expected, "Hover ranks the actual displayed values");
    assert.deepEqual(legend.visualOrder, legend.expected);
    await page.mouse.move(0, 0);
    await frame.locator("#reset").click();
  }
  // Available vertical room determines the switch, with scrolling retained after two columns.
  const seasonal = frames[1];
  await page.locator("[data-chart-frame]").nth(1).scrollIntoViewIfNeeded();
  for (const [height, two, overflow] of [["1000px", false, false], ["250px", true, true], ["", true, false]]) {
    await seasonal.evaluate((height) => {
      document.querySelector(".indicators").style.height = height;
      dispatchEvent(new Event("resize"));
    }, height);
    await seasonal.waitForFunction((two) => document.body.classList.contains("legend-two-columns") === two, two, {timeout: 3000});
    const legend = await desktopLegend(seasonal);
    assert.equal(legend.two, two);
    assert.equal(legend.overflow, overflow);
    assert.equal(legend.columns, two ? 2 : 1);
    assert.equal(legend.horizontalOverflow, false);
    assert.deepEqual(legend.visualOrder, legend.expected);
  }
  async function checkReferenceLevels(technicalVisible = true) {
    const referenceChecks = await price.evaluate(async () => {
    window.__referenceDraws = {};
    const a = SecretSatoshisChart, scale = a.view.charts[0].timeScale();
    a.view.charts[0].applyOptions({});
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    return a.payload.referenceLines.map((level) => {
      const entry = Object.entries(window.__referenceDraws).find(([label]) => label.startsWith(level.name + " · "));
      let expected = 0;
      if (level.startDate) {
        const i = a.payload.x.findIndex((date) => date >= level.startDate);
        const left = i > 0 ? scale.timeToCoordinate(a.payload.x[i - 1]) : scale.timeToCoordinate(a.payload.x[0]);
        const right = scale.timeToCoordinate(a.payload.x[i]);
        const fraction = i > 0 ? (Date.parse(level.startDate) - Date.parse(a.payload.x[i - 1])) /
          (Date.parse(a.payload.x[i]) - Date.parse(a.payload.x[i - 1])) : 0;
        expected = Math.max(0, left + (right - left) * fraction);
      }
      return { level, draw: entry?.[1], expected };
    });
  });
    for (const {level, draw, expected} of referenceChecks) {
      if (!technicalVisible && ["support", "resistance"].includes(level.type)) {
        assert.equal(draw, undefined, level.name + " line and label hidden");
        continue;
      }
      assert.ok(draw, level.name + " label drawn");
      assert.ok(Math.abs(draw.from[0] - expected) < 0.01, level.name + " starts at its supplied date");
      assert.equal(draw.width, 1);
      assert.deepEqual(draw.dash, level.type ? [] : [4, 5]);
      assert.equal(draw.align, level.type ? "right" : "left");
      assert.equal(draw.x, level.type ? draw.to[0] - 8 : 8);
      assert.ok(draw.y < draw.from[1], "label is above the line");
    }
  }
  await checkReferenceLevels();
  assert.equal(await price.locator("#bitcoin-style").inputValue(), "candles");
  assert.equal(await price.locator("#candle-interval").inputValue(), "weekly");
  for (const interval of ["daily", "monthly", "weekly"]) {
    await price.locator("#candle-interval").selectOption(interval);
    assert.equal(
      await price.evaluate(() => SecretSatoshisChart.state.interval),
      interval,
    );
    await checkReferenceLevels();
  }
  await price.locator("#bitcoin-style").selectOption("line");
  await checkReferenceLevels();
  assert.equal(await price.locator("#support-resistance").isVisible(), true);
  assert.equal(await seasonal.locator("#support-resistance").isVisible(), false);
  await price.locator("#support-resistance").click();
  assert.equal(await price.locator("#support-resistance").getAttribute("aria-pressed"), "false");
  assert.equal(await price.evaluate(() => SecretSatoshisChart.state.supportResistance), false);
  assert.equal(await price.evaluate(() => SecretSatoshisChart.state.events), true);
  await checkReferenceLevels(false);
  await price.locator("#bitcoin-style").selectOption("candles");
  await price.locator("#candle-interval").selectOption("monthly");
  await price.locator("#scale-right").selectOption("linear");
  await checkReferenceLevels(false);
  const exportedLevels = await price.evaluate(async () => {
    window.__exportReferenceLabels = [];
    const image = await SecretSatoshisChart.exportImage(false);
    return {png: image.startsWith("data:image/png"), labels: window.__exportReferenceLabels,
      levels: SecretSatoshisChart.payload.referenceLines};
  });
  assert.equal(exportedLevels.png, true);
  for (const level of exportedLevels.levels) {
    assert.equal(exportedLevels.labels.some(label => label.startsWith(level.name + " · ")),
      !["support", "resistance"].includes(level.type), level.name + " respects export visibility");
  }
  await price.locator("#events").click();
  assert.equal(await price.evaluate(() => SecretSatoshisChart.state.supportResistance), false);
  await price.locator("#support-resistance").click();
  assert.equal(await price.evaluate(() => SecretSatoshisChart.state.events), false);
  await checkReferenceLevels();
  await price.locator("#support-resistance").click();
  await price.locator("#reset").click();
  assert.equal(await price.locator("#support-resistance").getAttribute("aria-pressed"), "true");
  assert.equal(await price.evaluate(() => SecretSatoshisChart.state.supportResistance), true);
  await checkReferenceLevels();
  for (const range of ["YTD", "1Y", "4Y", "10Y", "ALL", "2Y"]) {
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
  const relative = page.locator("#relative-value-table");
  assert.deepEqual(await relative.locator('tbody th[scope="row"]').allTextContents(),
    data.relative.map((row) => row.Asset === "Berkshire Hathaway Class B" ? "Berkshire Hathaway" : row.Asset));
  const latestValues = await relative.locator("tbody tr").evaluateAll((rows) => rows.map((row) =>
    [...row.querySelectorAll("th,td")].slice(1).map((cell) => cell.textContent.trim()),
  ));
  await relative.locator("tbody tr").first().evaluate((row) => { window.relativeFirstRow = row; });
  assert.equal(await page.locator('[data-rank-toggle="ytd"]').getAttribute("aria-pressed"), "true");
  assert.equal(await page.locator("[data-rank-toggle]").count(), 3);
  assert.equal(await page.locator('[data-rank-toggle="daily"]').count(), 0);
  assert.equal(await page.locator('[data-rank-period="daily"]').count(), 0);
  const requestsBeforeToggle = [];
  const recordRankRequest = (request) => requestsBeforeToggle.push(request.url());
  page.on("request", recordRankRequest);
  for (const period of ["wtd", "qtd", "ytd"]) {
    await page.locator(`[data-rank-toggle="${period}"]`).click();
    assert.equal(await page.locator(`[data-rank-toggle="${period}"]`).getAttribute("aria-pressed"), "true");
    assert.equal(await page.locator(`[data-rank-comparison="${period}"]`).isVisible(), true);
    assert.equal(await relative.locator(`[data-rank-period="${period}"]:visible`).count(), data.relative.length);
    const chips = await relative.locator(`[data-rank-period="${period}"]`).allTextContents();
    assert.deepEqual(chips, data.relative.map((row) => {
      const change = data.relativeRankChanges?.[period][row.Asset].change ?? null;
      return change === null ? "N/A" : change === 0 ? "—" : `${change > 0 ? "↑" : "↓"} ${Math.abs(change)}`;
    }));
    assert.deepEqual(await relative.locator("tbody tr").evaluateAll((rows) => rows.map((row) =>
      [...row.querySelectorAll("th,td")].slice(1).map((cell) => cell.textContent.trim()),
    )), latestValues, "Toggles preserve current values and row order");
    assert.equal(await relative.locator("tbody tr").first().evaluate((row) => row === window.relativeFirstRow), true, "Table rows are not rebuilt");
  }
  page.off("request", recordRankRequest);
  assert.deepEqual(requestsBeforeToggle, [], "Timeframe switching must not fetch data");
  assert.equal(await relative.locator('tbody th[scope="row"]').count(), data.relative.length);
  const flatChip = relative.locator('.rank-chip.flat:not([hidden])').first();
  if (await flatChip.count()) assert.equal(await flatChip.evaluate((el) => getComputedStyle(el).backgroundColor), "rgba(0, 0, 0, 0)");
  const rankToggle = page.locator('[data-rank-toggle="ytd"]');
  await rankToggle.press("End");
  assert.equal(await page.locator('[data-rank-toggle="ytd"]').getAttribute("aria-pressed"), "true");
  await page.locator('[data-rank-toggle="ytd"]').press("Home");
  await page.locator('[data-rank-toggle="wtd"]').press("ArrowLeft");
  assert.equal(await page.locator('[data-rank-toggle="ytd"]').getAttribute("aria-pressed"), "true");
  await page.locator('[data-rank-toggle="ytd"]').press("ArrowRight");
  await page.locator('[data-rank-toggle="wtd"]').press("ArrowRight");
  assert.equal(await page.locator('[data-rank-toggle="qtd"]').getAttribute("aria-pressed"), "true");
  await page.locator('[data-rank-toggle="qtd"]').press("ArrowRight");
  const chip = relative.locator('[data-rank-period="ytd"]').first();
  await chip.focus();
  assert.equal(await page.locator("#relative-rank-tooltip").isVisible(), true);
  assert.equal(await page.locator("#relative-rank-tooltip").textContent(), await chip.getAttribute("data-rank-description"));
  await chip.press("Escape");
  assert.equal(await page.locator("#relative-rank-tooltip").isVisible(), false);
  await rankToggle.focus();
  await chip.hover();
  assert.equal(await page.locator("#relative-rank-tooltip").isVisible(), true);
  await page.mouse.move(0, 0);
  assert.equal(await page.locator("#relative-rank-tooltip").isVisible(), false);
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
    await page.locator("#relative-valuation").scrollIntoViewIfNeeded();
    if (width === 1440) {
      assert.equal(await relative.evaluate((table) => table.parentElement.scrollWidth <= table.parentElement.clientWidth), true, "Desktop valuation table fits without horizontal scrolling");
      assert.equal(await relative.locator("thead th").evaluateAll((headers) => headers.every((header) => {
        const range = document.createRange();
        range.selectNodeContents(header);
        return range.getClientRects().length === 1;
      })), true, "Desktop valuation headers each occupy one line");
    }
    await page.locator("#relative-valuation").screenshot({ path: resolve(artifacts, `relative-valuation-${width}.png`) });
    if (width === 390) {
      await page.locator('[data-rank-toggle="ytd"]').click();
      const mobileChip = relative.locator('[data-rank-period="ytd"]').first();
      await mobileChip.click();
      const tip = page.locator("#relative-rank-tooltip");
      assert.equal(await tip.isVisible(), true);
      const bounds = await tip.boundingBox();
      assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= width, "Tooltip stays within mobile viewport");
      await page.locator('[data-rank-toggle="wtd"]').click();
      assert.equal(await tip.isVisible(), false);
      await page.evaluate(() => scrollTo(0, 0));
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
  assert.equal(await staticPage.locator('[data-rank-period="ytd"]:visible').count(), data.relative.length);
  assert.equal(await staticPage.locator('[data-rank-period="wtd"]:visible').count(), 0);
  assert.equal(await staticPage.locator('[data-rank-period="daily"]:visible').count(), 0);
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
