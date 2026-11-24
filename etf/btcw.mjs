// Read WisdomTree's BTCW holdings and NAV history from inside a headless browser, since the fund
// data loads only within its web page. Usage: node btcw.mjs <playwright index.mjs> <page url>
const [module, page] = process.argv.slice(2);
if (!module || !page) {
  throw new Error("Usage: node btcw.mjs <playwright index.mjs> <page url>");
}
const { chromium } = await import(module);

const HOLDINGS_API = "/api/fund-holdings/48684713";
const HISTORY_API = "/api/fund-history/48684713?dataset=navHistory&format=json";

const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ locale: "en-US", viewport: { width: 1440, height: 900 } });
  const tab = await context.newPage();

  // Capture API responses triggered by the page itself when available.
  let capturedHoldings = null;
  let capturedHistory = null;
  const holdingsPromise = tab.waitForResponse(
    (res) => res.url().includes(HOLDINGS_API) && res.status() === 200,
    { timeout: 30000 },
  ).then(async (res) => { capturedHoldings = await res.json(); }).catch(() => null);
  const historyPromise = tab.waitForResponse(
    (res) => res.url().includes("/api/fund-history/48684713") && res.url().includes("navHistory") && res.status() === 200,
    { timeout: 30000 },
  ).then(async (res) => { capturedHistory = await res.json(); }).catch(() => null);

  const response = await tab.goto(page, { waitUntil: "domcontentloaded", timeout: 90000 });
  if (response && response.status() >= 400) {
    throw new Error(`BTCW page returned HTTP ${response.status()} for ${page}`);
  }

  // Wait for fund metrics or API responses rather than a blind fixed sleep.
  await Promise.race([
    Promise.all([holdingsPromise, historyPromise]),
    tab.waitForFunction(
      () => /Shares Outstanding\s+[\d,]+/i.test(document.body?.innerText || "") &&
            /NAV\s+\$[\d.,]+/.test(document.body?.innerText || ""),
      { timeout: 30000 },
    ).catch(() => null),
  ]);

  const result = await tab.evaluate(async ({ holdingsApi, historyApi, initialHoldings, initialHistory }) => {
    let holdings = initialHoldings;
    if (!holdings) {
      const holdingsRes = await fetch(holdingsApi);
      if (!holdingsRes.ok) {
        throw new Error(`holdings request (${holdingsApi}) returned HTTP ${holdingsRes.status}`);
      }
      holdings = await holdingsRes.json();
    }
    let navHistory = initialHistory;
    if (!navHistory) {
      const historyRes = await fetch(historyApi);
      if (!historyRes.ok) {
        throw new Error(`NAV history request (${historyApi}) returned HTTP ${historyRes.status}`);
      }
      navHistory = await historyRes.json();
    }
    const text = (document.body?.innerText || "").replace(/\s+/g, " ");
    const shares = text.match(/Shares Outstanding[:\s]+([\d,]+)/i);
    const nav = text.match(/NAV[:\s]+\$([\d.,]+)/i);
    return {
      holdings,
      navHistory,
      sharesOutstanding: shares ? shares[1] : null,
      nav: nav ? nav[1] : null,
    };
  }, {
    holdingsApi: HOLDINGS_API,
    historyApi: HISTORY_API,
    initialHoldings: capturedHoldings,
    initialHistory: capturedHistory,
  });

  if (!Array.isArray(result.holdings) || result.holdings.length === 0) {
    throw new Error("BTCW holdings payload is empty or not an array");
  }
  if (!Array.isArray(result.navHistory) || result.navHistory.length === 0) {
    throw new Error("BTCW navHistory payload is empty or not an array");
  }
  process.stdout.write(JSON.stringify(result));
} finally {
  await browser.close();
}
