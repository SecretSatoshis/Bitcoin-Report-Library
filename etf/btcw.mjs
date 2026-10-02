// Read WisdomTree's BTCW holdings and NAV history from inside a headless browser, since the fund
// data loads only within its web page. Usage: node btcw.mjs <playwright index.mjs> <page url>
const [module, page] = process.argv.slice(2);
const { chromium } = await import(module);

const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ locale: "en-US", viewport: { width: 1440, height: 900 } });
  const tab = await context.newPage();
  await tab.goto(page, { waitUntil: "domcontentloaded", timeout: 90000 });
  await tab.waitForTimeout(6000);
  const result = await tab.evaluate(async () => {
    const response = await fetch("/api/fund-holdings/48684713");
    if (!response.ok) throw new Error(`holdings request returned ${response.status}`);
    const history = await fetch("/api/fund-history/48684713?dataset=navHistory&format=json");
    if (!history.ok) throw new Error(`NAV history request returned ${history.status}`);
    const text = document.body.innerText.replace(/\s+/g, " ");
    const shares = text.match(/Shares Outstanding ([\d,]+)/i);
    const nav = text.match(/NAV \$([\d.,]+)/);
    return { holdings: await response.json(), navHistory: await history.json(),
             sharesOutstanding: shares && shares[1], nav: nav && nav[1] };
  });
  process.stdout.write(JSON.stringify(result));
} finally {
  await browser.close();
}
