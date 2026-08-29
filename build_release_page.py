"""Generate the public data-release landing page, sitemap and structured data.

Run from the daily pipeline so the page and its dateModified track the release
rather than the moment someone last edited HTML by hand. Everything here is
derived from the CSVs on disk — no hardcoded dates.
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv"
BASE = "https://secretsatoshis.github.io/Bitcoin-Report-Library"
SITE = "https://secretsatoshis.com"
CODE_REPOSITORY = "https://github.com/SecretSatoshis/Bitcoin-Report-Library"
CODE_LICENSE = "https://www.gnu.org/licenses/gpl-3.0.html"
SOCIAL_IMAGE = f"{SITE}/assets/images/social-card.jpg"
SOCIAL_ALT = "Secret Satoshis — AI-Native Bitcoin Market Intelligence"
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def describe(path: Path) -> dict:
    """Row count, column count and real date coverage, read from the file."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        rows = list(csv.reader(handle))
    header, body = (rows[0], rows[1:]) if rows else ([], [])
    # Some compact summary tables place their date after metric columns (for
    # example roi_table.csv uses its third column). Search every cell so the
    # public per-file coverage cannot silently become incomplete when a CSV's
    # column order changes.
    dates = [c.strip() for r in body for c in r if DATE.fullmatch(c.strip())]
    return {
        "name": path.name,
        "label": path.stem.replace("_", " ").replace("brk ", "BRK ").strip().capitalize(),
        "columns": len(header),
        "rows": len(body),
        "first": min(dates) if dates else None,
        "last": max(dates) if dates else None,
        "size": path.stat().st_size,
    }


def human_size(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1048576:.1f} MB"


def collect() -> tuple[list[dict], str, str]:
    files = [describe(p) for p in sorted(CSV_DIR.glob("*.csv"))]
    dated = [f for f in files if f["first"]]
    first = min(f["first"] for f in dated) if dated else ""
    last = max(f["last"] for f in dated) if dated else ""
    return files, first, last


def structured(files: list[dict], first: str, last: str) -> dict:
    return {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": "Secret Satoshis Bitcoin Data Release",
        "description": (
            "A daily Bitcoin market and on-chain data release covering price, market "
            "capitalisation, supply, mining, network activity, valuation models and "
            "multi-asset performance comparisons. Published as CSV and regenerated "
            "each day by an open pipeline."
        ),
        "url": f"{BASE}/",
        "isAccessibleForFree": True,
        "dateModified": last,
        "temporalCoverage": f"{first}/{last}",
        "keywords": ["Bitcoin", "on-chain data", "market data", "open data", "BTC"],
        "creator": {"@type": "Organization", "name": "Secret Satoshis", "url": f"{SITE}/"},
        # The pipeline is GPL-3.0; the market data it ingests is third-party and
        # keeps its publishers' terms. The licence belongs on the code.
        "isBasedOn": {
            "@type": "SoftwareSourceCode",
            "name": "Bitcoin Report Library",
            "codeRepository": CODE_REPOSITORY,
            "license": CODE_LICENSE,
        },
        "distribution": [
            {
                "@type": "DataDownload",
                "encodingFormat": "text/csv",
                "contentUrl": f"{BASE}/csv/{f['name']}",
                "name": f["label"],
            }
            for f in files
        ],
    }


def write_sitemap(files: list[dict], last: str) -> None:
    urls = [(f"{BASE}/", "1.0")] + [(f"{BASE}/csv/{f['name']}", "0.5") for f in files]
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for loc, priority in urls:
        lines += ["  <url>", f"    <loc>{loc}</loc>", f"    <lastmod>{last}</lastmod>",
                  "    <changefreq>daily</changefreq>",
                  f"    <priority>{priority}</priority>", "  </url>"]
    lines.append("</urlset>")
    (ROOT / "sitemap.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index(files: list[dict], first: str, last: str) -> None:
    esc = html.escape
    rows = "\n".join(
        "        <tr>\n"
        f'          <td class="f"><a href="csv/{f["name"]}">{esc(f["label"])}</a>'
        f'<br><code>{esc(f["name"])}</code></td>\n'
        f'          <td>{f["first"] + " → " + f["last"] if f["first"] else "—"}</td>\n'
        f'          <td class="num">{f["rows"]:,}</td>\n'
        f'          <td class="num">{f["columns"]}</td>\n'
        f'          <td class="num">{human_size(f["size"])}</td>\n'
        "        </tr>"
        for f in files
    )
    description = (
        f"The daily Secret Satoshis Bitcoin data release: {len(files)} CSV files covering "
        "price, supply, mining, network activity and valuation, regenerated each day by an "
        "open pipeline."
    )
    (ROOT / "index.html").write_text(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bitcoin Data Release | Secret Satoshis</title>
<meta name="description" content="{esc(description, quote=True)}">
<link rel="canonical" href="{BASE}/">
<meta name="robots" content="index, follow, max-image-preview:large">
<meta name="theme-color" content="#08080c">
<link rel="icon" href="{SITE}/favicon.ico" sizes="32x32">
<link rel="apple-touch-icon" href="{SITE}/assets/images/favicon.png">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Secret Satoshis">
<meta property="og:title" content="Bitcoin Data Release | Secret Satoshis">
<meta property="og:description" content="{esc(description, quote=True)}">
<meta property="og:url" content="{BASE}/">
<meta property="og:image" content="{SOCIAL_IMAGE}">
<meta property="og:image:alt" content="{SOCIAL_ALT}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{SOCIAL_IMAGE}">
<script type="application/ld+json">
{json.dumps(structured(files, first, last), indent=2)}
</script>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#08080c; color:#e4e4ef;
         font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:15px; line-height:1.65; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:64px clamp(20px,5vw,48px) 96px; }}
  .eyebrow {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase; color:#5a5a74; margin:0 0 16px; }}
  .eyebrow span {{ color:#F7931A; }}
  h1 {{ font-size:clamp(28px,5vw,40px); line-height:1.1; margin:0 0 20px; letter-spacing:-.02em; }}
  p {{ max-width:66ch; color:#9090a8; font-weight:300; }}
  a {{ color:#F7931A; text-underline-offset:3px; }}
  h2 {{ font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:#5a5a74; margin:48px 0 14px; }}
  .tablewrap {{ overflow-x:auto; border:1px solid #1c1c2e; background:#0e0e16; }}
  table {{ border-collapse:collapse; width:100%; min-width:720px; font-size:13px; }}
  th {{ text-align:left; font-size:10px; letter-spacing:.14em; text-transform:uppercase;
        color:#5a5a74; padding:12px 16px; border-bottom:1px solid #2a2a42; background:#0b0b12; white-space:nowrap; }}
  td {{ padding:11px 16px; border-bottom:1px solid #1c1c2e; color:#9090a8; font-weight:300; vertical-align:top; }}
  tr:last-child td {{ border-bottom:0; }}
  td.f {{ color:#e4e4ef; }}
  td.f code {{ display:inline-block; margin-top:3px; font-size:11px; color:#5a5a74; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  footer {{ margin-top:56px; padding-top:22px; border-top:1px solid #1c1c2e; font-size:12px; color:#5a5a74; }}
</style>
</head>
<body>
  <div class="wrap">
    <p class="eyebrow"><span>//</span> Open Data</p>
    <h1>Bitcoin Data Release</h1>
    <p>
      The daily Secret Satoshis Bitcoin data release. {len(files)} CSV files covering price, market
      capitalisation, supply, mining, network activity, valuation models and multi-asset
      performance, regenerated each day by an open pipeline and read directly by
      <a href="https://dashboard.secretsatoshis.com/">the Market Dashboard</a>,
      <a href="https://charts.secretsatoshis.com/">the Chart Library</a> and
      <a href="{SITE}/">secretsatoshis.com</a>.
    </p>
    <p>
      Coverage runs <strong>{first}</strong> to <strong>{last}</strong>. The code that produces
      these files is public at
      <a href="{CODE_REPOSITORY}">Bitcoin-Report-Library</a> and licensed GPL-3.0; the market data
      it ingests is third-party and keeps the terms of its upstream publishers. Nothing here is
      investment advice.
    </p>

    <h2>Files</h2>
    <div class="tablewrap">
      <table>
        <thead>
          <tr><th>File</th><th>Coverage</th><th>Rows</th><th>Cols</th><th>Size</th></tr>
        </thead>
        <tbody>
{rows}
        </tbody>
      </table>
    </div>

    <footer>
      Updated {last}. Created by <a href="https://treybrunson.com/">Trey Brunson</a>.
      Don't trust. Verify.
    </footer>
  </div>
</body>
</html>
""", encoding="utf-8")


def main() -> None:
    files, first, last = collect()
    write_index(files, first, last)
    write_sitemap(files, last)
    print(f"release page: {len(files)} files, coverage {first} to {last}")


if __name__ == "__main__":
    main()
