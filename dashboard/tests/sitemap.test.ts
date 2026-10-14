import { test } from "node:test";
import assert from "node:assert/strict";
import { updateSitemap } from "../scripts/sitemap";

const template = "<url><loc>https://dashboard.secretsatoshis.com/</loc><lastmod>2026-10-14</lastmod></url>";

test("a newer data release advances the sitemap date reproducibly", () => {
  const output = updateSitemap(template, "2026-10-15T20:34:43+00:00");
  assert.equal(output, template.replace("2026-10-14", "2026-10-15"));
  assert.equal(updateSitemap(output, "2026-10-15T20:34:43+00:00"), output);
});

test("older data cannot undo a newer content update", () => {
  assert.equal(updateSitemap(template, "2026-10-12T20:34:43+00:00"), template);
});

test("missing or invalid dates cannot publish an invalid sitemap", () => {
  assert.throws(() => updateSitemap("<url></url>", "2026-10-15T20:34:43Z"), /Invalid sitemap/);
  assert.throws(() => updateSitemap(template, "not-a-date"), /Invalid sitemap/);
});
