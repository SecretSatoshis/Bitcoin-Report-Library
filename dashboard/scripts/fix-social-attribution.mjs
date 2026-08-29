#!/usr/bin/env node
/**
 * Replace Evidence's hardcoded publisher attribution in the built output.
 *
 * @evidence-dev/preprocess writes `twitter:site="@evidence_dev"` into every
 * page's <svelte:head>. A page-level <svelte:head> can add tags but cannot
 * remove one, so the only deterministic fix is a post-build rewrite. The
 * value exists in both prerendered HTML and the client-side hydration bundle;
 * both must be corrected or the browser can restore the upstream value.
 *
 * Fails loudly if the tag disappears upstream, so this stops being a silent
 * no-op the day Evidence changes it.
 */
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BUILD = new URL('../build/', import.meta.url).pathname;
const FROM = /@evidence_dev/g;
const TO = '@SecretSatoshis';

async function* builtDocuments(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* builtDocuments(path);
    else if (entry.name.endsWith('.html') || entry.name.endsWith('.js')) yield path;
  }
}

let patched = 0;
let scanned = 0;
for await (const file of builtDocuments(BUILD)) {
  scanned += 1;
  const before = await readFile(file, 'utf8');
  const after = before.replace(FROM, TO);
  if (after !== before) {
    await writeFile(file, after, 'utf8');
    patched += 1;
  }
}

if (scanned === 0) {
  console.error('fix-social-attribution: no built HTML or JavaScript found — did the build run?');
  process.exit(1);
}
if (patched === 0) {
  console.error(
    `fix-social-attribution: scanned ${scanned} files and found no @evidence_dev tag.\n` +
      'Evidence may have changed its metadata template. Verify the built output ' +
      'still carries correct attribution, then update or remove this script.'
  );
  process.exit(1);
}

for await (const file of builtDocuments(BUILD)) {
  if ((await readFile(file, 'utf8')).includes('@evidence_dev')) {
    console.error(`fix-social-attribution: upstream attribution remains in ${file}`);
    process.exit(1);
  }
}

console.log(
  `fix-social-attribution: rewrote publisher attribution in ${patched}/${scanned} built files`
);
