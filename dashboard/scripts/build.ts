import { build } from "vite";
import { cpSync, existsSync, renameSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { prepareData, root } from "./prepare-data";

prepareData();
const stage = resolve(root, ".build-staging"),
  output = resolve(root, "build"),
  previous = resolve(root, ".build-previous");
rmSync(stage, { recursive: true, force: true });
try {
  await build({
    configFile: resolve(root, "vite.config.ts"),
    build: { outDir: stage, emptyOutDir: true },
  });
  cpSync(resolve(root, ".generated/data"), resolve(stage, "data"), {
    recursive: true,
  });
  rmSync(previous, { recursive: true, force: true });
  if (existsSync(output)) renameSync(output, previous);
  try {
    renameSync(stage, output);
  } catch (e) {
    if (existsSync(previous)) renameSync(previous, output);
    throw e;
  }
  rmSync(previous, { recursive: true, force: true });
} catch (e) {
  rmSync(stage, { recursive: true, force: true });
  throw e;
}
