import { defineConfig } from "vite";
export default defineConfig({
  root: ".generated",
  publicDir: "../static",
  build: { outDir: "../build", emptyOutDir: true, target: "es2022" },
  server: { fs: { allow: [".."] } },
});
