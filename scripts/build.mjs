import fs from "node:fs";
import path from "node:path";
import { build } from "esbuild";

const extensionDist = path.resolve("dist/extension");
fs.rmSync(path.resolve("dist"), { recursive: true, force: true });
fs.mkdirSync(extensionDist, { recursive: true });
fs.cpSync("apps/extension/static", extensionDist, { recursive: true });

await build({
  entryPoints: {
    "service-worker": "apps/extension/src/service-worker.ts",
    popup: "apps/extension/src/popup.ts",
  },
  outdir: extensionDist,
  bundle: true,
  platform: "browser",
  format: "esm",
  target: "chrome120",
  sourcemap: true,
});

await build({
  entryPoints: {
    "content-script": "apps/extension/src/content-script.ts",
    search: "apps/extension/src/search.ts",
  },
  outdir: extensionDist,
  bundle: true,
  platform: "browser",
  format: "iife",
  target: "chrome120",
  sourcemap: true,
});

console.log("Built Indexed Chrome extension into dist/extension/");
