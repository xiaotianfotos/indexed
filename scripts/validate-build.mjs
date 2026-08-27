import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const extensionRoot = path.resolve("dist/extension");
const manifest = JSON.parse(fs.readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
const referenced = new Set([
  manifest.background?.service_worker,
  manifest.action?.default_popup,
  ...Object.values(manifest.icons || {}),
  ...Object.values(manifest.action?.default_icon || {}),
  ...(manifest.content_scripts || []).flatMap((entry) => entry.js || []),
].filter(Boolean));

for (const relative of referenced) {
  if (!fs.existsSync(path.join(extensionRoot, relative))) {
    throw new Error(`Extension manifest references a missing file: ${relative}`);
  }
}

for (const file of [
  "dist/extension/service-worker.js",
  "dist/extension/content-script.js",
  "dist/extension/popup.js",
  "dist/extension/search.js",
]) execFileSync(process.execPath, ["--check", file], { stdio: "pipe" });

const bundledText = [
  "dist/extension/service-worker.js",
  "dist/extension/content-script.js",
  "dist/extension/popup.js",
  "dist/extension/search.js",
].map((file) => fs.readFileSync(file, "utf8")).join("\n");

if (/@indexed\//.test(bundledText)) throw new Error("Bundled output contains unresolved workspace imports");
if (/LTAI[A-Za-z0-9]{12,}|192\.168\.\d+\.\d+|\/(?:Users|Volumes)\//.test(bundledText)) {
  throw new Error("Bundled output contains a blocked environment value");
}

console.log(`Build validation passed (${referenced.size} manifest assets checked).`);
