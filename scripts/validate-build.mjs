import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const extensionRoot = path.resolve("dist/extension");
for (const file of fs.readdirSync(extensionRoot, { recursive: true })) {
  if (/\.(?:md|markdown|mdx|rst|adoc)$/i.test(file)
      && !/^(?:LICENSE|LICENCE|COPYING|NOTICE|THIRD_PARTY_NOTICES|ACKNOWLEDGMENTS)(?:\.|$)/i.test(path.basename(file))) {
    throw new Error(`Local documentation leaked into the extension: ${file}`);
  }
}
if (!fs.existsSync(path.join(extensionRoot, "THIRD_PARTY_NOTICES.md"))) {
  throw new Error("Extension distribution is missing its third-party notices");
}
const manifest = JSON.parse(fs.readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));
const canonicalMark = fs.readFileSync("apps/dashboard/indexed.svg", "utf8");
for (const builtMark of ["apps/extension/static/icons/indexed.svg", "dist/dashboard/indexed.svg", "dist/extension/icons/indexed.svg"]) {
  if (fs.readFileSync(builtMark, "utf8") !== canonicalMark) {
    throw new Error(`Built brand mark differs from the player-derived canonical icon: ${builtMark}`);
  }
}
const referenced = new Set([
  manifest.background?.service_worker,
  manifest.action?.default_popup,
  ...Object.values(manifest.icons || {}),
  ...Object.values(manifest.action?.default_icon || {}),
  ...(manifest.content_scripts || []).flatMap((entry) => entry.js || []),
].filter(Boolean));

for (const relative of referenced) {
  const target = path.join(extensionRoot, relative);
  if (!fs.existsSync(target)) throw new Error(`Extension manifest references a missing file: ${relative}`);
}

for (const file of [
  "dist/cli/index.js",
  "dist/server/index.js",
  "dist/dashboard/app.js",
  "dist/extension/service-worker.js",
  "dist/extension/content-script.js",
  "dist/extension/popup.js",
  "dist/extension/search.js",
]) execFileSync(process.execPath, ["--check", file], { stdio: "pipe" });

// Syntax checks do not execute bundled module initializers. Keep a real CLI
// startup smoke test so CommonJS dependencies cannot silently break the ESM build.
execFileSync(process.execPath, ["dist/cli/index.js", "--help"], { stdio: "pipe" });

const bundledText = [
  "dist/cli/index.js",
  "dist/server/index.js",
  "dist/extension/service-worker.js",
  "dist/extension/content-script.js",
  "dist/extension/popup.js",
].map((file) => fs.readFileSync(file, "utf8")).join("\n");

if (/@indexed\//.test(bundledText)) throw new Error("Bundled output still contains unresolved workspace imports");
if (/youtubevisual4096|192\.168\.\d+\.\d+|\/(?:Users|Volumes)\//.test(bundledText)) {
  throw new Error("Bundled output contains a blocked hard-coded environment value");
}

console.log(`Build validation passed (${referenced.size} manifest assets checked).`);
