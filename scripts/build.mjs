import fs from "node:fs";
import path from "node:path";
import { build } from "esbuild";

const root = process.cwd();
const dist = path.join(root, "dist");
fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(dist, { recursive: true });

await build({
  entryPoints: ["apps/server/src/index.ts"],
  outfile: "dist/server/index.js",
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node20",
  sourcemap: true,
  external: ["@zvec/zvec", "word-extractor"],
});

await build({
  entryPoints: ["apps/cli/src/index.ts"],
  outfile: "dist/cli/index.js",
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node20",
  sourcemap: true,
  external: ["@zvec/zvec", "word-extractor"],
});

await build({
  entryPoints: ["apps/dashboard/src/main.ts"],
  outfile: "dist/dashboard/app.js",
  bundle: true,
  platform: "browser",
  format: "esm",
  target: "chrome120",
  sourcemap: true,
});
fs.copyFileSync("apps/dashboard/index.html", "dist/dashboard/index.html");
fs.copyFileSync("apps/dashboard/src/styles.css", "dist/dashboard/styles.css");
fs.copyFileSync("apps/dashboard/indexed.svg", "dist/dashboard/indexed.svg");

const extensionDist = path.join(dist, "extension");
fs.cpSync("apps/extension/static", extensionDist, {
  recursive: true,
  // Ignored documents still exist locally. Keep them out of built extensions
  // too, while preserving the third-party attribution shipped with the code.
  filter: (source) => !/\.(?:md|markdown|mdx|rst|adoc)$/i.test(source)
    || /^(?:LICENSE|LICENCE|COPYING|NOTICE|THIRD_PARTY_NOTICES|ACKNOWLEDGMENTS)(?:\.|$)/i.test(path.basename(source)),
});
// The player control is the canonical Indexed mark. Dashboard and extension
// surfaces ship the exact same geometry instead of maintaining separate logos.
fs.copyFileSync("apps/dashboard/indexed.svg", path.join(extensionDist, "icons/indexed.svg"));
const extensionManifestPath = path.join(extensionDist, "manifest.json");
const extensionManifest = JSON.parse(fs.readFileSync(extensionManifestPath, "utf8"));
const localExtensionKeyPath = path.join(root, ".indexed-extension-key");
const extensionKey = String(
  process.env.INDEXED_EXTENSION_KEY
  || (fs.existsSync(localExtensionKeyPath) ? fs.readFileSync(localExtensionKeyPath, "utf8") : "")
  || extensionManifest.key
).trim();
if (extensionKey) extensionManifest.key = extensionKey;
else delete extensionManifest.key;
fs.writeFileSync(extensionManifestPath, `${JSON.stringify(extensionManifest, null, 2)}\n`);
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

for (const target of ["bin/indexed", "dist/cli/index.js"]) fs.chmodSync(target, 0o755);
console.log(`Built Indexed into ${path.relative(root, dist)}/`);
