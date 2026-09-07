import fs from "node:fs";

// Retired database tools must not return through transitive dependencies.
const lock = JSON.parse(fs.readFileSync("package-lock.json", "utf8"));
const forbidden = ["@lancedb/lancedb", "@huggingface/transformers", "sharp"];
const found = Object.keys(lock.packages || {}).filter((entry) => forbidden.some((name) => entry.endsWith(`node_modules/${name}`)));
if (found.length) throw new Error(`Retired migration dependencies leaked into the normal lockfile: ${found.join(", ")}`);
for (const name of forbidden) {
  if (fs.existsSync(`node_modules/${name}/package.json`)) throw new Error(`Unexpected root dependency ${name}; run npm ci to remove stale migration packages`);
}
console.log("Runtime dependency boundary passed: no LanceDB, Transformers or sharp in the normal installation.");
