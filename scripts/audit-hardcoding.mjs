import fs from "node:fs";
import path from "node:path";

const skippedDirectories = new Set([".git", "dist", "node_modules"]);
const skippedFiles = new Set(["scripts/audit-hardcoding.mjs"]);
const rules = [
  { name: "private LAN address", pattern: /192\.168\.\d+\.\d+/ },
  { name: "absolute user path", pattern: /\/(?:Users|Volumes)\/[A-Za-z0-9_.-]+\// },
  { name: "Alibaba AccessKey", pattern: /LTAI[A-Za-z0-9]{12,}/ },
  { name: "private key", pattern: /BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY/ },
  { name: "historical personal Index", pattern: /youtubevisual4096/ },
];

function files(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    if (skippedDirectories.has(entry.name)) return [];
    const target = path.join(directory, entry.name);
    return entry.isDirectory() ? files(target) : [target];
  });
}

const findings = [];
for (const file of files(".")) {
  const normalized = file.replace(/^\.\//, "");
  if (skippedFiles.has(normalized)) continue;
  if (!/\.(?:ts|tsx|js|mjs|json|html|md|css|ya?ml)$/.test(file)) continue;
  const content = fs.readFileSync(file, "utf8");
  for (const rule of rules) if (rule.pattern.test(content)) findings.push(`${normalized}: ${rule.name}`);
}

if (findings.length) {
  console.error(findings.join("\n"));
  process.exitCode = 1;
} else {
  console.log("Hardcoding audit passed.");
}
