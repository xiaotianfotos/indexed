import fs from "node:fs";
import path from "node:path";

const roots = ["apps", "packages", "skills", "scripts"];
const blocked = [
  { name: "personal bundle namespace", pattern: /\bcom\.(?!indexed\b)[a-z0-9_-]+\.indexed\b/i },
  { name: "private LAN address", pattern: /192\.168\.\d+\.\d+/ },
  { name: "absolute user path", pattern: /\/(?:Users|Volumes)\/[A-Za-z0-9_.-]+\// },
  { name: "Alibaba AccessKey", pattern: /LTAI[A-Za-z0-9]{12,}/ },
];

function files(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) return files(target);
    return [target];
  });
}

const findings = [];
for (const root of roots) {
  if (!fs.existsSync(root)) continue;
  for (const file of files(root)) {
    if (["scripts/audit-hardcoding.mjs", "scripts/validate-build.mjs"].includes(file)) continue;
    if (!/\.(?:ts|tsx|js|mjs|json|html|md|css|ya?ml)$/.test(file)) continue;
    const content = fs.readFileSync(file, "utf8");
    for (const rule of blocked) if (rule.pattern.test(content)) findings.push(`${file}: ${rule.name}`);
  }
}

if (findings.length) {
  console.error(findings.join("\n"));
  process.exitCode = 1;
} else {
  console.log("Hardcoding audit passed.");
}
