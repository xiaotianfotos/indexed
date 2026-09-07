import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

/** Match the bundled extension's public manifest key; never trust arbitrary extension origins. */
export function bundledExtensionOrigins(dashboardDirectory: string): string[] {
  const candidates = [
    path.resolve(dashboardDirectory, "../extension/manifest.json"),
    path.resolve(dashboardDirectory, "../extension/static/manifest.json"),
  ];
  for (const candidate of candidates) {
    if (!fs.existsSync(candidate)) continue;
    const manifest = JSON.parse(fs.readFileSync(candidate, "utf8")) as { key?: unknown };
    if (typeof manifest.key !== "string" || !manifest.key.trim()) return [];
    const key = Buffer.from(manifest.key, "base64");
    if (!key.length) return [];
    // Chromium's crx_file::id_util::GenerateId: first 16 SHA256 bytes, alphabet a-p.
    const id = createHash("sha256").update(key).digest("hex").slice(0, 32)
      .replace(/[0-9a-f]/g, (digit) => String.fromCharCode(97 + Number.parseInt(digit, 16)));
    return [`chrome-extension://${id}`];
  }
  return [];
}
