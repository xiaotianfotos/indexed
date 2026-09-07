import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

// Keep the observer outside the sandbox: macOS refuses to execute setuid ps
// inside sandbox-exec, even with allow-default. The command and its descendants
// inherit the no-Python policy; the observer only records their executable names.
const { values, positionals } = parseArgs({ allowPositionals: true, options: { report: { type: "string" } } });
assert(process.platform === "darwin" && values.report && positionals.length, "Usage: no-python-run.ts --report PATH -- COMMAND ARGS...");
const report = path.resolve(values.report);
assert(!fs.existsSync(report), "Refusing to overwrite a validation report");
const policy = fileURLToPath(new URL("no-python.sb", import.meta.url));
const child = spawn("/usr/bin/sandbox-exec", ["-f", policy, ...positionals], { stdio: "inherit" });
let spawnError = "", samplerError = "", peakHelperRSSBytes = 0;
const executables = new Set<string>();
const startedAt = new Date().toISOString();
const sample = () => {
  if (!child.pid) return;
  try {
    const rows = execFileSync("/bin/ps", ["-A", "-o", "pid=,ppid=,rss=,comm="], { encoding: "utf8", timeout: 1000 })
      .trim().split("\n").map(line => /^\s*(\d+)\s+(\d+)\s+(\d+)\s+(.+)$/.exec(line)).filter(row => row !== null);
    const owned = new Set([child.pid]);
    for (let pass = 0; pass < 6; pass++) for (const row of rows) if (owned.has(Number(row[1])) || owned.has(Number(row[2]))) {
      owned.add(Number(row[1]));
      const executable = path.basename(row[4]!);
      executables.add(executable);
      if (executable === "indexed-apple-embedding") peakHelperRSSBytes = Math.max(peakHelperRSSBytes, Number(row[3]) * 1024);
    }
  } catch (error) { samplerError = String(error); }
};
const timer = setInterval(sample, 500);
const forward = (signal: NodeJS.Signals) => { child.kill(signal); };
process.on("SIGINT", forward); process.on("SIGTERM", forward);
const code = await new Promise<number | null>(resolve => {
  child.once("error", error => { spawnError = String(error); });
  child.once("close", resolve);
});
clearInterval(timer);
process.off("SIGINT", forward); process.off("SIGTERM", forward);
fs.mkdirSync(path.dirname(report), { recursive: true });
const unexpectedPython = [...executables].some(name => /^(python|pip)[\d.]*$/i.test(name));
fs.writeFileSync(report, JSON.stringify({ schema: 1, startedAt, finishedAt: new Date().toISOString(), exitCode: code,
  policy: "no-python.sb", executables: [...executables], peakHelperRSSBytes, unexpectedPython, spawnError, samplerError }, null, 2), { mode: 0o600 });
process.exitCode = code === 0 && !spawnError && !samplerError && !unexpectedPython ? 0 : 1;
