import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { runOwnedCommand } from "../packages/core/src/owned-command.js";

test("owned conversion waits for termination, escalates an uncooperative child, and removes its listeners", { timeout: 10_000 }, async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-owned-command-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const started = path.join(directory, "started.json");
  const program = path.join(directory, "child.mjs");
  fs.writeFileSync(program, `import fs from 'node:fs';
process.on('SIGTERM', () => {});
fs.writeFileSync(process.argv[2], JSON.stringify({pid:process.pid}));
setInterval(() => {}, 1000);
`);
  const controller = new AbortController();
  const result = runOwnedCommand(process.execPath, [program, started], { signal: controller.signal });
  const rejected = assert.rejects(result, { name: "AbortError" });
  const limit = Date.now() + 3000;
  while (!fs.existsSync(started)) { assert(Date.now() < limit); await new Promise(resolve => setTimeout(resolve, 5)); }
  const pid = Number(JSON.parse(fs.readFileSync(started, "utf8")).pid);
  controller.abort(); await rejected;
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
  await assert.rejects(runOwnedCommand(path.join(directory, "missing-binary"), []), { code: "ENOENT" });
  await runOwnedCommand(process.execPath, ["-e", "process.exit(0)"]);
});
