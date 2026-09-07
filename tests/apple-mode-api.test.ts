import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { startServer } from "@indexed/server";

test("API and CLI reject E without overwriting configuration; old E remains repairable through GET", async context => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-apple-mode-api-"));
  const filename = path.join(root, "config.json");
  const env = { ...process.env, INDEXED_CONFIG: filename, INDEXED_HOME: root, INDEXED_STATE_DIR: path.join(root, "state") };
  process.env.INDEXED_CONFIG = filename;
  process.env.INDEXED_HOME = root;
  process.env.INDEXED_STATE_DIR = env.INDEXED_STATE_DIR;
  const server = await startServer({ host: "127.0.0.1", port: 0 });
  context.after(async () => { await server.close(); fs.rmSync(root, { recursive: true, force: true }); });
  const before = fs.readFileSync(filename, "utf8");
  const rejected = await fetch(`${server.url}api/profiles/default`, { method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ embedding: { provider: "apple-native", native: { executionMode: "e" } }, storage: { provider: "local" } }) });
  assert.equal(rejected.status, 400);
  assert.match((await rejected.json()).error, /E 模式已退出/);
  assert.equal(fs.readFileSync(filename, "utf8"), before);
  await assert.rejects(promisify(execFile)(process.execPath,
    ["--import", "tsx", "apps/cli/src/index.ts", "embedding", "configure", "--model-package", root, "--execution-mode", "E"],
    { env, timeout: 10_000 }), /E 模式已退出/);
  assert.equal(fs.readFileSync(filename, "utf8"), before);
  const legacy = JSON.parse(before);
  legacy.profiles.default.embedding.provider = "apple-native";
  legacy.profiles.default.embedding.native.executionMode = "e";
  fs.writeFileSync(filename, JSON.stringify(legacy));
  const visible = await (await fetch(`${server.url}api/config`)).json();
  assert.equal(visible.profiles.default.embedding.native.executionMode, "e");
  assert.match(visible.profiles.default.embedding.native.executionModeIssue, /E 模式已退出/);
  assert.deepEqual(JSON.parse(fs.readFileSync(filename, "utf8")), legacy);
});
