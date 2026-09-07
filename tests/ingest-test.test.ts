import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { normalizeConfig } from "@indexed/config";
import type { ScanResult } from "@indexed/core";
import { IngestTestController, type IngestTestRow } from "../apps/server/src/ingest-test.js";
import { bestTestModes, testMetricValue } from "../apps/dashboard/src/ingest-test.js";

test("per-file time divides batch milliseconds by files and excludes invalid counts", () => {
  const row: IngestTestRow = { mode: "a", state: "done", elapsedMs: 4350, files: 10, samples: [] };
  assert.equal(testMetricValue(row, "averageFileMs"), 435);
  assert.equal(testMetricValue(row, "elapsedMs"), 4350);
  const missingCount = { ...row };
  delete missingCount.files;
  assert.equal(testMetricValue(missingCount, "averageFileMs"), undefined);
  for (const files of [0, -1, NaN, Infinity, 1.5]) {
    assert.equal(testMetricValue({ ...row, files }, "averageFileMs"), undefined);
  }
  assert.deepEqual(bestTestModes([row, { ...row, mode: "b", elapsedMs: 3623 }, { ...row, mode: "c", files: 0 }], "averageFileMs"), ["b"]);
});

function setup(t: { after(fn: () => void): void }) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-dry-test-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const config = normalizeConfig({ activeProfile: "default", profiles: { default: { embedding: { provider: "apple-native", native: { executionMode: "b" } } } } });
  return { root, config, storePath: path.join(root, "report.json") };
}
function sample(count = 10): ScanResult {
  return { ok: true, root: "/test", embeddingSpace: "test", discovered: count, indexed: count, skipped: 0, failed: 0,
    deleted: 0, vectors: count, remaining: 0, elapsedMs: 2000, indexedByKind: { image: count, video: 0, document: 0 },
    workMsByKind: { image: 2000, video: 0, document: 0 }, errors: [], warnings: [], inputSignature: "same-input",
    modelTimings: { requests: count, preprocessMs: 10, visionMs: 400, languageMs: 1000, totalMs: 1500 } };
}

test("isolated test runs ABCD without warmup, cools between rounds, restores config", async (t) => {
  const { root, config, storePath } = setup(t);
  const original = JSON.stringify(config);
  const modes: string[] = [];
  const calls: number[] = [];
  let cooldowns = 0;
  const runtime = { reconcile: async (next: typeof config) => { modes.push(next.profiles.default.embedding.native.executionMode); }, stop: async () => {} };
  const controller = new IngestTestController({ runtime, storePath, cooldown: async () => { cooldowns++; }, config: () => config, scan: async (_root, options) => {
    assert.equal(options.dryRun, true); assert.equal(options.recordPerformance, false); assert.equal(options.benchmark, true);
    calls.push(options.limit!); return { ...sample(options.limit), elapsedMs: options.limit === 1 ? 99000 : 2000 };
  } });
  controller.start(root, 10, 3);
  assert.throws(() => controller.start(root), /正在运行/);
  const report = await controller.completed();
  assert.deepEqual(modes, ["a", "b", "c", "d", "b"]);
  assert.deepEqual(calls, Array(12).fill(10));
  assert.equal(cooldowns, 11);
  assert.equal(report.protocol, "no-warmup-cooldown-v1");
  assert.ok(report.rows.every((row) => row.warmupMs === undefined));
  assert.equal(report.state, "done");
  assert.ok(report.rows.every((row) => row.elapsedMs === 2000 && row.filesPerSecond === 5 && row.samples.length === 3));
  assert.ok(report.rows.every((row) => row.modelMs === 1400));
  assert.equal(JSON.stringify(config), original);
  const reloaded = new IngestTestController({ runtime, storePath });
  assert.deepEqual(reloaded.report, JSON.parse(JSON.stringify(report)));
});

test("mode failure remains visible, other modes continue, input changes are rejected", async (t) => {
  const { root, config, storePath } = setup(t);
  let mode = "";
  const controller = new IngestTestController({ storePath, cooldown: async () => {}, config: () => config,
    runtime: { reconcile: async (next) => { mode = next.profiles.default.embedding.native.executionMode; }, stop: async () => {} },
    scan: async (_root, options) => {
      if (mode === "b") throw new Error("model failed");
      return { ...sample(options.limit), inputSignature: mode === "c" ? "changed" : "same-input" };
    },
  });
  controller.start(root); const report = await controller.completed();
  assert.equal(report.state, "error");
  assert.deepEqual(report.rows.map((row) => row.state), ["done", "error", "error", "done"]);
  assert.match(report.rows[2]!.error!, /发生变化/);
  assert.equal(mode, "b");
});

test("stop aborts model work, does not run further modes, restores original mode", async (t) => {
  const { root, config, storePath } = setup(t);
  let scanning!: () => void;
  const entered = new Promise<void>((resolve) => { scanning = resolve; });
  const modes: string[] = [];
  let stops = 0;
  const controller = new IngestTestController({ storePath, config: () => config,
    runtime: { reconcile: async (next) => { modes.push(next.profiles.default.embedding.native.executionMode); }, stop: async () => { stops++; } },
    scan: async (_root, options) => { scanning(); return new Promise((_resolve, reject) => options.signal!.addEventListener("abort", () => reject(new Error("aborted")), { once: true })); },
  });
  controller.start(root); await entered; await controller.stop();
  assert.equal(controller.report.state, "cancelled"); assert.equal(stops, 1);
  assert.deepEqual(modes, ["a", "b"]);
});

test("best metrics exclude errors/missing timings and allow ties", () => {
  const rows: IngestTestRow[] = [
    { mode: "a", state: "done", elapsedMs: 2000, filesPerSecond: 5, samples: [] },
    { mode: "b", state: "done", elapsedMs: 1000, filesPerSecond: 10, samples: [] },
    { mode: "c", state: "done", elapsedMs: 1000, filesPerSecond: 10, samples: [] },
    { mode: "d", state: "error", elapsedMs: 100, filesPerSecond: 100, samples: [] },
  ];
  assert.deepEqual(bestTestModes(rows, "elapsedMs"), ["b", "c"]);
  assert.deepEqual(bestTestModes(rows, "filesPerSecond"), ["b", "c"]);
  assert.deepEqual(bestTestModes(rows, "visionMs"), []);
  assert.deepEqual(bestTestModes(rows.slice(0, 1), "elapsedMs"), []);
});

test("stopping during cooldown does not start another scan", async (t) => {
  const { root, config, storePath } = setup(t);
  let entered!: () => void;
  const waiting = new Promise<void>((resolve) => { entered = resolve; });
  let scans = 0;
  const controller = new IngestTestController({ storePath, config: () => config,
    runtime: { reconcile: async () => {}, stop: async () => {} },
    scan: async () => { scans++; return sample(); },
    cooldown: async (signal) => { entered(); await new Promise<void>((resolve) => signal.addEventListener("abort", () => resolve(), { once: true })); },
  });
  controller.start(root, 10, 3);
  await waiting;
  assert.match(controller.report.phase, /5 秒/);
  await controller.stop();
  assert.equal(scans, 1);
  assert.equal(controller.report.state, "cancelled");
});

test("existing requests prevent a test from taking over the model", (t) => {
  const { root, config, storePath } = setup(t);
  const controller = new IngestTestController({ storePath, config: () => config,
    runtime: { reconcile: async () => { assert.fail("must not switch mode"); }, stop: async () => {} },
    canStart: () => { throw new Error("existing request"); },
  });
  assert.throws(() => controller.start(root), /existing request/);
  assert.equal(controller.report.state, "idle");
});

test("a report interrupted by server exit reloads as cancelled, retaining completed rows", (t) => {
  const { storePath } = setup(t);
  fs.writeFileSync(storePath, JSON.stringify({ schema: 1, state: "running", root: "/test", rows: [{ mode: "a", state: "done", elapsedMs: 20, samples: [] }] }));
  const controller = new IngestTestController({ storePath, runtime: { reconcile: async () => {}, stop: async () => {} } });
  assert.equal(controller.busy, false);
  assert.equal(controller.report.state, "cancelled");
  assert.equal(controller.report.rows[0]?.elapsedMs, 20);
});
