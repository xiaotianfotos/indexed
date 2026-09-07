import assert from "node:assert/strict";
import test from "node:test";
import { scanPresentation, recordComparison, compareRuns } from "../apps/dashboard/src/scan-presentation.js";

test("mode comparisons require complete fresh runs of the identical input group", () => {
  const job = { state: "done", startedAt: 1000, finishedAt: 11000, total: 10, done: 10, indexed: 10,
    benchmarkKey: "same-input", executionMode: "a" };
  const first = recordComparison([], job);
  assert.equal(first.length, 1);
  assert.equal(recordComparison(first, job), first);
  for (const change of [{ failed: 1 }, { skipped: 1 }, { indexed: 9 }, { state: "cancelled" }, { benchmarkKey: "" }]) {
    assert.equal(recordComparison(first, { ...job, ...change, startedAt: 2000 }), first);
  }
  let runs = recordComparison(first, { ...job, executionMode: "b", startedAt: 20000, finishedAt: 28000 });
  runs = recordComparison(runs, { ...job, executionMode: "b", startedAt: 30000, finishedAt: 36000 });
  runs = recordComparison(runs, { ...job, executionMode: "c", benchmarkKey: "different-input", startedAt: 40000, finishedAt: 41000 });
  const comparison = compareRuns(runs, "same-input");
  assert.equal(comparison.length, 2);
  assert.equal(comparison[1]?.milliseconds, 7000);
  assert.equal(comparison[1]?.speedup, 10000 / 7000);
  assert.equal(compareRuns(runs, "different-input")[0]?.speedup, null);
});

test("scan card reports successful wall-time throughput", () => {
  const result = scanPresentation({ state: "done", startedAt: 1000, finishedAt: 11000, total: 10, done: 10, indexed: 10 }, 99000);
  assert.equal(result.elapsedMs, 10000);
  assert.equal(result.filesPerSecond, 1);
  assert.equal(result.percent, 100);
  assert.equal(result.busy, false);
});

test("failed files contribute to progress but not successful throughput", () => {
  const result = scanPresentation({ state: "running", startedAt: 1000, total: 10, done: 5, indexed: 3 }, 6000);
  assert.equal(result.percent, 50);
  assert.equal(result.filesPerSecond, 0.6);
});

test("queued, empty and incomplete terminal scans do not invent measurements", () => {
  assert.equal(scanPresentation({ state: "queued" }).filesPerSecond, null);
  assert.equal(scanPresentation({ state: "done", startedAt: 1000, total: 0 }).elapsedMs, 0);
  assert.equal(scanPresentation({ state: "cancelled", total: 10, done: 3 }).percent, 30);
  assert.equal(scanPresentation({ state: "running", startedAt: 5000 }, 1000).elapsedMs, 0);
});
