import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  estimateIngestPerformance,
  readIngestPerformanceHistory,
  recordIngestPerformance,
} from "@indexed/core";

function performanceConfig(databasePath: string, model = "vendor-model", endpoint = "https://vectors.vendor.test/v1") {
  return {
    version: 1,
    activeProfile: "default",
    library: { roots: [], rootKinds: {}, autoScan: false },
    profiles: {
      default: {
        label: "Default",
        spaceId: "",
        embedding: { baseUrl: endpoint, model, dimension: 4096, inputStyle: "wemm" },
        reranker: { enabled: false, baseUrl: "", model: "", candidates: 20 },
        video: { chunkSeconds: 30, maxChunkSeconds: 60, fps: 2, width: 1280, maxSegments: 240 },
        storage: {
          provider: "local",
          path: databasePath,
          assetIndex: "library-assets",
          accessKeyId: "must-not-be-recorded",
          accessKeySecret: "secret-must-not-be-recorded",
        },
      },
    },
  };
}

test("ingest history follows the supplier workload across new databases without storing user data", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-ingest-performance-"));
  const target = path.join(directory, "state", "history.json");
  const originalDatabase = path.join(directory, "private-project-name", "first-db");
  const config = performanceConfig(originalDatabase);
  try {
    const sample = recordIngestPerformance({
      config,
      historyPath: target,
      sampledAt: 1_800_000_000_000,
      elapsedMs: 12_000,
      concurrency: 2,
      indexedByKind: { video: 2, image: 2, document: 0 },
      workMsByKind: { video: 20_000, image: 2_000, document: 0 },
      failed: 0,
      vectors: 6,
    });
    assert.ok(sample);
    assert.equal(sample.documentChunkVersion, "paragraph-window-v1");
    assert.equal(fs.statSync(target).mode & 0o777, 0o600);
    const raw = fs.readFileSync(target, "utf8");
    assert.doesNotMatch(raw, /private-project-name|must-not-be-recorded|secret-must-not-be-recorded|first-db/);

    const freshDatabase = performanceConfig(path.join(directory, "brand-new-db"));
    const exact = estimateIngestPerformance({
      config: freshDatabase,
      historyPath: target,
      pendingByKind: { video: 4, image: 2, document: 0 },
    });
    assert.equal(exact.source, "exact");
    assert.equal(exact.sampleCount, 1);
    assert.equal(exact.pendingAssets, 6);
    assert.equal(exact.ratesByKind.video.millisecondsPerAsset, 10_000);
    assert.equal(exact.ratesByKind.image.millisecondsPerAsset, 1_000);
    assert.equal(exact.estimatedMs, 21_000);
    assert.equal(exact.lastSampleAt, 1_800_000_000_000);

    const newModel = estimateIngestPerformance({
      config: performanceConfig(path.join(directory, "other-db"), "newer-model"),
      historyPath: target,
      pendingByKind: { video: 1 },
    });
    assert.equal(newModel.source, "provider");
    assert.equal(newModel.estimatedMs, 10_000);

    const newSupplier = estimateIngestPerformance({
      config: performanceConfig(path.join(directory, "third-db"), "vendor-model", "https://other.vendor.test/v1"),
      historyPath: target,
      pendingByKind: { video: 1 },
    });
    assert.equal(newSupplier.source, "none");
    assert.equal(newSupplier.estimatedMs, null);
    assert.deepEqual(newSupplier.unavailableKinds, ["video"]);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("empty and corrupted performance history degrade to no estimate", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-ingest-performance-empty-"));
  const target = path.join(directory, "history.json");
  try {
    fs.writeFileSync(target, "not json");
    assert.deepEqual(readIngestPerformanceHistory(target), { version: 1, samples: [] });
    const estimate = estimateIngestPerformance({
      config: performanceConfig(path.join(directory, "db")),
      historyPath: target,
      pendingByKind: { document: 3 },
    });
    assert.equal(estimate.sampleCount, 0);
    assert.equal(estimate.estimatedMs, null);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
