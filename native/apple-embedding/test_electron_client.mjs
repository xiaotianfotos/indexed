#!/usr/bin/env node

import assert from "node:assert/strict";
import { resolve } from "node:path";
import { AppleEmbeddingService } from "./electron-service-client.mjs";

const [binary, modelPackage, decoderAsset, decoderAssetType = "segment"] = process.argv.slice(2);
if (!binary || !modelPackage) {
  throw new Error(
    "usage: test_electron_client.mjs BINARY MODEL_PACKAGE [DECODER_ASSET] [segment|bundle]",
  );
}
if (!new Set(["segment", "bundle"]).has(decoderAssetType)) {
  throw new Error("decoder asset type must be segment or bundle");
}

const service = new AppleEmbeddingService({
  binary: resolve(binary),
  modelPackage: resolve(modelPackage),
  decoderSegment:
    decoderAsset && decoderAssetType === "segment" ? resolve(decoderAsset) : undefined,
  decoderBundle:
    decoderAsset && decoderAssetType === "bundle" ? resolve(decoderAsset) : undefined,
  decoderMinimumTokens: decoderAssetType === "bundle" ? 128 : undefined,
  extraArgs: ["--development-deterministic-engine", "--skip-warmup"],
});

try {
  const prepared = await service.prepare();
  assert.equal(prepared.status, "prepared");
  if (decoderAssetType === "segment" && decoderAsset) {
    assert.equal(prepared.decoder_segment_fingerprint.length, 64);
  }
  if (decoderAssetType === "bundle" && decoderAsset) {
    assert.equal(prepared.decoder_bundle_fingerprint.length, 64);
  }
  const [first, second] = await Promise.all([service.start(), service.start()]);
  assert.equal(first.url, second.url);
  assert.equal(first.default_embedding_space, "deterministic-test-2048");
  const health = await service.health();
  assert.equal(health.status, "ready");
  const response = await service.embed({
    input: "Electron lifecycle smoke test",
    requestId: "electron-smoke",
  });
  assert.equal(response.data[0].embedding.length, 2048);
  assert.equal(response.indexed.request_id, "electron-smoke");
  process.stdout.write(`${JSON.stringify({ status: "passed", url: first.url })}\n`);
} finally {
  await service.stop();
}
