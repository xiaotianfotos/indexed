import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  AppleEmbeddingBackend,
  appleEmbeddingOptionsFromProfile,
  managedAppleEmbeddingAssets,
} from "@indexed/apple-embedding-backend";
import {
  clearRuntimeEmbeddingOverride,
  directConfig,
  setRuntimeEmbeddingOverride,
} from "@indexed/config";
import { embedDocument } from "@indexed/clients/direct-cloud";

const fixture = fileURLToPath(new URL("./fixtures/fake-apple-embedding-helper.mjs", import.meta.url));

function environment() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-apple-backend-"));
  const modelPackage = path.join(directory, "model");
  fs.mkdirSync(modelPackage);
  const profile = {
    id: "apple-test",
    label: "Apple test",
    spaceId: "",
    embedding: {
      provider: "apple-native",
      baseUrl: "",
      apiKey: "",
      model: "",
      dimension: 256,
      inputStyle: "wemm",
      native: {
        binary: fixture,
        modelPackage,
        decoderSegment: "",
        decoderBundles: [],
        coreMLCache: "",
        mode: "fast",
        visionCompute: "ane",
        decoderMinimumTokens: 128,
        maxQueuedRequests: 16,
        startupTimeoutSeconds: 30,
        maintenanceTimeoutSeconds: 30,
        autoRestart: false,
        maxRestarts: 0,
      },
    },
    storage: { provider: "local", path: "" },
    video: {},
  };
  return { directory, modelPackage, profile };
}

test("managed Apple backend owns auth, lifecycle and runtime semantic space", async (context) => {
  const { directory, profile } = environment();
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile));
  assert.equal(appleEmbeddingOptionsFromProfile(profile).privateANE?.videoDownProjection, "q8");
  assert.equal(appleEmbeddingOptionsFromProfile(profile, { privateANE: { videoDownProjection: "q8" } }).privateANE?.videoDownProjection, "q8");
  context.after(() => backend.stop());
  assert.equal((await backend.validate(true)).status, "valid");
  assert.equal((await backend.prepare()).status, "prepared");
  const [first, second] = await Promise.all([backend.start(), backend.start()]);
  assert.equal(first.baseUrl, second.baseUrl);
  assert.equal(first.embeddingSpace, "fake-wemm-256-same-space");
  assert.equal(backend.status().state, "ready");
  assert.equal(backend.status().ready?.default_embedding_space, first.embeddingSpace);
  assert.equal((await backend.health()).status, "ready");

  const unauthorized = await fetch(`${first.baseUrl}/health`);
  assert.equal(unauthorized.status, 401);
  setRuntimeEmbeddingOverride(profile.id, first);
  context.after(() => clearRuntimeEmbeddingOverride(profile.id));
  const direct = directConfig(profile);
  assert.equal(direct.embeddingBaseUrl, first.baseUrl);
  assert.equal(direct.embeddingSpace, first.embeddingSpace);
  assert.equal(direct.embeddingApiKey, first.apiKey);
  assert.deepEqual(await embedDocument("native backend", direct), [1, ...Array(255).fill(0)]);

  const oldPid = backend.status().pid;
  const restarted = await backend.restart();
  assert.equal(restarted.baseUrl, first.baseUrl);
  assert.notEqual(backend.status().pid, oldPid);
  assert.equal(backend.status().restartCount, 1);
  await backend.stop();
  assert.equal(backend.status().state, "stopped");
});

test("managed Apple backend supervises one crash on the same loopback endpoint", async (context) => {
  const { directory, profile } = environment();
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const marker = path.join(directory, "crash-once");
  const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile, {
    autoRestart: true,
    maxRestarts: 2,
    extraArgs: ["--crash-once-marker", marker],
  }));
  context.after(() => backend.stop());
  const first = await backend.start();
  const firstPid = backend.status().pid;
  const firstToken = first.apiKey;
  const deadline = Date.now() + 5_000;
  while (Date.now() < deadline) {
    if (backend.status().state === "ready" && backend.status().restartCount === 1 && backend.status().pid !== firstPid) break;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  assert.equal(backend.status().state, "ready");
  assert.equal(backend.status().restartCount, 1);
  assert.notEqual(backend.status().pid, firstPid);
  assert.equal(backend.runtime?.baseUrl, first.baseUrl);
  assert.notEqual(backend.runtime?.apiKey, firstToken);
  assert.equal((await backend.health()).status, "ready");
});

test("stopping during cold start cannot resurrect a late native helper", async (context) => {
  const { directory, profile } = environment();
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile, {
    extraArgs: ["--ready-delay-ms", "2000"],
  }));
  context.after(() => backend.stop());
  const starting = backend.start();
  await new Promise((resolve) => setTimeout(resolve, 50));
  await backend.stop();
  await assert.rejects(starting, /启动已取消|就绪前退出/);
  assert.equal(backend.status().state, "stopped");
  assert.equal(backend.status().pid, null);
  assert.equal(backend.runtime, null);
});

test("startup rejects a helper that advertises the wrong vector contract", async (context) => {
  const { directory, profile } = environment();
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile, {
    extraArgs: ["--advertised-dimension", "128"],
  }));
  context.after(() => backend.stop());
  await assert.rejects(backend.start(), /维度不匹配/);
  assert.equal(backend.status().state, "failed");
  assert.equal(backend.status().pid, null);
  assert.equal(backend.runtime, null);
});

test("A and B select the expected public Apple compute stack", async (context) => {
  const { directory, profile } = environment();
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const cases = [
    { executionMode: "a" as const, vision: "gpu", language: "gpu", bundles: [] },
    { executionMode: "b" as const, vision: "ane", language: "gpu", bundles: [] },
  ];
  for (const selected of cases) {
    const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile, {
      executionMode: selected.executionMode,
    }));
    try {
      await backend.start();
      const health = await backend.health();
      assert.equal(backend.status().executionMode, selected.executionMode);
      assert.equal(health.vision_compute, selected.vision);
      assert.equal(health.language_compute, selected.language);
    } finally {
      await backend.stop();
    }
  }
});

test("retired E and unknown modes fail before resolving or spawning any helper", () => {
  for (const native of [{ executionMode: "e" }, { mode: "ane" }, { executionMode: "invalid" }]) {
    assert.throws(() => appleEmbeddingOptionsFromProfile({ embedding: { native } }), /退出产品|重新选择模式/);
    // Exercise untyped disk/API input crossing the typed constructor boundary.
    const value = JSON.parse(JSON.stringify({ ...native, modelPackage: "/missing-model", binary: "/missing-helper" }));
    assert.throws(() => new AppleEmbeddingBackend(value), /退出产品|重新选择模式/);
  }
});

test("C and D use the same native helper without Python assets", async (context) => {
  const { directory, profile } = environment();
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  for (const executionMode of ["c", "d"] as const) {
    const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile({ ...profile,
      embedding: { ...profile.embedding, native: { ...profile.embedding.native, executionMode,
        researchPython: "/absent/python", researchService: "/absent/service.py",
        privateANE: { experimentRoot: "/absent/experiment", bridge: "/absent/bridge", mil: "/absent/mil" } } },
    }));
    try {
      await backend.start();
      const health = await backend.health();
      assert.equal(backend.options.binary, fixture);
      assert.equal(health.execution_mode, executionMode);
      assert.equal(health.video_pipeline, 2);
      assert.equal(health.video_down_projection, "q8");
      assert.equal(health.recurrence_block_size, 8);
      assert.equal(health.recurrence_layer_slots, "0");
      assert.equal(health.recurrence_query_scale, 4096);
      assert.equal(health.recurrence_max_tokens, 8192);
      assert.equal(health.recurrence_verify_reference, false);
    } finally { await backend.stop(); }
  }
});

test("unsupported Swift kernel controls fail instead of silently falling back", (context) => {
  const { directory, profile } = environment();
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  for (const privateANE of [{ mlpVariant: 9 }, { videoDownProjection: "fp16" as const }]) {
    assert.throws(() => new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile, { executionMode: "c", privateANE })), /Swift C\/D/);
  }
  for (const privateANE of [{ sequenceLength: 256 }, { recurrenceBlockSize: 4 }, { recurrenceLayerSlots: [0, 2] }, { recurrenceIODtype: "fp32" as const }, { recurrenceVerifyReference: true }]) {
    assert.throws(() => new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile, { executionMode: "d", privateANE })), /Swift D/);
  }
});

test("D profile override is optional and discovered without a research environment", (context) => {
  const { directory } = environment();
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  assert.deepEqual(managedAppleEmbeddingAssets(directory), { recurrenceProfile: "" });
  const recurrenceProfile = path.join(directory, "runtime", "apple-embedding", "profiles", "q8_seq2112_mlp24_recurrence_slot0_fp16_block8_verified.json");
  fs.mkdirSync(path.dirname(recurrenceProfile), { recursive: true });
  fs.writeFileSync(recurrenceProfile, "fixture\n");
  assert.deepEqual(managedAppleEmbeddingAssets(directory), { recurrenceProfile });
});
