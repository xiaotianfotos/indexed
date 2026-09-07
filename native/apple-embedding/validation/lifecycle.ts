import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { parseArgs } from "node:util";
import { setTimeout as delay } from "node:timers/promises";
import { AppleEmbeddingBackend, appleEmbeddingOptionsFromProfile, installAppleEmbeddingModel } from "@indexed/apple-embedding-backend";
import { activeProfile, clearRuntimeEmbeddingOverride, directConfig, loadConfig, setRuntimeEmbeddingOverride } from "@indexed/config";
import { listLocalVectors } from "@indexed/clients/local-vectors";
import { scanLibrary, searchAssets } from "@indexed/core";

// Functional acceptance only. Performance acceptance uses backend-ingest.ts.
// Install a real converted package and use fresh caches/indexes for every mode.
const { values } = parseArgs({ options: { package: { type: "string" }, binary: { type: "string" }, output: { type: "string" } } });
for (const key of ["package", "binary", "output"] as const) assert(values[key], `Missing --${key}`);
const output = path.resolve(values.output!);
assert(!fs.existsSync(output), "Refusing to overwrite validation results");
const binary = path.resolve(values.binary!);
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-native-lifecycle-"));
process.env.INDEXED_HOME = path.join(scratch, "home");
process.env.INDEXED_CONFIG = path.join(scratch, "config.json");
process.env.INDEXED_DATA_DIR = path.join(scratch, "data");
process.env.INDEXED_STATE_DIR = path.join(scratch, "state");
const rows: Array<Record<string, unknown>> = [];
try {
  const installed = await installAppleEmbeddingModel({ binary, source: path.resolve(values.package!), modelsDirectory: path.join(scratch, "models") });
  assert.equal(typeof installed.path, "string");
  const reused = await installAppleEmbeddingModel({ binary, source: path.resolve(values.package!), modelsDirectory: path.join(scratch, "models") });
  assert.equal(reused.path, installed.path);
  console.log("Native model installation and verified reuse passed");
  const library = path.join(scratch, "library");
  fs.mkdirSync(library);
  const fixture = fileURLToPath(new URL("../mixed-query-set/assets/mug_red_wood.png", import.meta.url));
  fs.copyFileSync(fixture, path.join(library, "mug.png"));
  fs.writeFileSync(path.join(library, "notes.md"), "# Indexed functional fixture\n\nVideo, image and document search uses an isolated local vector index.\n");
  execFileSync("ffmpeg", ["-v", "error", "-loop", "1", "-i", fixture, "-t", "10", "-vf", "scale=640:-2", "-r", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", path.join(library, "clip.mp4")], { timeout: 60_000, stdio: "pipe" });
  for (const mode of ["a", "b", "c", "d"] as const) {
    const config = loadConfig();
    config.library = { ...config.library, roots: [library], kinds: ["video", "image", "document"], rootKinds: {}, autoScan: false };
    const profile = config.profiles[config.activeProfile];
    profile.storage = { ...profile.storage, provider: "local", path: path.join(scratch, `vectors-${mode}`) };
    profile.embedding = { ...profile.embedding, provider: "apple-native", model: "wemm-embedding-2b-apple-2048", dimension: 2048, inputStyle: "wemm",
      native: { ...profile.embedding.native, binary, modelPackage: installed.path, executionMode: mode, coreMLCache: path.join(scratch, "coreml"), autoRestart: false } };
    profile.video = { ...profile.video, chunkSeconds: 10, maxChunkSeconds: 10, fps: 2, width: 768 };
    const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile));
    try {
      if (mode === "a") { await backend.validate(true); await backend.prepare(); console.log("Full package verification and native Core ML preparation passed"); }
      const runtime = await backend.start();
      setRuntimeEmbeddingOverride(config.activeProfile, runtime);
      assert.equal(backend.status().executionMode, mode);
      const scan = await scanLibrary(library, { config, recordPerformance: false });
      assert(scan.ok && scan.indexed === 3 && scan.failed === 0, JSON.stringify(scan.errors));
      const vectors = await listLocalVectors(profile.storage.assetIndex, directConfig(activeProfile(config)), { returnData: true });
      assert(vectors.length >= 3 && vectors.every(row => row.data?.float32?.length === 2048 && row.data.float32.every(Number.isFinite)));
      const queryCounts: Record<string, number> = {};
      for (const kind of ["video", "image", "document"] as const) {
        const search = await searchAssets("Indexed video image document", { config, kind, limit: 3 });
        assert(search.hits.length && search.hits.every(hit => hit.path.startsWith(library + path.sep)));
        queryCounts[kind] = search.hits.length;
      }
      let cancellationMS: number | null = null;
      let mlpOperationsAtCancellation = 0;
      if (mode === "c" || mode === "d") {
        const base = runtime.baseUrl.replace(/\/v1\/?$/, "").replace(/\/$/, "");
        const headers = { "content-type": "application/json", Authorization: `Bearer ${runtime.apiKey}` };
        const requestID = `lifecycle-${mode}`;
        const mlpOperations = (health: Record<string, unknown>) => {
          const native = health.private_ane as { native_profile?: { mlp?: { operations?: number } } } | undefined;
          return Number(native?.native_profile?.mlp?.operations || 0);
        };
        const beforeOperations = mlpOperations(await backend.health());
        const pending = fetch(`${base}/v1/embeddings`, { method: "POST", headers, signal: AbortSignal.timeout(30_000),
          body: JSON.stringify({ model: runtime.model, request_id: requestID, input: "Video memory stores visual features and temporal context. ".repeat(500) }) });
        void pending.catch(() => {}); // The awaited request below still owns failures.
        // Wait for actual private MLP work, so admission/tokenization alone
        // cannot satisfy the cancellation check.
        let active = false;
        for (let attempt = 0; attempt < 1000; attempt++) {
          const health = await backend.health();
          if (Number(health.running_requests) > 0 && mlpOperations(health) > beforeOperations) {
            mlpOperationsAtCancellation = mlpOperations(health) - beforeOperations;
            active = true; break;
          }
          await delay(10);
        }
        assert(active, "Real inference did not execute private MLP work");
        const started = performance.now();
        const cancelled = await fetch(`${base}/v1/requests/${requestID}`, { method: "DELETE", headers });
        assert.equal(cancelled.status, 202, await cancelled.text());
        const response = await pending;
        assert.equal(response.status, 499, await response.text());
        cancellationMS = performance.now() - started;
        const health = await backend.health();
        assert.equal(health.running_requests, 0); assert.equal(health.queued_requests, 0);
        const recovered = await searchAssets("Indexed document", { config, kind: "document", limit: 1 });
        assert.equal(recovered.hits.length, 1);
      }
      rows.push({ mode, indexed: scan.indexed, vectors: vectors.length, queryCounts, cancellationMS, mlpOperationsAtCancellation });
      console.log(`${mode.toUpperCase()} ingestion, persisted vectors, retrieval${cancellationMS === null ? "" : ", cancellation and recovery"} passed`);
    } catch (error) {
      console.error(`${mode.toUpperCase()} helper diagnostics:`, backend.status().stderrTail.join("\n"));
      throw error;
    } finally {
      clearRuntimeEmbeddingOverride(config.activeProfile);
      await backend.stop();
    }
  }
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify({ schema: 1, status: "passed", modelFingerprint: installed.package_fingerprint, installation: true, reuse: true, preparation: true, rows }, null, 2), { mode: 0o600 });
} finally {
  fs.rmSync(scratch, { recursive: true, force: true });
}
