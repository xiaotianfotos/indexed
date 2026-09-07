import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { parseArgs } from "node:util";
import { setTimeout as delay } from "node:timers/promises";
import { AppleEmbeddingBackend, appleEmbeddingOptionsFromProfile } from "@indexed/apple-embedding-backend";
import { activeProfile, clearRuntimeEmbeddingOverride, directConfig, loadConfig, setRuntimeEmbeddingOverride } from "@indexed/config";
import { listLocalVectors } from "@indexed/clients/local-vectors";
import { scanLibrary, searchAssets } from "@indexed/core";
import { vectorAgreement } from "./reference.js";

// Explicit maintainer command. Only the selected input is copied into a temporary
// library; every run gets a fresh real zvec index. Raw vectors stay in --output,
// which must be a private/ignored location for user-provided videos.
const { values } = parseArgs({ options: {
  video: { type: "string" }, package: { type: "string" }, binary: { type: "string" },
  output: { type: "string" }, modes: { type: "string", default: "a,b,c,c,b,a" },
} });
for (const key of ["video", "package", "binary", "output"] as const) if (!values[key]) throw new Error(`Missing --${key}`);
const modes = values.modes!.split(",");
assert(modes.every(mode => ["a", "b", "c", "d"].includes(mode)), "Unknown mode");
const output = path.resolve(values.output!);
assert(!fs.existsSync(output), "Use a new output; never overwrite measurements");
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-real-video-"));
process.env.INDEXED_CONFIG = path.join(scratch, "config.json");
process.env.INDEXED_DATA_DIR = path.join(scratch, "data");
process.env.INDEXED_STATE_DIR = path.join(scratch, "state");
process.env.INDEXED_HOME = path.join(scratch, "home");
const library = path.join(scratch, "library");
fs.mkdirSync(library);
fs.copyFileSync(path.resolve(values.video!), path.join(library, "input.mp4"), fs.constants.COPYFILE_FICLONE);
const digest = createHash("sha256");
for await (const chunk of fs.createReadStream(path.join(library, "input.mp4"))) digest.update(chunk as Buffer);
const binary = path.resolve(values.binary!), packagePath = path.resolve(values.package!);
execFileSync(binary, ["validate-model", "--package", packagePath, "--full"], { timeout: 180_000, stdio: "pipe" });
const runs: Array<Record<string, unknown>> = [];
const references = new Map<number, number[]>();
let baselineSignature = "";
try {
  for (const [index, mode] of modes.entries()) {
    if (index) await delay(5000);
    const config = loadConfig();
    config.library = { ...config.library, roots: [library], kinds: ["video"], rootKinds: {}, autoScan: false };
    const profile = config.profiles[config.activeProfile];
    profile.storage = { ...profile.storage, provider: "local", path: path.join(scratch, `store-${index}`) };
    profile.embedding = { ...profile.embedding, provider: "apple-native", baseUrl: "", model: "wemm-embedding-2b-apple-2048",
      dimension: 2048, inputStyle: "wemm", native: { ...profile.embedding.native, binary, modelPackage: packagePath,
        coreMLCache: path.join(scratch, `coreml-${mode}`), executionMode: mode, autoRestart: false } };
    profile.spaceId = "";
    profile.video = { ...profile.video, chunkSeconds: 10, maxChunkSeconds: 10, fps: 2, width: 768, maxSegments: 1000 };
    const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile));
    const loadStarted = performance.now();
    try {
      const runtime = await backend.start();
      const startupMS = performance.now() - loadStarted;
      setRuntimeEmbeddingOverride(config.activeProfile, runtime);
      const before = await backend.health();
      console.log(`${index + 1}/${modes.length} ${mode!.toUpperCase()} ready (${Math.round(startupMS)} ms); full video ingestion`);
      let last = -1;
      const result = await scanLibrary(library, { config, recordPerformance: false, benchmark: true,
        onProgress(job) {
          const completed = job.currentUnitsDone || 0;
          if (completed !== last && (completed % 5 === 0 || completed === job.currentUnitsTotal)) {
            console.log(`${mode}: ${completed}/${job.currentUnitsTotal || 0} segments`); last = completed;
          }
        } });
      assert(result.ok && result.indexed === 1 && result.failed === 0 && result.remaining === 0, JSON.stringify(result.errors));
      if (!baselineSignature) baselineSignature = result.inputSignature || "";
      assert.equal(result.inputSignature, baselineSignature, "Ingest input signature changed");
      const after = await backend.health();
      const rows = await listLocalVectors(profile.storage.assetIndex, directConfig(activeProfile(config)), { returnData: true });
      rows.sort((a, b) => Number(a.metadata?.segment_index) - Number(b.metadata?.segment_index));
      assert.equal(rows.length, result.vectors);
      assert(rows.length > 0);
      let minimumCosine = 1, maximumRelativeL2 = 0, maximumAbsoluteError = 0;
      const vectors = rows.map((row, segment) => {
        const vector = row.data?.float32;
        assert(vector && vector.length === 2048 && vector.every(Number.isFinite));
        if (!references.has(segment)) references.set(segment, vector);
        const agreement = vectorAgreement(references.get(segment)!, vector);
        minimumCosine = Math.min(minimumCosine, agreement.cosine);
        maximumRelativeL2 = Math.max(maximumRelativeL2, agreement.relativeL2);
        maximumAbsoluteError = Math.max(maximumAbsoluteError, agreement.maximumAbsoluteError);
        return { segment, start: row.metadata?.start_time, end: row.metadata?.end_time, frames: row.metadata?.frame_count, vector };
      });
      const qualityPassed = minimumCosine >= 0.999 && maximumRelativeL2 <= 0.05 && maximumAbsoluteError <= 0.02;
      // Real query through Core -> embedding -> zvec, outside the timed scan.
      const search = await searchAssets("多人参与狼人杀游戏", { config, kind: "video", limit: 3 });
      assert(search.hits.length > 0 && search.hits.every(hit => hit.path === path.join(library, "input.mp4")));
      const row = { mode, startupMS, elapsedMS: result.elapsedMs, mediaSeconds: result.processedMediaSeconds,
        throughput: (result.processedMediaSeconds || 0) / (result.elapsedMs / 1000),
        vectorCount: rows.length, frameCount: vectors.reduce((sum, row) => sum + Number(row.frames), 0),
        inputSignature: result.inputSignature, embeddingSpace: result.embeddingSpace, modelTimings: result.modelTimings,
        qualityPassed, minimumCosine, maximumRelativeL2, maximumAbsoluteError, before, after, vectors, queryHits: search.hits.length };
      runs.push(row);
      fs.mkdirSync(path.dirname(output), { recursive: true });
      fs.writeFileSync(output, JSON.stringify({ schema: 1, status: "running", inputSHA256: digest.copy().digest("hex"),
        cpu: os.cpus()[0]?.model, totalMemoryBytes: os.totalmem(), binarySHA256: createHash("sha256").update(fs.readFileSync(binary)).digest("hex"),
        video: profile.video, runs }, null, 2), { mode: 0o600 });
      console.log(`${mode}: ${(result.elapsedMs / 1000).toFixed(2)} s, ${(row.throughput).toFixed(3)} media s/s, ${rows.length} vectors, ${row.frameCount} frames, min cosine ${minimumCosine}`);
    } finally {
      clearRuntimeEmbeddingOverride(config.activeProfile);
      await backend.stop();
    }
  }
  const record = JSON.parse(fs.readFileSync(output, "utf8")) as Record<string, unknown>;
  record.status = runs.every(run => run.qualityPassed) ? "completed" : "quality_failed";
  fs.writeFileSync(output, JSON.stringify(record, null, 2), { mode: 0o600 });
  if (record.status === "quality_failed") process.exitCode = 1;
} finally {
  fs.rmSync(scratch, { recursive: true, force: true });
}
