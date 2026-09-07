import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";
import { AppleEmbeddingBackend, appleEmbeddingOptionsFromProfile } from "@indexed/apple-embedding-backend";
import { loadConfig, setRuntimeEmbeddingOverride, clearRuntimeEmbeddingOverride } from "@indexed/config";
import { scanLibrary } from "@indexed/core";

// Fast full-workload diagnosis during optimization. Uses the same Core scan
// flags and elapsedMs as IngestTestController, but has no A/B/D comparison or
// preceding mode runs. Final acceptance still uses the real /api/ingest-test.
const { values } = parseArgs({ options: Object.fromEntries(["video", "package", "binary", "source-config", "mode", "output"].map(key => [key, { type: "string" as const }])) });
for (const key of ["video", "package", "binary", "source-config", "mode", "output"]) assert(values[key], `Missing --${key}`);
assert(values.mode === "c" || values.mode === "d");
const output = path.resolve(values.output!);
assert(!fs.existsSync(output), "Refusing to overwrite a diagnosis");
const source = JSON.parse(fs.readFileSync(values["source-config"]!, "utf8"));
const original = source.profiles[source.activeProfile];
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-mode-diagnostic-"));
process.env.INDEXED_HOME = path.join(scratch, "home");
process.env.INDEXED_CONFIG = path.join(scratch, "config.json");
process.env.INDEXED_DATA_DIR = path.join(scratch, "data");
process.env.INDEXED_STATE_DIR = path.join(scratch, "state");
const config = loadConfig(), profile = config.profiles[config.activeProfile];
profile.video = structuredClone(original.video);
profile.storage = { ...profile.storage, provider: "local", path: path.join(scratch, "vectors") };
profile.embedding = { ...profile.embedding, provider: "apple-native", dimension: 2048, model: "wemm-embedding-2b-apple-2048", inputStyle: "wemm",
  native: { ...profile.embedding.native, binary: path.resolve(values.binary!), modelPackage: path.resolve(values.package!),
    executionMode: values.mode, coreMLCache: path.join(scratch, "coreml"), autoRestart: false } };
for (const key of ["sequenceLength", "mlpFraction", "mlpVariant", "mlpMaxLayers", "videoDownProjection", "videoPipeline",
  "recurrenceBlockSize", "recurrenceLayerSlots", "recurrenceQueryScale", "recurrenceMaxTokens", "recurrenceIODtype", "recurrenceVerifyReference"]) {
  if (original.embedding?.native?.privateANE?.[key] !== undefined) profile.embedding.native.privateANE[key] = structuredClone(original.embedding.native.privateANE[key]);
}
const library = path.join(scratch, "input");
fs.mkdirSync(library);
fs.copyFileSync(values.video!, path.join(library, "input.mp4"), fs.constants.COPYFILE_FICLONE);
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const binarySHA256 = hash(profile.embedding.native.binary), inputSHA256 = hash(path.join(library, "input.mp4"));
const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile));
try {
  const started = performance.now();
  setRuntimeEmbeddingOverride(config.activeProfile, await backend.start());
  const loadMs = performance.now() - started, before = await backend.health();
  let completed = -1;
  const sample = await scanLibrary(library, { config, dryRun: true, benchmark: true, recordPerformance: false, limit: 1,
    onProgress: job => { if (job.currentUnitsDone !== completed) { completed = job.currentUnitsDone || 0; console.log(`${values.mode}: ${completed}/${job.currentUnitsTotal || 0}`); } } });
  const after = await backend.health();
  assert(sample.ok && sample.indexed === 1 && sample.failed === 0, JSON.stringify(sample.errors));
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify({ schema: 1, protocol: "single-mode-core-diagnostic-v1", mode: values.mode,
    binarySHA256, inputSHA256, video: profile.video, kernel: profile.embedding.native.privateANE,
    loadMs, sample, before, after }, null, 2), { mode: 0o600, flag: "wx" });
  console.log(JSON.stringify({ elapsedMs: sample.elapsedMs, timings: sample.modelTimings }));
} catch (error) {
  console.error(backend.status().stderrTail.join("\n")); throw error;
} finally {
  clearRuntimeEmbeddingOverride(config.activeProfile);
  await backend.stop();
  fs.rmSync(scratch, { recursive: true, force: true });
}
