import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";
import { setTimeout as delay } from "node:timers/promises";
import { loadConfig, writeConfig } from "@indexed/config";
import { startServer } from "@indexed/server";
import type { IngestTestReport } from "../../../apps/server/src/ingest-test.js";

// Uses the existing backend test mode and its statistics without reimplementing
// the benchmark. Only video/kernel settings are read from --source-config;
// credentials, libraries, storage and live state are never copied or modified.
const { values } = parseArgs({ options: {
  video: { type: "string" }, package: { type: "string" }, binary: { type: "string" },
  "source-config": { type: "string" }, output: { type: "string" }, repeats: { type: "string", default: "1" },
} });
for (const key of ["video", "package", "binary", "source-config", "output"] as const) assert(values[key], `Missing --${key}`);
const output = path.resolve(values.output!);
assert(!fs.existsSync(output), "Use a new output path for each benchmark");
const binarySHA256 = createHash("sha256").update(fs.readFileSync(values.binary!)).digest("hex");
const source = JSON.parse(fs.readFileSync(values["source-config"]!, "utf8"));
const original = source.profiles[source.activeProfile];
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-backend-benchmark-"));
const library = path.join(scratch, "input");
fs.mkdirSync(library);
fs.copyFileSync(path.resolve(values.video!), path.join(library, "input.mp4"), fs.constants.COPYFILE_FICLONE);
process.env.INDEXED_HOME = path.join(scratch, "home");
process.env.INDEXED_CONFIG = path.join(scratch, "config.json");
process.env.INDEXED_DATA_DIR = path.join(scratch, "data");
process.env.INDEXED_STATE_DIR = path.join(scratch, "state");
const config = loadConfig();
config.library = { ...config.library, roots: [], autoScan: false, kinds: ["video"], rootKinds: {} };
config.profiles.default.video = structuredClone(original.video);
config.profiles.default.storage = { ...config.profiles.default.storage, provider: "local", path: path.join(scratch, "vectors") };
config.profiles.default.embedding = { ...config.profiles.default.embedding, provider: "apple-native", dimension: 2048, model: "wemm-embedding-2b-apple-2048",
  inputStyle: "wemm", native: { ...config.profiles.default.embedding.native, binary: path.resolve(values.binary!),
    modelPackage: path.resolve(values.package!), coreMLCache: path.join(scratch, "coreml"), executionMode: "a", autoRestart: false } };
const kernel = config.profiles.default.embedding.native.privateANE;
for (const key of ["sequenceLength", "mlpFraction", "mlpVariant", "mlpMaxLayers", "videoDownProjection", "videoPipeline",
  "recurrenceBlockSize", "recurrenceLayerSlots", "recurrenceQueryScale", "recurrenceMaxTokens", "recurrenceIODtype", "recurrenceVerifyReference"]) {
  if (original.embedding?.native?.privateANE?.[key] !== undefined) kernel[key] = structuredClone(original.embedding.native.privateANE[key]);
}
writeConfig(config);
const configHash = createHash("sha256").update(fs.readFileSync(process.env.INDEXED_CONFIG)).digest("hex");
const digest = createHash("sha256");
for await (const chunk of fs.createReadStream(path.join(library, "input.mp4"))) digest.update(chunk as Buffer);
let server: Awaited<ReturnType<typeof startServer>> | undefined;
try {
  server = await startServer({ host: "127.0.0.1", port: 0 });
  await server.embeddingReady;
  const start = await fetch(`${server.url}api/ingest-test/start`, { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ path: library, limit: 1, repeats: Number(values.repeats) }) });
  assert.equal(start.status, 202, await start.text());
  let previous = "";
  for (;;) {
    const response = await fetch(`${server.url}api/ingest-test`);
    assert(response.ok);
    const report = await response.json() as IngestTestReport;
    const progress = report.progress;
    const state = `${report.mode} ${report.phase} ${progress?.currentUnitsDone || 0}/${progress?.currentUnitsTotal || 0}`;
    if (state !== previous) { console.log(state); previous = state; }
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(output, JSON.stringify({ schema: 1, inputSHA256: digest.copy().digest("hex"),
      binarySHA256,
      video: config.profiles.default.video, kernel, report }, null, 2), { mode: 0o600 });
    if (!["running", "restoring"].includes(report.state)) {
      assert.equal(report.state, "done", JSON.stringify(report.rows.map(row => ({ mode: row.mode, error: row.error }))));
      assert.equal(report.protocol, "no-warmup-cooldown-v1");
      assert(report.rows.every(row => row.state === "done" && row.samples.every(sample => sample.processedMediaSeconds && sample.indexed === 1 && sample.failed === 0)));
      assert.equal(createHash("sha256").update(fs.readFileSync(process.env.INDEXED_CONFIG)).digest("hex"), configHash, "Benchmark changed configuration");
      console.log(JSON.stringify(report.rows.map(row => ({ mode: row.mode, elapsedMs: row.elapsedMs, modelMs: row.modelMs,
        visionMs: row.visionMs, languageMs: row.languageMs, loadMs: row.loadMs, samples: row.samples.map(sample => ({ vectors: sample.vectors, mediaSeconds: sample.processedMediaSeconds })) })), null, 2));
      break;
    }
    await delay(1000);
  }
} finally {
  await server?.close();
  fs.rmSync(scratch, { recursive: true, force: true });
}
