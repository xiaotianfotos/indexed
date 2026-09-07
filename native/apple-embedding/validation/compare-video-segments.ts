import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

// A strict scheduling/layout regression check. This does not replace the full
// backend performance protocol or the frozen cross-runtime numerical gates.
interface Health {
  private_ane: { native_profile: { mlp: { operations: number } }; failed_layers: Record<string, unknown>;
    recurrence?: { ane_evaluations: number } | null };
  uptime_seconds: number;
  request_latency_ms_last: number;
}
interface Sample {
  inputSHA256: string; frames: number; tokens: number; embeddingSpace: string; vector: number[];
}
interface Capture {
  mode: string; video: unknown; duration: number; binarySHA256: string;
  cases: Array<Sample & { segment: unknown; before: Health; after: Health }>;
  concurrent: { samples: Sample[]; before: Health; after: Health };
}
const [baselinePath, candidatePath, outputPath] = process.argv.slice(2);
assert(baselinePath && candidatePath && outputPath, "Usage: compare-video-segments.ts BASELINE CANDIDATE OUTPUT");
assert(!fs.existsSync(outputPath), "Refusing to overwrite a comparison");
const baseline = JSON.parse(fs.readFileSync(baselinePath, "utf8")) as Capture;
const candidate = JSON.parse(fs.readFileSync(candidatePath, "utf8")) as Capture;
for (const key of ["mode", "video", "duration"] as const) assert.deepEqual(candidate[key], baseline[key], `Changed ${key}`);
assert.equal(baseline.cases.length, 3); assert.equal(candidate.cases.length, 3);
assert.equal(baseline.concurrent.samples.length, 2); assert.equal(candidate.concurrent.samples.length, 2);
const samples = (capture: Capture) => [...capture.cases, ...capture.concurrent.samples];
let valuesChecked = 0;
for (let index = 0; index < samples(baseline).length; index++) {
  const before = samples(baseline)[index]!, after = samples(candidate)[index]!;
  for (const key of ["inputSHA256", "frames", "tokens", "embeddingSpace"] as const) assert.deepEqual(after[key], before[key], `${index}: changed ${key}`);
  assert.equal(after.vector.length, 2048); assert(after.vector.every(Number.isFinite));
  assert.deepEqual(after.vector, before.vector, `${index}: embedding is not bit-identical`);
  valuesChecked += after.vector.length;
}
const operations = (capture: Capture) => [...capture.cases, capture.concurrent].map(sample => {
  assert.deepEqual(sample.after.private_ane.failed_layers, {});
  const count = sample.after.private_ane.native_profile.mlp.operations - sample.before.private_ane.native_profile.mlp.operations;
  assert(count > 0); return count;
});
assert.deepEqual(candidate.cases.map(sample => sample.segment), baseline.cases.map(sample => sample.segment));
assert.deepEqual(operations(candidate), operations(baseline), "Changed private ANE workload");
const recurrenceOperations = (capture: Capture) => [...capture.cases, capture.concurrent].map(sample => {
  const count = (sample.after.private_ane.recurrence?.ane_evaluations || 0) - (sample.before.private_ane.recurrence?.ane_evaluations || 0);
  if (capture.mode === "d") assert(count > 0, "D did not execute ANE recurrence");
  return count;
});
assert.deepEqual(recurrenceOperations(candidate), recurrenceOperations(baseline), "Changed private ANE recurrence workload");
const diagnosticTiming = (capture: Capture) => ({ soloMS: capture.cases.map(sample => sample.after.request_latency_ms_last),
  concurrentObservedMS: (capture.concurrent.after.uptime_seconds - capture.concurrent.before.uptime_seconds) * 1000 });
const result = { schema: 1, status: "passed", valuesChecked, mlpOperations: operations(candidate),
  recurrenceOperations: recurrenceOperations(candidate),
  baselineBinarySHA256: baseline.binarySHA256, candidateBinarySHA256: candidate.binarySHA256,
  diagnosticTiming: { baseline: diagnosticTiming(baseline), candidate: diagnosticTiming(candidate) } };
fs.mkdirSync(path.dirname(path.resolve(outputPath)), { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), { mode: 0o600, flag: "wx" });
console.log(JSON.stringify(result));
