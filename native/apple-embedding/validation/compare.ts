import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { vectorAgreement, type Reference, type CaseResult } from "./reference.js";

function at(value: unknown, keys: string[]): number {
  let current = value;
  for (const key of keys) {
    if (!current || typeof current !== "object" || Array.isArray(current)) return 0;
    current = (current as Record<string, unknown>)[key];
  }
  return typeof current === "number" && Number.isFinite(current) ? current : 0;
}
export function routes(result: CaseResult) {
  const delta = (keys: string[]) => at(result.after, keys) - at(result.before, keys);
  return {
    mlp: delta(["private_ane", "native_profile", "mlp", "operations"]),
    recurrence: delta(["private_ane", "recurrence", "ane_evaluations"]),
  };
}

function percentile(values: number[], fraction: number): number {
  if (!values.length || values.some(value => !Number.isFinite(value) || value <= 0)) throw new Error("Missing or invalid timing samples");
  return [...values].sort((a, b) => a - b)[Math.ceil(values.length * fraction) - 1]!;
}

export function validateReference(reference: Reference): string[] {
  const failures: string[] = [];
  const spec = reference.specification;
  const deviations = reference.runtimeInfo.known_deviations;
  if (Array.isArray(deviations) && deviations.length) failures.push(`Unresolved implementation deviations: ${deviations.join("; ")}`);
  if (!/^[abcd]$/.test(reference.mode) || !["python", "swift"].includes(reference.runtime)) failures.push("Invalid runtime/mode");
  if (reference.cases.length !== spec.cases.length || !reference.modelIdentity || !reference.suiteSHA256 || !reference.packageFingerprint) failures.push("Missing reference identity or cases");
  if (!(reference.peakRSSBytes > 0) || !(reference.startupMS > 0)) failures.push("Missing process measurements");
  for (const fixture of spec.cases) {
    const entry = reference.cases.find(value => value.id === fixture.id);
    if (!entry || !entry.inputSHA256 || entry.samples.length !== spec.repeats) { failures.push(`${fixture.id}: incomplete samples`); continue; }
    for (const sample of entry.samples) {
      try { vectorAgreement(sample.vector, sample.vector); } catch { failures.push(`${entry.id}: invalid vector`); }
      if (sample.vector.length !== spec.dimension || !sample.embeddingSpace || !Number.isInteger(sample.promptTokens) || sample.promptTokens <= 0 || sample.promptTokens > 8192) failures.push(`${entry.id}: invalid output contract`);
      if (!Number.isFinite(sample.wallMS) || sample.wallMS <= 0) failures.push(`${entry.id}: invalid latency`);
    }
    const actual = routes(entry);
    const mlpCases = spec.gates.requiredPrivateMLP as string[];
    const recurrenceCases = spec.gates.requiredPrivateRecurrence as string[];
    if (["c", "d"].includes(reference.mode) && mlpCases.includes(entry.id) && actual.mlp <= 0) failures.push(`${entry.id}: no measured private ANE MLP execution`);
    if (reference.mode === "d" && recurrenceCases.includes(entry.id) && actual.recurrence <= 0) failures.push(`${entry.id}: no measured private ANE recurrence execution`);
  }
  if (reference.concurrentVideo.samples.length !== 2 || !(reference.concurrentVideo.wallMS > 0)) failures.push("Missing two-request video pipeline evidence");
  if (reference.runtime === "swift" && ["c", "d"].includes(reference.mode)) {
    const { before, after } = reference.concurrentVideo;
    const metric = (value: unknown, key: string) => at(value, ["scheduling", key]);
    if (metric(after, "pipeline_depth") !== 2 || metric(after, "peak_in_flight") !== 2
      || metric(after, "peak_vision_active") !== 1 || metric(after, "peak_language_active") !== 1
      || metric(after, "vision_language_overlap_events") <= metric(before, "vision_language_overlap_events")
      || metric(after, "vision_language_overlap_ms") <= metric(before, "vision_language_overlap_ms")) {
      failures.push("Missing measured two-request vision/language overlap with single-owner stages");
    }
    if (reference.mode === "d" && metric(after, "peak_ane_lane_active") !== 1) failures.push("Missing D shared ANE lane evidence");
  }
  for (const sample of reference.concurrentVideo.samples) {
    try { vectorAgreement(sample.vector, sample.vector); } catch { failures.push("Invalid concurrent video vector"); }
    const video = reference.cases.find(value => value.id === "video-10s");
    if (video && sample.promptTokens !== video.samples[0]?.promptTokens) failures.push("Concurrent video workload differs from the fixed video case");
  }
  return failures;
}

export function compareReferences(baseline: Reference, candidate: Reference) {
  const failures = [...validateReference(baseline).map(value => `baseline: ${value}`), ...validateReference(candidate).map(value => `candidate: ${value}`)];
  for (const key of ["mode", "suiteSHA256", "modelIdentity", "packageFingerprint"] as const) {
    if (baseline[key] !== candidate[key]) failures.push(`Mismatched ${key}`);
  }
  if (JSON.stringify(baseline.specification) !== JSON.stringify(candidate.specification)) failures.push("Specification changed after freezing");
  const gates = baseline.specification.gates;
  const summaries = [];
  for (const original of baseline.cases) {
    const replacement = candidate.cases.find(value => value.id === original.id);
    if (!replacement || replacement.samples.length !== original.samples.length) { failures.push(`${original.id}: sample count mismatch`); continue; }
    if (original.inputSHA256 !== replacement.inputSHA256) failures.push(`${original.id}: input changed`);
    const agreement = [];
    for (let index = 0; index < original.samples.length; index++) {
      const left = original.samples[index]!, right = replacement.samples[index]!;
      if (left.promptTokens !== right.promptTokens) failures.push(`${original.id}: token count changed`);
      if (left.embeddingSpace !== right.embeddingSpace) failures.push(`${original.id}: embedding space changed`);
      try {
        const result = vectorAgreement(left.vector, right.vector);
        agreement.push(result);
        if (result.cosine < Number(gates.minimumCosine) || result.relativeL2 > Number(gates.maximumRelativeL2) || result.maximumAbsoluteError > Number(gates.maximumAbsoluteError)) failures.push(`${original.id}: numerical tolerance exceeded`);
      } catch { failures.push(`${original.id}: invalid vectors`); }
    }
    const originalRoutes = routes(original), replacementRoutes = routes(replacement);
    if (JSON.stringify(originalRoutes) !== JSON.stringify(replacementRoutes)) failures.push(`${original.id}: ANE work count changed`);
    let medianRatio = Infinity, p95Ratio = Infinity;
    try {
      medianRatio = percentile(replacement.samples.map(value => value.wallMS), 0.5) / percentile(original.samples.map(value => value.wallMS), 0.5);
      p95Ratio = percentile(replacement.samples.map(value => value.wallMS), 0.95) / percentile(original.samples.map(value => value.wallMS), 0.95);
    } catch { failures.push(`${original.id}: invalid timing samples`); }
    if (medianRatio > Number(gates.maximumMedianLatencyRatio) || p95Ratio > Number(gates.maximumP95LatencyRatio)) failures.push(`${original.id}: latency tolerance exceeded`);
    summaries.push({ id: original.id, minimumCosine: Math.min(...agreement.map(value => value.cosine)), maximumRelativeL2: Math.max(...agreement.map(value => value.relativeL2)), medianRatio, p95Ratio, originalRoutes, replacementRoutes });
  }
  const memoryRatio = candidate.peakRSSBytes / baseline.peakRSSBytes;
  if (memoryRatio > Number(gates.maximumPeakRSSRatio)) failures.push("Peak RSS tolerance exceeded");
  const pipelineRatio = candidate.concurrentVideo.wallMS / baseline.concurrentVideo.wallMS;
  if (pipelineRatio > Number(gates.maximumP95LatencyRatio)) failures.push("Concurrent video latency tolerance exceeded");
  for (let index = 0; index < baseline.concurrentVideo.samples.length; index++) {
    const expected = baseline.concurrentVideo.samples[index]!, actual = candidate.concurrentVideo.samples[index];
    if (!actual) { failures.push("Missing concurrent video sample"); continue; }
    if (actual.promptTokens !== expected.promptTokens || actual.embeddingSpace !== expected.embeddingSpace) failures.push("Concurrent video token count or space changed");
    try {
      const agreement = vectorAgreement(expected.vector, actual.vector);
      if (agreement.cosine < Number(gates.minimumCosine) || agreement.relativeL2 > Number(gates.maximumRelativeL2) || agreement.maximumAbsoluteError > Number(gates.maximumAbsoluteError)) failures.push("Concurrent video numerical tolerance exceeded");
    } catch { failures.push("Invalid concurrent video embedding"); }
  }
  const pipelineRoutes = (reference: Reference) => routes({ id: "pipeline", inputSHA256: "", ...reference.concurrentVideo });
  if (JSON.stringify(pipelineRoutes(baseline)) !== JSON.stringify(pipelineRoutes(candidate))) failures.push("Concurrent video ANE work count changed");
  return { passed: failures.length === 0, failures, memoryRatio, pipelineRatio, cases: summaries };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [baselinePath, candidatePath] = process.argv.slice(2);
  if (!baselinePath) throw new Error("Usage: compare.ts BASELINE.json [CANDIDATE.json]");
  const baseline = JSON.parse(fs.readFileSync(baselinePath, "utf8")) as Reference;
  const result = candidatePath
    ? compareReferences(baseline, JSON.parse(fs.readFileSync(candidatePath, "utf8")) as Reference)
    : { failures: validateReference(baseline), routes: baseline.cases.map(value => ({ id: value.id, ...routes(value) })) };
  console.log(JSON.stringify(result, null, 2));
  process.exitCode = result.failures.length ? 1 : 0;
}
