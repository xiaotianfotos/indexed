import assert from "node:assert/strict";
import test from "node:test";
import { vectorAgreement, type Reference } from "../native/apple-embedding/validation/reference.js";
import { compareReferences, validateReference } from "../native/apple-embedding/validation/compare.js";

function fixture(): Reference {
  const samples = Array.from({ length: 5 }, () => ({ vector: [1, 0], promptTokens: 2045, embeddingSpace: "fixture-space", wallMS: 10, timings: {} }));
  return {
    version: 1, runtime: "python", mode: "d", suiteSHA256: "suite", modelIdentity: "model", packageFingerprint: "fingerprint",
    sourceHashes: {}, createdAt: "fixture", environment: {}, startupMS: 1, peakRSSBytes: 100, runtimeInfo: {},
    specification: { version: 1, name: "fixture", dimension: 2, repeats: 5, warmupPerCase: 1, kernel: {}, gates: {
      minimumCosine: 0.999, maximumRelativeL2: 0.05, maximumAbsoluteError: 0.02, maximumMedianLatencyRatio: 1.15, maximumP95LatencyRatio: 1.25, maximumPeakRSSRatio: 1.15,
      requiredPrivateMLP: ["video"], requiredPrivateRecurrence: ["video"],
    }, cases: [{ id: "video" }] },
    cases: [{ id: "video", inputSHA256: "input", samples, before: {}, after: { private_ane: { native_profile: { mlp: { operations: 120 } }, recurrence: { ane_evaluations: 160 } } } }],
    concurrentVideo: { wallMS: 20, samples: samples.slice(0, 2), before: {}, after: {} },
  };
}

test("native parity gate rejects incompatible identity, changed workload, GPU substitution and regressions", () => {
  const baseline = fixture();
  assert.equal(compareReferences(baseline, structuredClone(baseline)).passed, true);
  const mutations: Array<(value: Reference) => void> = [
    value => { value.modelIdentity = "other"; },
    value => { value.mode = "b"; },
    value => { value.cases[0]!.inputSHA256 = "less frames"; },
    value => { value.cases[0]!.samples[0]!.promptTokens--; },
    value => { value.cases[0]!.samples[0]!.embeddingSpace = "other"; },
    value => { value.cases[0]!.after = {}; },
    value => { value.cases[0]!.samples[0]!.vector = [0, 1]; },
    value => { value.cases[0]!.samples[0]!.wallMS = 100; },
    value => { value.peakRSSBytes *= 2; },
    value => { value.concurrentVideo.wallMS *= 2; },
    value => { value.concurrentVideo.samples[0]!.promptTokens--; },
    value => { value.concurrentVideo.samples[0]!.vector = [0, 1]; },
    value => { value.runtimeInfo.known_deviations = ["video_pipeline=1; reference requires 2"]; },
  ];
  for (const mutate of mutations) {
    const candidate = structuredClone(baseline); mutate(candidate);
    assert.equal(compareReferences(baseline, candidate).passed, false);
  }
  const empty = fixture(); empty.cases[0]!.samples = [];
  assert.match(validateReference(empty).join(" "), /incomplete samples/);
});

test("native numerical agreement rejects empty, nonfinite and zero vectors", () => {
  assert.equal(vectorAgreement([1, 0], [1, 0]).cosine, 1);
  for (const invalid of [[], [NaN, 1], [0, 0], [1]]) assert.throws(() => vectorAgreement([1, 0], invalid));
});


test("Swift C/D parity requires actual stage overlap, bounded admission and a D ANE lane", () => {
  const baseline = fixture(), candidate = fixture();
  candidate.runtime = "swift";
  candidate.concurrentVideo.after.scheduling = {
    pipeline_depth: 2, peak_in_flight: 2, peak_vision_active: 1, peak_language_active: 1,
    vision_language_overlap_events: 1, vision_language_overlap_ms: 5, peak_ane_lane_active: 1,
  };
  assert.equal(compareReferences(baseline, candidate).passed, true);
  for (const key of ["pipeline_depth", "peak_in_flight", "peak_vision_active", "peak_language_active",
    "vision_language_overlap_events", "vision_language_overlap_ms", "peak_ane_lane_active"]) {
    const altered = structuredClone(candidate);
    (altered.concurrentVideo.after.scheduling as Record<string, number>)[key] = 0;
    assert.equal(compareReferences(baseline, altered).passed, false, key);
  }
  const noNewOverlap = structuredClone(candidate);
  noNewOverlap.concurrentVideo.before = structuredClone(noNewOverlap.concurrentVideo.after);
  assert.equal(compareReferences(baseline, noNewOverlap).passed, false);
});
