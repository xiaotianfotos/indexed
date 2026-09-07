import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  activeProfile,
  directConfig,
  ingestPerformanceHistoryPath,
  loadConfig,
} from "@indexed/config";
import { DOCUMENT_CHUNK_VERSION } from "./document-assets.js";

export type PerformanceAssetKind = "video" | "image" | "document";
export type KindCounters = Record<PerformanceAssetKind, number>;

const KINDS: PerformanceAssetKind[] = ["video", "image", "document"];
const HISTORY_VERSION = 1;
const MAX_HISTORY_SAMPLES = 240;
const MAX_WORKLOAD_SAMPLES = 40;

export interface IngestPerformanceSample {
  sampledAt: number;
  providerKey: string;
  workloadKey: string;
  supplierLabel: string;
  storageProvider: "local" | "aliyun";
  embeddingModel: string;
  embeddingDimension: number;
  embeddingInputStyle: string;
  documentChunkVersion: string;
  executionMode: string;
  elapsedMs: number;
  concurrency: number;
  indexedByKind: KindCounters;
  workMsByKind: KindCounters;
  failed: number;
  vectors: number;
}

interface IngestPerformanceFile {
  version: 1;
  samples: IngestPerformanceSample[];
}

export interface IngestPerformanceObservation {
  config?: ReturnType<typeof loadConfig>;
  historyPath?: string;
  elapsedMs: number;
  concurrency: number;
  indexedByKind: KindCounters;
  workMsByKind: KindCounters;
  failed: number;
  vectors: number;
  sampledAt?: number;
}

export interface KindPerformanceRate {
  millisecondsPerAsset: number | null;
  samples: number;
  source: "exact" | "provider" | "none";
}

export interface IngestPerformanceEstimate {
  supplierLabel: string;
  storageProvider: "local" | "aliyun";
  embeddingModel: string;
  embeddingDimension: number;
  source: "exact" | "provider" | "none";
  sampleCount: number;
  lastSampleAt: number;
  pendingByKind: KindCounters;
  pendingAssets: number;
  estimatedMs: number | null;
  lowerEstimatedMs: number | null;
  upperEstimatedMs: number | null;
  unavailableKinds: PerformanceAssetKind[];
  ratesByKind: Record<PerformanceAssetKind, KindPerformanceRate>;
  historyPath: string;
}

interface PerformanceIdentity {
  providerKey: string;
  workloadKey: string;
  supplierLabel: string;
  storageProvider: "local" | "aliyun";
  embeddingModel: string;
  embeddingDimension: number;
  embeddingInputStyle: string;
  executionMode: string;
  documentChunkVersion: string;
}

function counters(value: Partial<KindCounters> | undefined): KindCounters {
  return {
    video: Math.max(0, Number(value?.video || 0)),
    image: Math.max(0, Number(value?.image || 0)),
    document: Math.max(0, Number(value?.document || 0)),
  };
}

function digest(value: unknown): string {
  return crypto.createHash("sha256").update(JSON.stringify(value)).digest("hex").slice(0, 24);
}

function endpointLabel(raw: string): string {
  const value = String(raw || "").trim();
  if (!value) return "未配置 Embedding";
  try {
    return new URL(value).host || value;
  } catch {
    return value.replace(/^https?:\/\//i, "").split("/")[0] || value;
  }
}

function identity(config: ReturnType<typeof loadConfig>): PerformanceIdentity {
  const profile = activeProfile(config);
  const direct = directConfig(profile);
  const executionMode = direct.embeddingProvider === "apple-native"
    ? String(profile.embedding?.native?.executionMode || "b").toLowerCase()
    : "";
  const supplierLabel = executionMode
    ? `Apple 原生 · ${executionMode.toUpperCase()}`
    : endpointLabel(direct.embeddingBaseUrl);
  const storageProvider = direct.storageProvider === "aliyun" ? "aliyun" : "local";
  const providerKey = digest({ storageProvider, supplierLabel: supplierLabel.toLowerCase(), executionMode });
  const video = direct.video || {};
  const workloadKey = digest({
    providerKey,
    model: direct.embeddingModel,
    dimension: direct.embeddingDimension,
    inputStyle: direct.embeddingInputStyle,
    executionMode,
    documentChunkVersion: DOCUMENT_CHUNK_VERSION,
    video: {
      chunkSeconds: Number(video.chunkSeconds || 0),
      maxChunkSeconds: Number(video.maxChunkSeconds || 0),
      fps: Number(video.fps || 0),
      width: Number(video.width || 0),
    },
  });
  return {
    providerKey,
    workloadKey,
    supplierLabel,
    storageProvider,
    embeddingModel: direct.embeddingModel,
    embeddingDimension: direct.embeddingDimension,
    embeddingInputStyle: direct.embeddingInputStyle,
    executionMode,
    documentChunkVersion: DOCUMENT_CHUNK_VERSION,
  };
}

function emptyHistory(): IngestPerformanceFile {
  return { version: HISTORY_VERSION, samples: [] };
}

export function readIngestPerformanceHistory(target = ingestPerformanceHistoryPath()): IngestPerformanceFile {
  try {
    const parsed = JSON.parse(fs.readFileSync(target, "utf8")) as Partial<IngestPerformanceFile>;
    if (parsed.version !== HISTORY_VERSION || !Array.isArray(parsed.samples)) return emptyHistory();
    return {
      version: HISTORY_VERSION,
      samples: parsed.samples.filter((sample): sample is IngestPerformanceSample => (
        Boolean(sample)
        && typeof sample.sampledAt === "number"
        && typeof sample.providerKey === "string"
        && typeof sample.workloadKey === "string"
      )),
    };
  } catch {
    return emptyHistory();
  }
}

function writeHistory(history: IngestPerformanceFile, target: string): void {
  fs.mkdirSync(path.dirname(target), { recursive: true, mode: 0o700 });
  const temporary = `${target}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(history, null, 2)}\n`, { mode: 0o600 });
  fs.renameSync(temporary, target);
  fs.chmodSync(target, 0o600);
}

/**
 * Persist one privacy-safe scan sample. Paths, filenames, vector data and credentials
 * are deliberately absent, and the storage path is excluded so a new database can
 * reuse measurements from an earlier database with the same supplier and workload.
 */
export function recordIngestPerformance(observation: IngestPerformanceObservation): IngestPerformanceSample | null {
  const indexedByKind = counters(observation.indexedByKind);
  if (!KINDS.some((kind) => indexedByKind[kind] > 0)) return null;
  const config = observation.config || loadConfig();
  const keys = identity(config);
  const target = observation.historyPath || ingestPerformanceHistoryPath();
  const sample: IngestPerformanceSample = {
    sampledAt: Math.max(1, Math.floor(observation.sampledAt || Date.now())),
    ...keys,
    elapsedMs: Math.max(1, Math.round(observation.elapsedMs)),
    concurrency: Math.max(1, Math.floor(observation.concurrency)),
    indexedByKind,
    workMsByKind: counters(observation.workMsByKind),
    failed: Math.max(0, Math.floor(observation.failed)),
    vectors: Math.max(0, Math.floor(observation.vectors)),
  };
  const history = readIngestPerformanceHistory(target);
  const sameWorkload = history.samples.filter((item) => item.workloadKey === sample.workloadKey);
  const evict = sameWorkload.length >= MAX_WORKLOAD_SAMPLES ? sameWorkload[0] : undefined;
  const samples = history.samples.filter((item) => item !== evict);
  samples.push(sample);
  writeHistory({ version: HISTORY_VERSION, samples: samples.slice(-MAX_HISTORY_SAMPLES) }, target);
  return sample;
}

function median(values: number[]): number | null {
  const sorted = values.filter((value) => Number.isFinite(value) && value > 0).sort((left, right) => left - right);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2) return sorted[middle] ?? null;
  return ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}

function rates(samples: IngestPerformanceSample[], kind: PerformanceAssetKind): number[] {
  return samples.flatMap((sample) => {
    const count = Number(sample.indexedByKind?.[kind] || 0);
    const work = Number(sample.workMsByKind?.[kind] || 0);
    return count > 0 && work > 0 ? [work / count] : [];
  });
}

/** Estimate a fresh database from exact-workload history, then same-supplier history. */
export function estimateIngestPerformance(options: {
  config?: ReturnType<typeof loadConfig>;
  pendingByKind?: Partial<KindCounters>;
  historyPath?: string;
} = {}): IngestPerformanceEstimate {
  const config = options.config || loadConfig();
  const keys = identity(config);
  const target = options.historyPath || ingestPerformanceHistoryPath();
  const history = readIngestPerformanceHistory(target);
  const exact = history.samples.filter((sample) => sample.workloadKey === keys.workloadKey).slice(-20);
  const provider = history.samples.filter((sample) => sample.providerKey === keys.providerKey).slice(-40);
  const pendingByKind = counters(options.pendingByKind);
  const ratesByKind = {} as Record<PerformanceAssetKind, KindPerformanceRate>;
  const selectedSamples = new Set<IngestPerformanceSample>();
  let usedProviderFallback = false;
  for (const kind of KINDS) {
    const exactRates = rates(exact, kind);
    // Text chunking changes how many model calls one document costs. Old provider
    // samples remain useful for media but must not estimate a new document pipeline.
    const compatibleProvider = kind === "document"
      ? provider.filter((sample) => sample.documentChunkVersion === keys.documentChunkVersion)
      : provider;
    const providerRates = exactRates.length ? [] : rates(compatibleProvider, kind);
    const selected = exactRates.length ? exact : providerRates.length ? compatibleProvider : [];
    selected.forEach((sample) => {
      if (Number(sample.indexedByKind?.[kind] || 0) > 0) selectedSamples.add(sample);
    });
    if (!exactRates.length && providerRates.length) usedProviderFallback = true;
    ratesByKind[kind] = {
      millisecondsPerAsset: median(exactRates.length ? exactRates : providerRates),
      samples: exactRates.length || providerRates.length,
      source: exactRates.length ? "exact" : providerRates.length ? "provider" : "none",
    };
  }

  const pendingAssets = KINDS.reduce((sum, kind) => sum + pendingByKind[kind], 0);
  const unavailableKinds = KINDS.filter((kind) => pendingByKind[kind] > 0 && ratesByKind[kind].millisecondsPerAsset === null);
  let estimatedMs: number | null = null;
  if (selectedSamples.size && !unavailableKinds.length) {
    const work = KINDS.reduce((sum, kind) => (
      sum + pendingByKind[kind] * Number(ratesByKind[kind].millisecondsPerAsset || 0)
    ), 0);
    const longestTask = Math.max(0, ...KINDS
      .filter((kind) => pendingByKind[kind] > 0)
      .map((kind) => Number(ratesByKind[kind].millisecondsPerAsset || 0)));
    const concurrency = Math.max(1, Math.round(median([...selectedSamples].map((sample) => sample.concurrency)) || 1));
    estimatedMs = pendingAssets ? Math.ceil(Math.max(longestTask, work / concurrency)) : 0;
  }
  const sampleCount = selectedSamples.size;
  const uncertainty = sampleCount >= 5 ? [0.8, 1.25] : sampleCount >= 2 ? [0.7, 1.4] : [0.6, 1.7];
  const source = !sampleCount ? "none" : usedProviderFallback ? "provider" : "exact";
  return {
    ...keys,
    source,
    sampleCount,
    lastSampleAt: Math.max(0, ...[...selectedSamples].map((sample) => sample.sampledAt)),
    pendingByKind,
    pendingAssets,
    estimatedMs,
    lowerEstimatedMs: estimatedMs === null ? null : Math.floor(estimatedMs * (uncertainty[0] ?? 1)),
    upperEstimatedMs: estimatedMs === null ? null : Math.ceil(estimatedMs * (uncertainty[1] ?? 1)),
    unavailableKinds,
    ratesByKind,
    historyPath: target,
  };
}
