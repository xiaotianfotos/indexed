export type SourceSite = "youtube" | "bilibili";
export { APPLE_EXECUTION_MODES, readAppleExecutionMode, appleExecutionModeIssue, requireAppleExecutionMode } from "./apple-execution.js";
export type { AppleExecutionMode } from "./apple-execution.js";
export type MemoryModality = "visual" | "transcript";
export type StorageProvider = "local" | "aliyun";
export type ExtensionBackend = "local" | "cloud";

export const APPLE_EMBEDDING_MODEL_PREFIX = "wemm-embedding-2b-apple";
export const APPLE_EMBEDDING_DEFAULT_DIMENSION = 2048;

/** Apple-native WeMM exposes one fixed model ID for each output dimension. */
export function appleEmbeddingModel(dimension: unknown = APPLE_EMBEDDING_DEFAULT_DIMENSION): string {
  const selected = Number(dimension);
  const normalized = Number.isFinite(selected) && selected > 0
    ? Math.floor(selected)
    : APPLE_EMBEDDING_DEFAULT_DIMENSION;
  return `${APPLE_EMBEDDING_MODEL_PREFIX}-${normalized}`;
}

export const PRODUCT_DEFAULTS = {
  server: { host: "127.0.0.1", port: 18767 },
  vectorDimension: 4096,
  ossRegion: "cn-hangzhou",
  capture: {
    segmentSeconds: 10,
    framesPerSecond: 4,
    maxWidth: 1280,
    maxHeight: 720,
    videoBitsPerSecond: 1_600_000,
  },
  queue: {
    concurrency: 4,
    maxAttempts: 3,
    maxRepairAttempts: 3,
  },
} as const;

export const SEGMENT_SCHEMAS = {
  video: "native-video-10s-v1",
  videoFrames: "native-video-frames-10s-v2",
  visual: "visual-10s-v2",
  transcript: "transcript-10s-v3",
  transcriptLegacy: "transcript-10s-v2",
} as const;

export const OSS_VECTOR_PRICING = {
  storageCnyPerGibMonth: 0.35,
  queryCnyPerTib: 0.012,
  getCnyPer10k: 0.01,
  freeWriteGibPerMonth: 20,
} as const;

export interface ModelProfile {
  id: string;
  label: string;
  embedding: {
    baseUrl: string;
    model: string;
    dimension: number;
    inputStyle: "auto" | "qwen" | "wemm" | string;
  };
  spaceId: string;
  storage: {
    provider: StorageProvider;
    path?: string;
    region: string;
    accountId: string;
    bucket: string;
    visualIndex: string;
    transcriptIndex: string;
    documentIndex?: string;
  };
}

export interface VideoIdentity {
  sourceSite: SourceSite;
  videoId: string;
}

export interface VideoMemory extends VideoIdentity {
  title: string;
  sourceUrl: string;
  channel?: string;
  note?: string;
  tags?: string[];
  visualCount: number;
  transcriptCount: number;
  updatedAt?: string;
}

export interface SearchRequest {
  query: string;
  limit?: number;
  videoId?: string;
  modalities?: MemoryModality[];
}

export interface SearchHit extends VideoIdentity {
  modality: MemoryModality;
  score: number;
  startSeconds: number;
  endSeconds: number;
  sourceUrl: string;
  title: string;
  transcript?: string;
  previewUrl?: string;
  embeddingModel: string;
  embeddingSpace: string;
}

export interface SearchResponse {
  query: string;
  embeddingSpace: string;
  visual: SearchHit[];
  transcript: SearchHit[];
}
