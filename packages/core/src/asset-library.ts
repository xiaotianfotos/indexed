import { withOperation, throwIfAborted, type OperationOptions } from "@indexed/clients/operation";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import {
  discoverModelInfo,
  embedDocument,
  embedImage,
  embedTextWithInstruction,
  embedVideoFrames,
  embedVisualQuery,
  embeddingCorpusInputVersion,
  metadataFilter,
  scoreFromDistance,
  uuid5Url,
} from "@indexed/clients/direct-cloud";
import {
  deleteLocalVectors,
  listLocalVectors,
  localVectorDatabasePath,
  putLocalVectors,
  queryLocalVectors,
  withLocalVectorSession,
  type LocalVectorSessionTimings,
} from "@indexed/clients/local-vectors";
import { activeProfile, directConfig, loadConfig, writeConfig } from "@indexed/config";
import {
  DEFAULT_MAX_MODEL_LEN,
  extractFrame,
  planChunks,
  probeMedia,
  withFramePacket,
} from "./asset-chunks.js";
import {
  estimateIngestPerformance,
  recordIngestPerformance,
  type KindCounters,
} from "./ingest-performance.js";
import {
  chunkDocumentText,
  documentFormatCapabilities,
  DOCUMENT_CHUNK_VERSION,
  extractDocumentText,
} from "./document-assets.js";

export const ASSET_RECORD_TYPE = "asset";

export const ASSET_EXTENSIONS = {
  video: new Set([".mp4", ".mov", ".mkv", ".webm", ".m4v"]),
  image: new Set([".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".avif"]),
  document: new Set([
    ".pdf", ".doc", ".docx",
    ".md", ".mdx", ".txt", ".json", ".jsonl", ".csv", ".srt", ".vtt",
    ".ts", ".tsx", ".js", ".jsx", ".py", ".html", ".css",
  ]),
};

/** Text files that usually describe a same-stem media file in a shot library. */
export const SIDECAR_EXTENSIONS = new Set([".txt", ".srt", ".vtt", ".md", ".json", ".csv"]);

/**
 * Only note-like text may be rewritten through the asset document route. Code
 * extensions are indexed as documents but deliberately stay out of this set, so
 * registering a source checkout as a library can never turn the Dashboard into an
 * editor for the files inside it.
 */
export const EDITABLE_ASSET_EXTENSIONS = new Set([".md", ".mdx", ".txt", ".json", ".jsonl", ".csv", ".srt", ".vtt"]);

const SKIPPED_DIRECTORIES = new Set([
  ".git", ".cache", ".next", ".pytest_cache", ".venv", "build", "dist", "node_modules",
  "out", "target", "venv", "__pycache__",
]);

// Generated intermediates must never land in someone's footage, so ffmpeg scratch
// files live beside the vector store instead of next to the clips. Saving a text
// asset is the one sanctioned write into a root, and it replaces that file only.
const CHUNK_SUBDIRECTORY = "chunks";
const PREVIEW_SUBDIRECTORY = "previews";

const DEFAULT_MAX_FILES_PER_LIBRARY = 100_000;
const MAX_DOCUMENT_BYTES = 2 * 1024 * 1024;
const MAX_IMAGE_BYTES = 32 * 1024 * 1024;
const MAX_EDITED_BYTES = 1024 * 1024;
const PREVIEW_CHARACTERS = 280;
const EMBED_CONCURRENCY = 2;
const STATUS_CACHE_MILLISECONDS = 15_000;
const MODEL_INFO_CACHE_MILLISECONDS = 900_000;

const ASSET_DOCUMENT_QUERY_INSTRUCTION =
  "Retrieve the transcripts, notes and documents of personal creative assets relevant to this request.";

const ASSET_MIME_TYPES: Record<string, string> = {
  ".mp4": "video/mp4", ".mov": "video/quicktime", ".m4v": "video/x-m4v", ".mkv": "video/x-matroska",
  ".webm": "video/webm", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
  ".webp": "image/webp", ".gif": "image/gif", ".heic": "image/heic", ".avif": "image/avif",
  ".pdf": "application/pdf", ".doc": "application/msword",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
};

export type AssetKind = "video" | "image" | "document";
const ASSET_KIND_ORDER: AssetKind[] = ["video", "image", "document"];

export interface AssetHit {
  id: string;
  source: "local" | "web";
  path: string;
  openUrl: string;
  name: string;
  kind: AssetKind | string;
  libraryPath: string;
  libraryName: string;
  size: number;
  modifiedAt: number;
  score: number;
  segmentIndex: number;
  startSeconds: number;
  endSeconds: number;
  duration: number;
  embeddingBasis: string;
  previewUrl: string;
  sidecarText: string;
  legacy: boolean;
}

export type AssetJobState = "idle" | "queued" | "running" | "done" | "cancelled" | "error";

export interface AssetJob {
  root: string;
  /** Configuration snapshot for this run; never infer it from later UI selections. */
  executionMode?: string;
  storageTimings?: LocalVectorSessionTimings;
  /** Comparable only for explicitly recomputed, identical inputs/settings. */
  benchmarkKey?: string;
  state: AssetJobState;
  phase: string;
  done: number;
  total: number;
  current: string;
  currentStartedAt: number;
  currentUnitsDone: number;
  currentUnitsTotal: number;
  currentUnitLabel: string;
  currentUnitStartedAt: number;
  currentCompletedUnitMs: number;
  lastUnitElapsedMs: number;
  currentMediaDurationSeconds: number;
  currentProcessedMediaSeconds: number;
  lastFileElapsedMs: number;
  completedWorkMs: number;
  indexed: number;
  skipped: number;
  failed: number;
  error: string;
  startedAt: number;
  finishedAt: number;
  discovered: number;
  remaining: number;
  cancelRequested: boolean;
  processingMode: "full-speed" | "background";
  concurrency: number;
}

export interface AssetLibraryEntry {
  id: string;
  path: string;
  name: string;
  kinds: AssetKind[];
  assetCount: number;
  lastScanAt: number;
  state: AssetJobState;
  job: AssetJob | null;
}

export interface AssetLibrariesStatus {
  libraries: AssetLibraryEntry[];
  autoScan: boolean;
  scanIntervalSeconds: number;
  maxAssetsPerScan: number;
  maxFilesPerLibrary: number;
}

export interface ScanOptions {
  /** Execute real preprocessing/model work without reading or mutating vector storage. */
  dryRun?: boolean;
  limit?: number;
  config?: ReturnType<typeof loadConfig>;
  onProgress?: (job: AssetJob) => void;
  /** Resident server and CLI scans opt in; direct library calls stay side-effect free. */
  recordPerformance?: boolean;
  /** Test/support override; production uses the user-level state path. */
  performanceHistoryPath?: string;
  /** Cooperative cancellation for directory walking, ffmpeg and embedding requests. */
  signal?: AbortSignal;
  /** Low-interference mode serializes work without artificial scheduling delays. */
  background?: boolean;
  /** Explicit per-directory demo: recompute cached files through normal ingestion. */
  benchmark?: boolean;
}

export interface ScanResult {
  modelTimings?: { requests: number; preprocessMs: number; visionMs: number; languageMs: number; totalMs: number };
  inputSignature?: string;
  processedMediaSeconds?: number;
  ok: boolean;
  root: string;
  embeddingSpace: string;
  discovered: number;
  indexed: number;
  skipped: number;
  failed: number;
  deleted: number;
  vectors: number;
  remaining: number;
  elapsedMs: number;
  indexedByKind: KindCounters;
  workMsByKind: KindCounters;
  warnings: string[];
  errors: string[];
}

export interface SearchOptions extends OperationOptions {
  kind?: string;
  limit?: number;
  /** Keep hits whose file is no longer on disk. Off by default: a result you cannot open is a bug report. */
  includeMissing?: boolean;
  /** Search the read-only legacy table too. On by default; it may fill at most half a page. */
  includeLegacy?: boolean;
  config?: ReturnType<typeof loadConfig>;
}

export interface SearchAssetsResult {
  query: string;
  embeddingSpace: string;
  reranked: false;
  hits: AssetHit[];
  /** Hits dropped because the file behind them is gone; reported so the gap is visible. */
  hiddenMissing: number;
}

export interface ListOptions {
  limit?: number;
  kind?: string;
  library?: string;
  includeMissing?: boolean;
  config?: ReturnType<typeof loadConfig>;
}

export interface PruneOptions {
  library?: string;
  /** Report what would be removed without touching the index. */
  dryRun?: boolean;
  config?: ReturnType<typeof loadConfig>;
}

export interface PruneResult {
  ok: true;
  dryRun: boolean;
  embeddingSpace: string;
  scanned: number;
  /** Distinct asset files whose rows are being removed. */
  missingAssets: Array<{ path: string; kind: string; rows: number; libraryPath: string }>;
  removedRows: number;
}

export interface ServedAsset {
  path: string;
  name: string;
  kind: AssetKind;
  extension: string;
  contentType: string;
  size: number;
}

export class AssetAccessError extends Error {
  readonly status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = "AssetAccessError";
    this.status = status;
  }
}

interface VectorRow {
  key: string;
  distance?: number;
  metadata?: Record<string, unknown>;
}

interface DiscoveredFile {
  path: string;
  name: string;
  extension: string;
  stem: string;
  kind: AssetKind;
  size: number;
  modifiedAt: number;
  fingerprint: string;
}

type DirectConfig = ReturnType<typeof directConfig>;

interface AssetContext {
  config: ReturnType<typeof loadConfig>;
  profile: ReturnType<typeof activeProfile>;
  direct: DirectConfig;
  space: string;
  roots: string[];
  settings: { autoScan: boolean; scanIntervalSeconds: number; maxAssetsPerScan: number; maxFilesPerLibrary: number; kinds: AssetKind[] };
}

function normalizedAssetKinds(value: unknown, fallback: AssetKind[] = ASSET_KIND_ORDER): AssetKind[] {
  if (!Array.isArray(value)) return [...fallback];
  const selected = new Set(value.map((item) => String(item || "").trim().toLowerCase()));
  return ASSET_KIND_ORDER.filter((kind) => selected.has(kind));
}

function libraryRoots(config: ReturnType<typeof loadConfig>): string[] {
  const raw: unknown[] = Array.isArray(config.library?.roots) ? config.library.roots as unknown[] : [];
  const roots = raw.map((value: unknown) => String(value || "").trim()).filter(Boolean);
  return [...new Set<string>(roots.map((value: string) => path.resolve(value)))];
}

function librarySettings(config: ReturnType<typeof loadConfig>) {
  const library = (config.library || {}) as Record<string, unknown>;
  const kinds = normalizedAssetKinds(library.kinds);
  return {
    autoScan: library.autoScan === undefined ? true : Boolean(library.autoScan),
    scanIntervalSeconds: Number(library.scanIntervalSeconds || 900),
    maxAssetsPerScan: Number(library.maxAssetsPerScan || 400),
    maxFilesPerLibrary: Math.max(1, Math.floor(Number(library.maxFilesPerLibrary || DEFAULT_MAX_FILES_PER_LIBRARY))),
    kinds: kinds.length ? kinds : [...ASSET_KIND_ORDER],
  };
}

function libraryKinds(config: ReturnType<typeof loadConfig>, root: string): AssetKind[] {
  const settings = librarySettings(config);
  const raw = config.library?.rootKinds;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return settings.kinds;
  const selected = normalizedAssetKinds((raw as Record<string, unknown>)[path.resolve(root)], settings.kinds);
  return selected.length ? selected : settings.kinds;
}

function context(config = loadConfig()): AssetContext {
  const profile = activeProfile(config);
  const direct = directConfig(profile);
  // The asset library writes zvec only; cloud profiles keep their own index
  // contract and must never receive asset records from this scanner.
  if (direct.storageProvider !== "local") throw new Error("素材库需要本地 zvec 档案");
  return { config, profile, direct, space: direct.embeddingSpace, roots: libraryRoots(config), settings: librarySettings(config) };
}

function dataSibling(direct: DirectConfig, relative: string): string {
  return path.resolve(path.dirname(localVectorDatabasePath(direct)), relative);
}

export function assetChunkDirectory(direct: DirectConfig): string {
  return dataSibling(direct, CHUNK_SUBDIRECTORY);
}

export function assetPreviewDirectory(direct: DirectConfig): string {
  return dataSibling(direct, PREVIEW_SUBDIRECTORY);
}

export function assetKindOf(extension: string): AssetKind | "" {
  if (ASSET_EXTENSIONS.video.has(extension)) return "video";
  if (ASSET_EXTENSIONS.image.has(extension)) return "image";
  if (ASSET_EXTENSIONS.document.has(extension)) return "document";
  return "";
}

export function isServingExtension(extension: string): boolean {
  return Boolean(assetKindOf(extension));
}

export function assetContentType(extension: string): string {
  return ASSET_MIME_TYPES[extension] || "application/octet-stream";
}

function assetKey(assetPath: string, segmentIndex: number, space: string): Promise<string> {
  return uuid5Url(`asset:${assetPath}:${segmentIndex}:${space}`);
}

const modelInfoCache = new Map<string, { at: number; value: { model: string; maxModelLen: number; embeddingSpace: string } }>();

/** Model name, advertised context length and configured dimension for the active profile. */
export async function assetModelInfo(config = loadConfig(), options: OperationOptions = {}): Promise<{ model: string; maxModelLen: number; dimension: number; embeddingSpace: string }> {
  throwIfAborted(options.signal);
  const { profile, direct } = context(config);
  const cacheKey = `${direct.embeddingBaseUrl}|${direct.embeddingModel}|${direct.embeddingSpace}`;
  const cached = modelInfoCache.get(cacheKey);
  if (cached && Date.now() - cached.at < MODEL_INFO_CACHE_MILLISECONDS) {
    return { ...cached.value, dimension: direct.embeddingDimension };
  }
  const discovered = await discoverModelInfo(direct, options);
  throwIfAborted(options.signal);
  if (direct.embeddingSpaceExplicit && discovered.embeddingSpace && direct.embeddingSpace
    && discovered.embeddingSpace !== direct.embeddingSpace) {
    throw new Error(
      `Embedding 服务向量空间不匹配：配置 ${direct.embeddingSpace}，服务 ${discovered.embeddingSpace}`,
    );
  }
  const value = {
    model: discovered.model || direct.embeddingModel,
    maxModelLen: discovered.maxModelLen || DEFAULT_MAX_MODEL_LEN,
    embeddingSpace: discovered.embeddingSpace || "",
  };
  modelInfoCache.set(cacheKey, { at: Date.now(), value });
  return { ...value, dimension: direct.embeddingDimension };
}

function toNumber(value: unknown, fallback = 0): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function previewUrlFor(assetPath: string, atSeconds: number): string {
  return `/api/assets/preview?asset=${encodeURIComponent(assetPath)}&at=${Math.max(0, Math.floor(atSeconds))}`;
}

function assetHit(row: VectorRow, legacy: boolean): AssetHit {
  const metadata = row.metadata || {};
  const assetPath = String(metadata.asset_path || metadata.file_path || metadata.source_path || metadata.source_uri || "");
  const rawKind = String(metadata.file_kind || metadata.modality || assetKindOf(path.extname(assetPath).toLowerCase()) || "document");
  const libraryPath = String(metadata.library_root || metadata.project_name || "");
  const startSeconds = toNumber(metadata.start_time ?? metadata.timestamp, 0);
  const endSeconds = toNumber(metadata.end_time ?? metadata.timestamp, startSeconds);
  const score = row.distance === undefined ? 0 : scoreFromDistance(row.distance);
  return {
    id: String(row.key || ""),
    source: "local",
    path: assetPath,
    openUrl: "",
    name: String(metadata.asset_name || metadata.file_name || metadata.title || path.basename(assetPath) || "Untitled"),
    kind: rawKind === "visual" ? "video" : rawKind,
    libraryPath,
    libraryName: libraryPath ? path.basename(libraryPath) : "",
    size: toNumber(metadata.file_size ?? metadata.size_bytes, 0),
    modifiedAt: toNumber(metadata.modified_at_ms, 0),
    score,
    segmentIndex: toNumber(metadata.segment_index, 0),
    startSeconds,
    endSeconds,
    duration: toNumber(metadata.duration, 0),
    embeddingBasis: String(metadata.embedding_basis || (legacy ? "legacy" : "")),
    previewUrl: legacy ? "" : (assetPath ? previewUrlFor(assetPath, startSeconds) : ""),
    sidecarText: String(metadata.text_preview || metadata.content_excerpt || metadata.transcript_text || ""),
    legacy,
  };
}

function scanAbortedError(): Error {
  const error = new Error("扫描已停止");
  error.name = "AbortError";
  return error;
}

function isScanAborted(error: unknown): boolean {
  return (error as Error)?.name === "AbortError" || String((error as Error)?.message || error) === "扫描已停止";
}

function throwIfScanAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw scanAbortedError();
}

interface WalkResult {
  files: DiscoveredFile[];
  truncated: boolean;
}

function walkLibrary(root: string, kinds: Set<string>, maxFiles = DEFAULT_MAX_FILES_PER_LIBRARY, signal?: AbortSignal): WalkResult {
  const output: DiscoveredFile[] = [];
  let truncated = false;
  const visit = (directory: string, depth: number) => {
    throwIfScanAborted(signal);
    if (output.length >= maxFiles) { truncated = true; return; }
    if (depth > 12) return;
    let entries: fs.Dirent[] = [];
    try { entries = fs.readdirSync(directory, { withFileTypes: true }); } catch { return; }
    for (const entry of entries) {
      throwIfScanAborted(signal);
      if (output.length >= maxFiles) { truncated = true; break; }
      if (entry.name.startsWith(".") || SKIPPED_DIRECTORIES.has(entry.name)) continue;
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) { visit(target, depth + 1); continue; }
      if (!entry.isFile()) continue;
      const extension = path.extname(entry.name).toLowerCase();
      const kind = assetKindOf(extension);
      if (!kind || !kinds.has(kind)) continue;
      let stat: fs.Stats;
      try { stat = fs.statSync(target); } catch { continue; }
      output.push({
        path: target,
        name: entry.name,
        extension,
        stem: path.basename(entry.name, extension),
        kind,
        size: stat.size,
        modifiedAt: Math.round(stat.mtimeMs),
        fingerprint: `${stat.size}:${Math.round(stat.mtimeMs)}`,
      });
    }
  };
  visit(root, 0);
  // Deterministic order makes the per-scan budget resumable across runs.
  return { files: output.sort((left, right) => left.path.localeCompare(right.path)), truncated };
}

function sidecarPartner(
  file: DiscoveredFile,
  byStem: Map<string, string[]>,
  wanted: ReadonlySet<string>,
): string {
  const candidates = byStem.get(file.stem.toLowerCase()) || [];
  if (!candidates.length) return "";
  const directory = path.dirname(file.path);
  // A same-directory match wins; otherwise the nearest directory keeps a shot
  // library's transcripts/<stem>.txt and analysis/<stem>.md convention working.
  const ranked = candidates
    .filter((candidate) => candidate !== file.path && wanted.has(path.extname(candidate).toLowerCase()))
    .map((candidate) => ({
      candidate,
      sameDirectory: path.dirname(candidate) === directory ? 0 : 1,
      distance: path.relative(directory, path.dirname(candidate)).split(path.sep).filter(Boolean).length,
    }))
    .sort((left, right) => left.sameDirectory - right.sameDirectory || left.distance - right.distance || left.candidate.localeCompare(right.candidate));
  return ranked[0]?.candidate || "";
}

function stemsOf(files: DiscoveredFile[], wanted: ReadonlySet<string>): Map<string, string[]> {
  const index = new Map<string, string[]>();
  for (const file of files) {
    if (!wanted.has(file.extension)) continue;
    const bucket = index.get(file.stem.toLowerCase());
    if (bucket) bucket.push(file.path);
    else index.set(file.stem.toLowerCase(), [file.path]);
  }
  return index;
}

function readAssetText(file: DiscoveredFile): string {
  if (file.size > MAX_DOCUMENT_BYTES) return "";
  try { return fs.readFileSync(file.path, "utf8").slice(0, 64 * 1024).replace(/\0/g, " ").trim(); } catch { return ""; }
}

interface AssetMetadataFields {
  assetPath: string;
  assetName: string;
  kind: AssetKind;
  extension: string;
  size: number;
  modifiedAt: number;
  libraryRoot: string;
  fingerprint: string;
  sidecarOf: string;
  textPreview: string;
  embeddingModel: string;
  space: string;
  embeddingInputVersion: string;
  overrides?: Record<string, unknown>;
}

/**
 * One metadata shape for every writer, so a document saved through the editor and
 * a document picked up by the scanner are the same record to the reader: identical
 * field names, identical defaults, identical embedding space.
 */
function assetMetadata(fields: AssetMetadataFields): Record<string, unknown> {
  return {
    record_type: ASSET_RECORD_TYPE,
    asset_path: fields.assetPath,
    asset_name: fields.assetName,
    file_kind: fields.kind,
    extension: fields.extension,
    file_size: fields.size,
    modified_at_ms: fields.modifiedAt,
    library_root: fields.libraryRoot,
    segment_index: 0,
    start_time: 0,
    end_time: 0,
    duration: 0,
    embedding_basis: "",
    fingerprint: fields.fingerprint,
    sidecar_of: fields.sidecarOf,
    text_preview: fields.textPreview,
    embedding_model: fields.embeddingModel,
    embedding_space: fields.space,
    embedding_input_version: fields.embeddingInputVersion,
    indexed_at_ms: Date.now(),
    ...(fields.overrides || {}),
  };
}

function assetEmbeddingInputVersion(kind: AssetKind, direct: ReturnType<typeof directConfig>): string {
  return embeddingCorpusInputVersion(direct, kind === "document" ? "text" : kind);
}

/**
 * Media input already used WeMM's released user-only shape before input versions
 * were recorded. Text did not, so only unversioned text rows require migration.
 */
function currentAssetInput(metadata: Record<string, unknown>, direct: ReturnType<typeof directConfig>): boolean {
  const kind = String(metadata.file_kind || "") as AssetKind;
  const recorded = String(metadata.embedding_input_version || "");
  if (!recorded) return kind !== "document";
  if (kind !== "video" && kind !== "image" && kind !== "document") return false;
  if (recorded !== assetEmbeddingInputVersion(kind, direct)) return false;
  return kind !== "document" || String(metadata.document_chunk_version || "") === DOCUMENT_CHUNK_VERSION;
}

/** The registered library a file belongs to, reported exactly as it is configured. */
function libraryRootOf(assetPath: string, roots: string[]): string {
  for (const root of roots) {
    let realRoot = "";
    try { realRoot = fs.realpathSync(root); } catch { continue; }
    if (assetPath === realRoot || assetPath.startsWith(`${realRoot}${path.sep}`)) return root;
  }
  return "";
}

/**
 * Keep the path stored in vector metadata in the namespace of the registered
 * library root. macOS exposes some directories through symlink aliases (notably
 * `/var` and `/private/var`): filesystem access must use the canonical path for
 * the traversal guard, while indexed identity must keep the configured spelling
 * used by the scanner and existing rows.
 */
function indexedAssetPath(assetPath: string, roots: string[]): string {
  for (const root of roots) {
    let realRoot = "";
    try { realRoot = fs.realpathSync(root); } catch { continue; }
    if (assetPath === realRoot) return root;
    if (assetPath.startsWith(`${realRoot}${path.sep}`)) {
      return path.join(root, path.relative(realRoot, assetPath));
    }
  }
  return assetPath;
}

/** Media file with the same stem next to a text file, mirroring the scanner convention. */
function mediaPartnerOf(assetPath: string): string {
  const extension = path.extname(assetPath).toLowerCase();
  const stem = path.basename(assetPath, extension).toLowerCase();
  const directory = path.dirname(assetPath);
  let entries: fs.Dirent[];
  try { entries = fs.readdirSync(directory, { withFileTypes: true }); } catch { return ""; }
  const partners = entries
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .sort((left, right) => left.localeCompare(right));
  for (const name of partners) {
    const partnerExtension = path.extname(name).toLowerCase();
    if (!ASSET_EXTENSIONS.video.has(partnerExtension) && !ASSET_EXTENSIONS.image.has(partnerExtension)) continue;
    if (path.basename(name, partnerExtension).toLowerCase() === stem) return path.join(directory, name);
  }
  return "";
}

/** Local zvec writes from concurrent workers are serialised for its single-writer contract. */
function createWriteLock() {
  let tail: Promise<unknown> = Promise.resolve();
  return <T>(task: () => Promise<T>): Promise<T> => {
    const result = tail.then(task, task);
    tail = result.then(() => undefined, () => undefined);
    return result;
  };
}

async function mapWithConcurrency<T>(items: T[], limit: number, worker: (item: T) => Promise<void>, signal?: AbortSignal): Promise<void> {
  let cursor = 0;
  const runners = Array.from({ length: Math.max(1, Math.min(limit, items.length || 1)) }, async () => {
    while (cursor < items.length) {
      throwIfScanAborted(signal);
      const index = cursor;
      cursor += 1;
      await worker(items[index] as T);
    }
  });
  // Do not close the shared database session while a sibling worker is still
  // unwinding its aborted model request or completing a write.
  const settled = await Promise.allSettled(runners);
  const failed = settled.find((result): result is PromiseRejectedResult => result.status === "rejected");
  if (failed) throw failed.reason;
}

function newJob(root: string): AssetJob {
  return {
    root, state: "queued", phase: "排队中", done: 0, total: 0, current: "",
    currentStartedAt: 0, currentUnitsDone: 0, currentUnitsTotal: 0, currentUnitLabel: "",
    currentUnitStartedAt: 0, currentCompletedUnitMs: 0, lastUnitElapsedMs: 0,
    currentMediaDurationSeconds: 0, currentProcessedMediaSeconds: 0,
    lastFileElapsedMs: 0, completedWorkMs: 0,
    indexed: 0, skipped: 0, failed: 0, error: "", startedAt: 0, finishedAt: 0,
    discovered: 0, remaining: 0, cancelRequested: false,
    processingMode: "full-speed", concurrency: EMBED_CONCURRENCY,
  };
}

/**
 * Scan one library root into the asset index: fingerprint-equal files are
 * skipped, so a scan interrupted by the per-run budget resumes where it stopped.
 */
export async function scanLibrary(root: string, options: ScanOptions = {}): Promise<ScanResult> {
  const ctx = context(options.config);
  const target = path.resolve(String(root || "").trim());
  if (!target || target === path.resolve(".")) throw new Error("需要素材库目录路径");
  if (!fs.existsSync(target) || !fs.statSync(target).isDirectory()) throw new Error(`素材库目录不存在：${target}`);

  const maxAssets = Math.max(1, Math.floor(toNumber(options.limit, ctx.settings.maxAssetsPerScan)));
  const concurrency = options.background ? 1 : EMBED_CONCURRENCY;
  const job: AssetJob = {
    ...newJob(target), state: "running", phase: "盘点文件", total: 0, startedAt: Date.now(),
    executionMode: ctx.profile.embedding?.provider === "apple-native"
      ? String(ctx.profile.embedding.native?.executionMode || "b") : "remote",
    processingMode: options.background ? "background" : "full-speed", concurrency,
  };
  const report = () => { options.onProgress?.(job); };
  report();
  const modelTimings = { requests: 0, preprocessMs: 0, visionMs: 0, languageMs: 0, totalMs: 0 };
  const embeddingOptions = { signal: options.signal, onTimings: (timings: Record<string, unknown>) => {
    modelTimings.requests += 1;
    for (const [source, target] of [["preprocess", "preprocessMs"], ["vision", "visionMs"], ["language", "languageMs"], ["total", "totalMs"]] as const) {
      const value = Number(timings[source]);
      if (Number.isFinite(value) && value >= 0) modelTimings[target] += value;
    }
  } };
  const result = await withLocalVectorSession(ctx.direct, async () => {
    const kinds = new Set(libraryKinds(ctx.config, target));
    const walked = walkLibrary(target, kinds, ctx.settings.maxFilesPerLibrary, options.signal);
    const discovered = walked.files;
    throwIfScanAborted(options.signal);
    const model = await assetModelInfo(ctx.config, options).catch(() => {
      throwIfAborted(options.signal);
      return { model: ctx.direct.embeddingModel, maxModelLen: DEFAULT_MAX_MODEL_LEN, dimension: ctx.direct.embeddingDimension };
    });

    const existing = options.dryRun ? [] : await listLocalVectors(ctx.direct.assetIndex, ctx.direct, {
      filter: metadataFilter({ record_type: ASSET_RECORD_TYPE, embedding_space: ctx.space }),
      returnData: false,
    });
    const fingerprints = new Map<string, Set<string>>();
    const keysByPath = new Map<string, string[]>();
    for (const row of existing) {
      const metadata = row.metadata || {};
      const assetPath = String(metadata.asset_path || "");
      if (!assetPath || String(metadata.library_root || "") !== target) continue;
      const fingerprint = currentAssetInput(metadata, ctx.direct) ? String(metadata.fingerprint || "") : "";
      const bucket = fingerprints.get(assetPath);
      if (bucket) bucket.add(fingerprint);
      else fingerprints.set(assetPath, new Set([fingerprint]));
      const keys = keysByPath.get(assetPath);
      if (keys) keys.push(row.key);
      else keysByPath.set(assetPath, [row.key]);
    }

    const result: ScanResult = {
      ok: true, root: target, embeddingSpace: ctx.space, discovered: discovered.length,
      indexed: 0, skipped: 0, failed: 0, deleted: 0, vectors: 0, remaining: 0, elapsedMs: 0,
      indexedByKind: { video: 0, image: 0, document: 0 },
      workMsByKind: { video: 0, image: 0, document: 0 },
      warnings: [], errors: [], processedMediaSeconds: 0,
    };
    job.discovered = discovered.length;
    if (walked.truncated) result.warnings.push(`目录盘点达到 ${ctx.settings.maxFilesPerLibrary} 个文件的上限，可在扫描设置中调高`);

    const writeLock = createWriteLock();
    const write = <T>(task: () => Promise<T>) => options.dryRun ? Promise.resolve(undefined) : writeLock(task);

    // Files that vanished or whose kind is no longer tracked lose their records;
    // other libraries and the read-only legacy index remain untouched.
    const discoveredPaths = new Set(discovered.map((file) => file.path));
    const vanished: string[] = [];
    for (const [assetPath, keys] of keysByPath) {
      if (!discoveredPaths.has(assetPath)) vanished.push(...keys);
    }
    if (vanished.length) {
      for (let offset = 0; offset < vanished.length; offset += 500) {
        await write(() => deleteLocalVectors(ctx.direct.assetIndex, vanished.slice(offset, offset + 500), ctx.direct));
      }
      result.deleted = vanished.length;
      job.phase = `清理 ${vanished.length} 条失效记录`;
      report();
    }

    const pending: DiscoveredFile[] = [];
    for (const file of discovered) {
      const known = fingerprints.get(file.path);
      if (!options.benchmark && known && known.size && !known.has("") && known.has(file.fingerprint)) {
        result.skipped += 1;
        continue;
      }
      pending.push(file);
    }
    result.remaining = Math.max(0, pending.length - maxAssets);
    // The UI reports files, not internal video segments. Start with every stale
    // or unseen file and count down; the final value is the next-run backlog.
    job.remaining = pending.length;
    const queue = pending.slice(0, maxAssets);
    if (options.benchmark) {
      job.benchmarkKey = crypto.createHash("sha256").update(JSON.stringify({
        version: 1, root: target, space: ctx.space, model: model.model,
        dimension: model.dimension, maxModelLen: model.maxModelLen,
        video: ctx.profile.video, concurrency,
        files: queue.map((file) => [file.path, file.fingerprint, assetEmbeddingInputVersion(file.kind, ctx.direct)]).sort(),
      })).digest("hex");
    }
    job.total = queue.length;
    job.phase = `索引 ${queue.length} 个素材`;
    report();

    const mediaByStem = stemsOf(discovered, new Set([...ASSET_EXTENSIONS.video, ...ASSET_EXTENSIONS.image]));
    const textByStem = stemsOf(discovered, SIDECAR_EXTENSIONS);
    const chunkDir = assetChunkDirectory(ctx.direct);
    const libraryRootsForGuard = [...new Set([...ctx.roots, target])];

    const embedAsset = async (file: DiscoveredFile): Promise<{ vectors: number; error: string }> => {
      const sidecarPath = file.kind === "document"
        ? sidecarPartner(file, mediaByStem, new Set([...ASSET_EXTENSIONS.video, ...ASSET_EXTENSIONS.image]))
        : "";
      const sidecarText = file.kind === "document"
        ? ""
        : (sidecarTextOf(file, textByStem));
      const partner = file.kind === "document" ? sidecarPath : "";
      const metadataOf = (overrides: Record<string, unknown> = {}): Record<string, unknown> => assetMetadata({
        assetPath: file.path,
        assetName: file.name,
        kind: file.kind,
        extension: file.extension,
        size: file.size,
        modifiedAt: file.modifiedAt,
        libraryRoot: target,
        fingerprint: file.fingerprint,
        sidecarOf: partner,
        textPreview: String(sidecarText || "").slice(0, PREVIEW_CHARACTERS),
        embeddingModel: ctx.direct.embeddingModel,
        space: ctx.space,
        embeddingInputVersion: assetEmbeddingInputVersion(file.kind, ctx.direct),
        overrides,
      });

      if (file.kind === "image") {
        throwIfScanAborted(options.signal);
        if (file.size > MAX_IMAGE_BYTES) return { vectors: 0, error: `${file.name} 图片过大（${file.size} 字节）` };
      const vector = await embedImage(fs.readFileSync(file.path).toString("base64"), assetContentType(file.extension), "", ctx.direct, embeddingOptions);
        const key = await assetKey(file.path, 0, ctx.space);
        await write(() => putLocalVectors(ctx.direct.assetIndex, [{
          key, data: { float32: vector }, metadata: { ...metadataOf(), embedding_basis: "image_v1" },
        }], ctx.direct));
        return { vectors: 1, error: "" };
      }

      if (file.kind === "document") {
        let text = "";
        try {
          text = await extractDocumentText(file.path);
        } catch (error) {
          return { vectors: 0, error: `${file.name} ${String((error as Error)?.message || error)}` };
        }
        const planned = chunkDocumentText(text);
        if (!planned.chunks.length) return { vectors: 0, error: `${file.name} 没有可读文本` };
        if (planned.truncated) result.warnings.push(`${file.name}: 文档超过 ${planned.characters} 字符，只索引前 ${planned.chunks.at(-1)?.endCharacter || 0} 字符`);
        const rows: Array<{ key: string; data: { float32: number[] }; metadata: Record<string, unknown> }> = [];
        job.currentUnitLabel = "文本块";
        job.currentUnitsDone = 0;
        job.currentUnitsTotal = planned.chunks.length;
        job.currentUnitStartedAt = 0;
        job.currentCompletedUnitMs = 0;
        job.lastUnitElapsedMs = 0;
        report();
        for (const chunk of planned.chunks) {
          throwIfScanAborted(options.signal);
          const unitStartedAt = Date.now();
          job.currentUnitStartedAt = unitStartedAt;
          report();
          const vector = await embedDocument(chunk.text, ctx.direct, embeddingOptions);
          rows.push({
            key: await assetKey(file.path, chunk.index, ctx.space),
            data: { float32: vector },
            metadata: {
              ...metadataOf({
                segment_index: chunk.index,
                character_start: chunk.startCharacter,
                character_end: chunk.endCharacter,
                text_preview: chunk.text.slice(0, PREVIEW_CHARACTERS),
                document_chunk_version: DOCUMENT_CHUNK_VERSION,
              }),
              embedding_basis: partner ? "sidecar_text_v1" : "document_text_v1",
            },
          });
          job.lastUnitElapsedMs = Math.max(1, Date.now() - unitStartedAt);
          job.currentCompletedUnitMs += job.lastUnitElapsedMs;
          job.currentUnitsDone += 1;
          job.currentUnitStartedAt = 0;
          report();
        }
        await write(() => putLocalVectors(ctx.direct.assetIndex, rows, ctx.direct));
        const fresh = new Set(rows.map((row) => row.key));
        const stale = (keysByPath.get(file.path) || []).filter((key) => !fresh.has(key));
        if (stale.length) await write(() => deleteLocalVectors(ctx.direct.assetIndex, stale, ctx.direct));
        return { vectors: rows.length, error: "" };
      }

      let duration = 0;
      try {
        throwIfScanAborted(options.signal);
        duration = (await probeMedia(file.path, options.signal ? { signal: options.signal } : {})).duration;
      } catch (error) {
        if (isScanAborted(error)) throw error;
        return { vectors: 0, error: `${file.name} 无法读取时长：${String((error as Error)?.message || error)}` };
      }
      const plan = planChunks(duration, ctx.profile.video, model.maxModelLen);
      if (plan.warning) result.warnings.push(`${file.name}: ${plan.warning}`);
      const rows: Array<{ key: string; data: { float32: number[] }; metadata: Record<string, unknown> }> = [];
      const failures: string[] = [];
      job.currentUnitLabel = "视频片段";
      job.currentUnitsDone = 0;
      job.currentUnitsTotal = plan.segments.length;
      job.currentUnitStartedAt = 0;
      job.currentCompletedUnitMs = 0;
      job.lastUnitElapsedMs = 0;
      job.currentMediaDurationSeconds = duration;
      job.currentProcessedMediaSeconds = 0;
      report();
      // A single long video otherwise never uses the file-level concurrency.
      // Keep the same two-request ceiling and do not multiply it for many files.
      const segmentConcurrency = queue.length === 1 && ctx.profile.embedding.provider === "apple-native"
        && ctx.profile.embedding.native.executionMode !== "e"
        ? Math.min(concurrency, ctx.profile.embedding.native.privateANE?.videoPipeline === 1 ? 1 : 2) : 1;
      const activeSegments = new Map<number, number>();
      await mapWithConcurrency(plan.segments, segmentConcurrency, async (segment) => {
        throwIfScanAborted(options.signal);
        const unitStartedAt = Date.now();
        activeSegments.set(segment.index, unitStartedAt);
        job.currentUnitStartedAt = Math.min(...activeSegments.values());
        report();
        let processed = false;
        try {
          const outcome = await withFramePacket(
            file.path,
            segment.startSeconds,
            segment.endSeconds,
            ctx.profile.video,
            chunkDir,
            async (frames) => ({
            vector: await embedVideoFrames(frames, ctx.direct, embeddingOptions),
              frameCount: frames.length,
            }),
            libraryRootsForGuard,
            options.signal,
          );
          rows.push({
            key: await assetKey(file.path, segment.index, ctx.space),
            data: { float32: outcome.vector },
            metadata: {
              ...metadataOf(),
              segment_index: segment.index,
              start_time: segment.startSeconds,
              end_time: segment.endSeconds,
              duration: Number(duration.toFixed(3)),
              frame_count: outcome.frameCount,
              capture_fps: ctx.profile.video.fps,
              video_transport: "ordered_frames",
              embedding_basis: "native_ordered_frames_2fps",
            },
          });
          processed = true;
        } catch (error) {
          if (isScanAborted(error)) throw error;
          failures.push(`${segment.startSeconds}-${segment.endSeconds}s ${String((error as Error)?.message || error)}`);
        }
        job.lastUnitElapsedMs = Math.max(1, Date.now() - unitStartedAt);
        job.currentCompletedUnitMs += job.lastUnitElapsedMs;
        job.currentUnitsDone += 1;
        if (processed) job.currentProcessedMediaSeconds = Math.min(duration, job.currentProcessedMediaSeconds + segment.endSeconds - segment.startSeconds);
        activeSegments.delete(segment.index);
        job.currentUnitStartedAt = activeSegments.size ? Math.min(...activeSegments.values()) : 0;
        report();
      }, options.signal);
      if (failures.length) {
        // A half-indexed video would be skipped forever by the fingerprint, so drop
        // the previous rows and let the next scan retry the whole clip.
        const stale = keysByPath.get(file.path) || [];
        if (stale.length) await write(() => deleteLocalVectors(ctx.direct.assetIndex, stale, ctx.direct));
        return { vectors: 0, error: `${file.name} 切片失败 ${failures.length}/${plan.segments.length}：${failures[0]}` };
      }
      if (!rows.length) return { vectors: 0, error: `${file.name} 没有可用片段` };
      await write(() => putLocalVectors(ctx.direct.assetIndex, rows, ctx.direct));
      result.processedMediaSeconds = (result.processedMediaSeconds || 0) + duration;
      return { vectors: rows.length, error: "" };
    };

    await mapWithConcurrency(queue, concurrency, async (file) => {
      const itemStartedAt = Date.now();
      job.current = file.name;
      job.currentStartedAt = itemStartedAt;
      job.currentUnitsDone = 0;
      job.currentUnitsTotal = 0;
      job.currentUnitLabel = "";
      job.currentUnitStartedAt = 0;
      job.currentCompletedUnitMs = 0;
      job.lastUnitElapsedMs = 0;
      job.currentMediaDurationSeconds = 0;
      job.currentProcessedMediaSeconds = 0;
      report();
      try {
        const outcome = await embedAsset(file);
        if (outcome.error) {
          result.failed += 1;
          if (result.errors.length < 20) result.errors.push(outcome.error);
          job.failed = result.failed;
        } else {
          result.indexed += 1;
          result.vectors += outcome.vectors;
          result.indexedByKind[file.kind] += 1;
          result.workMsByKind[file.kind] += Math.max(1, Date.now() - itemStartedAt);
          job.indexed = result.indexed;
        }
      } catch (error) {
        if (isScanAborted(error)) throw error;
        result.failed += 1;
        const message = `${file.name}: ${String((error as Error)?.message || error)}`;
        if (result.errors.length < 20) result.errors.push(message);
        job.failed = result.failed;
      }
      job.done += 1;
      job.lastFileElapsedMs = Math.max(1, Date.now() - itemStartedAt);
      job.completedWorkMs += job.lastFileElapsedMs;
      job.currentStartedAt = 0;
      job.currentUnitStartedAt = 0;
      job.remaining = Math.max(result.remaining, pending.length - job.done);
      report();
    }, options.signal);
    return result;
  }, (timings) => { job.storageTimings = timings; });

  job.phase = result.failed ? "完成（有失败项）" : "完成";
  job.state = result.failed && !result.indexed ? "error" : "done";
  job.error = result.errors[0] || "";
  job.finishedAt = Date.now();
  result.elapsedMs = Math.max(1, job.finishedAt - job.startedAt);
  result.modelTimings = modelTimings;
  if (job.benchmarkKey) result.inputSignature = job.benchmarkKey;
  job.current = "";
  report();
  if (options.recordPerformance && !options.dryRun) {
    try {
      recordIngestPerformance({
        config: ctx.config,
        elapsedMs: result.elapsedMs,
        concurrency,
        indexedByKind: result.indexedByKind,
        workMsByKind: result.workMsByKind,
        failed: result.failed,
        vectors: result.vectors,
        ...(options.performanceHistoryPath ? { historyPath: options.performanceHistoryPath } : {}),
      });
    } catch (error) {
      result.warnings.push(`入库性能历史保存失败：${String((error as Error)?.message || error)}`);
    }
  }
  return result;
}

function sidecarTextOf(file: DiscoveredFile, textByStem: Map<string, string[]>): string {
  const partner = sidecarPartner(
    { ...file, kind: "document" } as DiscoveredFile,
    textByStem,
    SIDECAR_EXTENSIONS,
  );
  if (!partner) return "";
  try {
    return fs.readFileSync(partner, "utf8").slice(0, 64 * 1024).replace(/\0/g, " ").trim().slice(0, PREVIEW_CHARACTERS * 4);
  } catch {
    return "";
  }
}

const jobs = new Map<string, AssetJob>();
const queue: string[] = [];
const queuedBackground = new Map<string, boolean>();
const queuedBenchmark = new Map<string, boolean>();
const scanControllers = new Map<string, AbortController>();
const activeRuns = new Map<string, Promise<ScanResult>>();
let pumping = false;
let scanTimer: ReturnType<typeof setInterval> | undefined;

export function queueEntries(): string[] {
  return [...queue];
}

/** In-memory check: a benchmark must not open vector storage just to inspect jobs. */
export function hasActiveAssetScans(): boolean {
  return queue.length > 0 || activeRuns.size > 0 || [...jobs.values()].some((job) => job.state === "running" || job.state === "queued");
}

function rootPaths(config: ReturnType<typeof loadConfig>): string[] {
  return libraryRoots(config);
}

async function pump(config: ReturnType<typeof loadConfig>): Promise<void> {
  if (pumping) return;
  pumping = true;
  try {
    while (queue.length) {
      const root = queue.shift() as string;
      const job = jobs.get(root) || newJob(root);
      jobs.set(root, job);
      job.state = "running";
      job.startedAt = Date.now();
      job.error = "";
      job.cancelRequested = false;
      const controller = new AbortController();
      scanControllers.set(root, controller);
      const background = queuedBackground.get(root) === true;
      queuedBackground.delete(root);
      const benchmark = queuedBenchmark.get(root) === true;
      queuedBenchmark.delete(root);
      try {
        // Resolve the config at execution time: a queued job must honor a mode
        // switch made while it was waiting behind another library.
        const currentConfig = loadConfig();
        const run = scanLibrary(root, {
          config: currentConfig,
          limit: librarySettings(currentConfig).maxAssetsPerScan,
          recordPerformance: true,
          signal: controller.signal,
          background,
          benchmark,
          onProgress: (progress) => {
            if (!controller.signal.aborted) Object.assign(job, progress, { state: "running" });
          },
        });
        activeRuns.set(root, run);
        const result = await run;
        job.phase = result.errors.length ? `完成（${result.errors.length} 个失败）` : "完成";
        job.state = result.failed && !result.indexed ? "error" : "done";
        job.error = result.errors[0] || "";
        invalidateStatus();
      } catch (error) {
        if (controller.signal.aborted || isScanAborted(error)) {
          job.state = "cancelled";
          job.phase = "已停止";
          job.error = "";
        } else {
          job.state = "error";
          job.phase = "失败";
          job.error = String((error as Error)?.message || error);
        }
      } finally {
        activeRuns.delete(root);
        scanControllers.delete(root);
      }
      job.finishedAt = Date.now();
      // Anything queued for this root while it was running was already collapsed.
      while (queue.includes(root)) queue.splice(queue.indexOf(root), 1);
    }
  } finally {
    pumping = false;
  }
}

/** Cancel one directory or every queued/running scan. Active model requests receive AbortSignal. */
export function cancelScan(root = ""): { cancelled: string[]; stopping: string[] } {
  const requested = String(root || "").trim();
  const targets = requested
    ? [path.resolve(requested)]
    : [...new Set([...queue, ...scanControllers.keys()])];
  const cancelled: string[] = [];
  const stopping: string[] = [];
  for (const target of targets) {
    const job = jobs.get(target);
    let queued = false;
    while (queue.includes(target)) {
      queue.splice(queue.indexOf(target), 1);
      queuedBackground.delete(target);
      queuedBenchmark.delete(target);
      queued = true;
    }
    const controller = scanControllers.get(target);
    if (controller && !controller.signal.aborted) {
      controller.abort(scanAbortedError());
      if (job) {
        job.cancelRequested = true;
        job.phase = "正在停止";
      }
      stopping.push(target);
    } else if (queued || job?.state === "queued") {
      if (job) {
        job.state = "cancelled";
        job.phase = "已停止";
        job.finishedAt = Date.now();
      }
      cancelled.push(target);
    }
  }
  return { cancelled, stopping };
}

/** Enqueue scans; one scan runs at a time and repeat requests collapse. */
export function queueScan(root = "", config = loadConfig(), options: { background?: boolean; benchmark?: boolean } = {}): string[] {
  if (options.benchmark && !String(root || "").trim()) throw new Error("对比测速需要选择一个素材目录");
  const targets = String(root || "").trim() ? [path.resolve(String(root).trim())] : rootPaths(config);
  const queued: string[] = [];
  for (const target of targets) {
    if (!fs.existsSync(target) || !fs.statSync(target).isDirectory()) continue;
    const active = jobs.get(target);
    if (active && (active.state === "queued" || active.state === "running")) continue;
    jobs.set(target, { ...newJob(target), startedAt: Date.now() });
    queue.push(target);
    queuedBackground.set(target, options.background === true);
    queuedBenchmark.set(target, options.benchmark === true);
    queued.push(target);
  }
  if (queued.length) void pump(config);
  return queued;
}

export function scanJob(root: string): AssetJob | null {
  return jobs.get(path.resolve(String(root || "").trim())) || null;
}

interface StatusSnapshot {
  counts: Map<string, { count: number; lastScanAt: number }>;
  /** root -> asset path -> the `size:mtime` fingerprint the row was written with. */
  assets: Map<string, Map<string, string>>;
}

interface StatusCache {
  at: number;
  key: string;
  snapshot: StatusSnapshot;
}

const statusCache: StatusCache = { at: 0, key: "", snapshot: { counts: new Map(), assets: new Map() } };

/** Rows and coverage read the same snapshot; one write invalidates both. */
function invalidateStatus(): void {
  statusCache.at = 0;
  coverageCache.clear();
}

async function assetCountsByRoot(config: ReturnType<typeof loadConfig>): Promise<Map<string, { count: number; lastScanAt: number }>> {
  return (await statusSnapshot(config)).counts;
}

async function assetsByRoot(config: ReturnType<typeof loadConfig>): Promise<Map<string, Map<string, string>>> {
  return (await statusSnapshot(config)).assets;
}

async function statusSnapshot(config: ReturnType<typeof loadConfig>): Promise<StatusSnapshot> {
  const ctx = context(config);
  const cacheKey = `${localVectorDatabasePath(ctx.direct)}|${ctx.direct.assetIndex}|${ctx.space}`;
  if (statusCache.key === cacheKey && statusCache.at && Date.now() - statusCache.at < STATUS_CACHE_MILLISECONDS) {
    return statusCache.snapshot;
  }
  const rows = await listLocalVectors(ctx.direct.assetIndex, ctx.direct, {
    filter: metadataFilter({ record_type: ASSET_RECORD_TYPE, embedding_space: ctx.space }),
    returnData: false,
  });
  const counts = new Map<string, { count: number; lastScanAt: number }>();
  const assets = new Map<string, Map<string, string>>();
  for (const row of rows) {
    const metadata = row.metadata || {};
    const root = String(metadata.library_root || "");
    if (!root) continue;
    const current = counts.get(root) || { count: 0, lastScanAt: 0 };
    current.count += 1;
    current.lastScanAt = Math.max(current.lastScanAt, toNumber(metadata.indexed_at_ms, 0));
    counts.set(root, current);
    const assetPath = String(metadata.asset_path || "");
    if (!assetPath) continue;
    const bucket = assets.get(root) || new Map<string, string>();
    const fingerprint = `${toNumber(metadata.file_size ?? metadata.size_bytes, 0)}:${toNumber(metadata.modified_at_ms, 0)}`;
    bucket.set(assetPath, currentAssetInput(metadata, ctx.direct) ? fingerprint : `outdated:${fingerprint}`);
    assets.set(root, bucket);
  }
  // assetCount reports distinct assets, not vector rows, which is what a person
  // reading the dashboard expects when a clip is split into chunks.
  for (const [root, bucket] of assets) {
    const current = counts.get(root);
    if (current) counts.set(root, { ...current, count: bucket.size });
  }
  statusCache.at = Date.now();
  statusCache.key = cacheKey;
  statusCache.snapshot = { counts, assets };
  return statusCache.snapshot;
}

export async function librariesStatus(config = loadConfig()): Promise<AssetLibrariesStatus> {
  const ctx = context(config);
  const counts = await assetCountsByRoot(config);
  return {
    libraries: ctx.roots.map((root) => {
      const job = jobs.get(root) || null;
      return {
        id: root,
        path: root,
        name: directoryLabel(root),
        kinds: libraryKinds(config, root),
        assetCount: counts.get(root)?.count || 0,
        lastScanAt: counts.get(root)?.lastScanAt || 0,
        state: job?.state || "idle",
        job,
      };
    }),
    autoScan: ctx.settings.autoScan,
    scanIntervalSeconds: ctx.settings.scanIntervalSeconds,
    maxAssetsPerScan: ctx.settings.maxAssetsPerScan,
    maxFilesPerLibrary: ctx.settings.maxFilesPerLibrary,
  };
}

/** Start the interval scanner used by the server; disabled when autoScan is off. */
export function startAutoScan(config = loadConfig()): { started: boolean; intervalSeconds: number; queued: string[] } {
  const ctx = context(config);
  if (!ctx.settings.autoScan) return { started: false, intervalSeconds: ctx.settings.scanIntervalSeconds, queued: [] };
  const interval = Math.max(30, Math.floor(ctx.settings.scanIntervalSeconds));
  const queued = queueScan("", config, { background: true });
  if (scanTimer) clearInterval(scanTimer);
  scanTimer = setInterval(() => {
    try {
      queueScan("", loadConfig(), { background: true });
    } catch {
      // A broken configuration must never take the server down; the next tick retries.
    }
  }, interval * 1000);
  scanTimer.unref?.();
  return { started: true, intervalSeconds: interval, queued };
}

export function stopAutoScan(): void {
  if (scanTimer) clearInterval(scanTimer);
  scanTimer = undefined;
}

/**
 * Whether the file behind a row is still on disk. Rows outlive files by design.
 * Relative paths (the shape a migrated legacy table carries) resolve against the
 * process working directory and therefore read as missing, which is what we want:
 * nothing outside the library can be opened or previewed by the dashboard.
 */
function assetFileExists(assetPath: string): boolean {
  if (!assetPath) return false;
  try {
    return fs.existsSync(assetPath) && fs.statSync(assetPath).isFile();
  } catch {
    return false;
  }
}

function byScore(left: AssetHit, right: AssetHit): number {
  return right.score - left.score;
}

function searchableName(value: string): string {
  return value
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/\.[a-z0-9]{1,8}$/i, "")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

/** Exact asset names are strong user intent that a content-only vector cannot see. */
function lexicalBoost(query: string, hit: AssetHit): number {
  const wanted = searchableName(query);
  const name = searchableName(hit.name);
  if (!wanted || !name) return 0;
  let boost = 0;
  if (wanted === name || wanted.includes(name)) boost = 0.18;
  const tokens = name.split(/\s+/).filter((token) => token.length >= 2);
  if (boost === 0 && tokens.length) {
    const matched = tokens.filter((token) => wanted.includes(token)).length;
    if (matched) boost = Math.min(0.12, 0.04 + 0.08 * (matched / tokens.length));
  }
  const extension = path.extname(hit.name).toLocaleLowerCase();
  if (extension === ".csv" && /(?:\bcsv\b|表格|逗号分隔)/iu.test(query)) boost += 0.14;
  else if (extension === ".pdf" && /(?:\bpdf\b|PDF文档|便携文档)/iu.test(query)) boost += 0.14;
  else if ((extension === ".doc" || extension === ".docx") && /(?:\bdocx?\b|\bword\b|Word文档)/iu.test(query)) boost += 0.12;
  else if ((extension === ".md" || extension === ".mdx") && /(?:\bmdx?\b|markdown|Markdown文档)/iu.test(query)) boost += 0.08;
  else if (extension === ".txt" && /(?:\btxt\b|纯文本|文本文件)/iu.test(query)) boost += 0.08;
  else if ((extension === ".json" || extension === ".jsonl") && /(?:\bjsonl?\b|结构化数据)/iu.test(query)) boost += 0.06;
  else if ((extension === ".srt" || extension === ".vtt") && /(?:\bsrt\b|\bvtt\b|字幕文件)/iu.test(query)) boost += 0.08;
  return Math.min(0.22, boost);
}

function applyLexicalBoost(query: string, hits: AssetHit[]): AssetHit[] {
  return hits.map((hit) => ({ ...hit, score: Math.min(1, hit.score + lexicalBoost(query, hit)) }));
}

/**
 * Merge vector recall from the new asset index and the read-only legacy index.
 * No reranking: the reranker stays opt-in through the profile.
 */
export async function searchAssets(query: string, options: SearchOptions = {}): Promise<SearchAssetsResult> {
  return withOperation(options, scope => searchAssetsWithin(query, { ...options, signal: scope.signal }));
}

async function searchAssetsWithin(query: string, options: SearchOptions): Promise<SearchAssetsResult> {
  const text = String(query || "").trim();
  if (!text) throw new Error("搜索文字不能为空");
  const ctx = context(options.config);
  const limit = Math.max(1, Math.min(200, Math.floor(toNumber(options.limit, 30))));
  const kind = String(options.kind || "").trim().toLowerCase();
  if (kind && !["video", "image", "document"].includes(kind)) throw new Error(`不支持的素材类型：${kind}`);
  // Hybrid signals can only repair rows the vector stage recalls. Keep a broad,
  // bounded candidate set even for a five-result page, then rank and slice below.
  const recallLimit = Math.min(500, Math.max(limit * 10, 100));
  const wantsVisual = !kind || kind === "video" || kind === "image";
  const wantsText = !kind || kind === "document";

  const [visualVector, textVector] = await Promise.all([
    wantsVisual ? embedVisualQuery(text, ctx.direct, options) : Promise.resolve<number[]>([]),
    wantsText ? embedTextWithInstruction(text, ASSET_DOCUMENT_QUERY_INSTRUCTION, ctx.direct, options) : Promise.resolve<number[]>([]),
  ]);

  const searches: Array<Promise<AssetHit[]>> = [];
  // An empty vector is not "nothing matched": reporting it that way hides a broken
  // embedding endpoint behind an empty result page.
  if (!visualVector.length && !textVector.length) {
    throw new Error("Embedding 服务没有返回向量，请检查服务地址、模型与维度配置");
  }
  if (wantsVisual && visualVector.length) {
    searches.push(queryLocalVectors(ctx.direct.assetIndex, visualVector, ctx.direct, {
      limit: recallLimit,
      filter: metadataFilter({
        record_type: ASSET_RECORD_TYPE,
        embedding_space: ctx.space,
        ...(kind ? { file_kind: kind } : {}),
      }),
    }).then((page) => page.vectors.map((row) => assetHit(row, false))));
  }
  if (wantsText && textVector.length) {
    if (kind !== "video" && kind !== "image") {
      searches.push(queryLocalVectors(ctx.direct.assetIndex, textVector, ctx.direct, {
        limit: recallLimit,
        filter: metadataFilter({
          record_type: ASSET_RECORD_TYPE,
          embedding_space: ctx.space,
          ...(kind ? { file_kind: kind } : {}),
        }),
      }).then((page) => page.vectors.map((row) => assetHit(row, false))));
    }
    // Legacy rows stay in their own table and are only ever read back. They come from
    // an earlier corpus whose paths may not belong to this library at all, so they are
    // filtered by the same kind a person picked and capped when merged below.
    if (options.includeLegacy !== false && ctx.direct.documentIndex) {
      searches.push(queryLocalVectors(ctx.direct.documentIndex, textVector, ctx.direct, {
        limit: recallLimit,
        filter: metadataFilter({
          embedding_space: ctx.space,
          ...(kind === "document" ? { modality: "document" } : {}),
        }),
      }).then((page) => page.vectors.map((row) => assetHit(row, true))));
    }
  }

  const settled = await Promise.allSettled(searches);
  throwIfAborted(options.signal);
  const failures = settled.filter((item) => item.status === "rejected") as Array<PromiseRejectedResult>;
  const hits = settled
    .filter((item): item is PromiseFulfilledResult<AssetHit[]> => item.status === "fulfilled")
    .flatMap((item) => item.value);
  if (!hits.length && failures.length === settled.length && settled.length) {
    throw new Error(String((failures[0] as PromiseRejectedResult)?.reason?.message || (failures[0] as { reason?: Error })?.reason || "向量检索失败"));
  }
  const unique = new Map<string, AssetHit>();
  for (const hit of hits) {
    // A long document owns many recall chunks but should remain one product result;
    // keep only its best matching passage. Video chunks remain independently useful.
    const identity = !hit.legacy && hit.kind === "document" && hit.path
      ? `document:${hit.path}`
      : hit.id || `${hit.path}:${hit.segmentIndex}`;
    const current = unique.get(identity);
    if (!current || hit.score > current.score) unique.set(identity, hit);
  }
  // A hit nobody can open is worse than no hit: files deleted outside the library
  // (a render folder cleaned by hand) keep their vectors until a prune runs.
  let hiddenMissing = 0;
  const alive: AssetHit[] = [];
  for (const hit of unique.values()) {
    if (!options.includeMissing && !assetFileExists(hit.path)) {
      hiddenMissing += 1;
      continue;
    }
    alive.push(hit);
  }
  const boosted = applyLexicalBoost(text, alive);
  const live = boosted.filter((hit) => !hit.legacy).sort(byScore);
  // The legacy table is a fallback corpus, not the library: measured on a real
  // library it outranks every live document row on any text query, so it may take
  // at most half a page and never the whole page.
  const fallback = boosted.filter((hit) => hit.legacy).sort(byScore);
  const fallbackSlots = Math.ceil(limit / 2);
  const merged: AssetHit[] = [];
  let liveIndex = 0;
  let fallbackIndex = 0;
  let fallbackTaken = 0;
  while (merged.length < limit && (liveIndex < live.length || fallbackIndex < fallback.length)) {
    const liveHit = live[liveIndex];
    const fallbackHit = fallbackTaken < fallbackSlots ? fallback[fallbackIndex] : undefined;
    if (fallbackHit && (!liveHit || fallbackHit.score > liveHit.score)) {
      merged.push(fallbackHit);
      fallbackIndex += 1;
      fallbackTaken += 1;
    } else if (liveHit) {
      merged.push(liveHit);
      liveIndex += 1;
    } else {
      break;
    }
  }
  return {
    query: text,
    embeddingSpace: ctx.space,
    reranked: false,
    hits: merged,
    hiddenMissing,
  };
}

export async function listAssets(options: ListOptions = {}): Promise<{ count: number; assets: AssetHit[] }> {
  const ctx = context(options.config);
  const limit = Math.max(1, Math.min(5_000, Math.floor(toNumber(options.limit, 200))));
  const kind = String(options.kind || "").trim().toLowerCase();
  if (kind && !["video", "image", "document"].includes(kind)) throw new Error(`不支持的素材类型：${kind}`);
  const library = String(options.library || "").trim() ? path.resolve(String(options.library)) : "";
  const rows = await listLocalVectors(ctx.direct.assetIndex, ctx.direct, {
    filter: metadataFilter({
      record_type: ASSET_RECORD_TYPE,
      embedding_space: ctx.space,
      ...(kind ? { file_kind: kind } : {}),
      ...(library ? { library_root: library } : {}),
    }),
    returnData: false,
  });
  const uniqueAssets = new Map<string, AssetHit>();
  for (const row of rows) {
    const asset = assetHit(row, false);
    const key = asset.kind === "document" && asset.path ? asset.path : asset.id;
    const held = uniqueAssets.get(key);
    if (!held || asset.modifiedAt > held.modifiedAt || asset.score > held.score) uniqueAssets.set(key, asset);
  }
  const assets = [...uniqueAssets.values()]
    .filter((asset) => options.includeMissing || assetFileExists(asset.path))
    .sort((left, right) => right.modifiedAt - left.modifiedAt || left.path.localeCompare(right.path))
    .slice(0, limit);
  return { count: assets.length, assets };
}

/**
 * Remove rows whose asset file no longer exists.
 *
 * A library root is a mirror of a directory a person also edits by hand: renders get
 * re-exported, scratch cuts get deleted, whole projects get moved. Without this the
 * index keeps serving hits for files that are gone, and the only repair was a full
 * rescan of every root.
 */
export async function pruneMissingAssets(options: PruneOptions = {}): Promise<PruneResult> {
  const ctx = context(options.config ?? loadConfig());
  const library = String(options.library || "").trim() ? path.resolve(String(options.library)) : "";
  const rows = await listLocalVectors(ctx.direct.assetIndex, ctx.direct, {
    filter: metadataFilter({
      record_type: ASSET_RECORD_TYPE,
      embedding_space: ctx.space,
      ...(library ? { library_root: library } : {}),
    }),
    returnData: false,
  });
  const missing = new Map<string, { path: string; kind: string; rows: number; libraryPath: string }>();
  const doomed: string[] = [];
  for (const row of rows) {
    const hit = assetHit(row, false);
    if (assetFileExists(hit.path)) continue;
    const entry = missing.get(hit.path) || { path: hit.path, kind: hit.kind, rows: 0, libraryPath: hit.libraryPath };
    entry.rows += 1;
    missing.set(hit.path, entry);
    doomed.push(String(row.key));
  }
  if (!options.dryRun) {
    for (let offset = 0; offset < doomed.length; offset += 500) {
      await deleteLocalVectors(ctx.direct.assetIndex, doomed.slice(offset, offset + 500), ctx.direct);
    }
    invalidateStatus();
  }
  return {
    ok: true,
    dryRun: Boolean(options.dryRun),
    embeddingSpace: ctx.space,
    scanned: rows.length,
    missingAssets: [...missing.values()].sort((left, right) => right.rows - left.rows || left.path.localeCompare(right.path)),
    removedRows: options.dryRun ? 0 : doomed.length,
  };
}

/**
 * Register a library directory.
 *
 * `scan: false` only writes the root into the configuration. A one-shot CLI cannot
 * watch a scan finish - the queue lives in the resident process - so it either hands
 * the work to the server or scans inline with scanLibrary and progress reporting.
 */
export async function addLibrary(
  assetPath: string,
  options: { config?: ReturnType<typeof loadConfig>; scan?: boolean; kinds?: AssetKind[] } = {},
): Promise<{ ok: true; library: AssetLibraryEntry; scanStarted: boolean }> {
  const target = path.resolve(String(assetPath || "").trim());
  if (!target || target === path.resolve(".")) throw new Error("需要素材库目录路径");
  if (!fs.existsSync(target)) throw new AssetAccessError(`素材库目录不存在：${target}`, 400);
  if (!fs.statSync(target).isDirectory()) throw new AssetAccessError(`不是目录：${target}`, 400);
  const next = options.config ?? loadConfig();
  const roots = libraryRoots(next);
  const selectedKinds = options.kinds === undefined
    ? libraryKinds(next, target)
    : normalizedAssetKinds(options.kinds, []);
  if (!selectedKinds.length) throw new AssetAccessError("至少选择一种要追踪的文件类型", 400);
  // Two roots that contain each other would index the same file twice and leave
  // it ambiguous which one owns the row, so only disjoint directories are allowed.
  const enclosing = roots.find((root) => target.startsWith(`${root}${path.sep}`));
  if (enclosing) throw new AssetAccessError(`这个目录在已登记的素材目录之内：${enclosing}，它的文件由那个目录负责索引`, 409);
  const contained = roots.find((root) => root.startsWith(`${target}${path.sep}`));
  if (contained) throw new AssetAccessError(`这个目录已经包含登记的素材目录：${contained}，重复添加会重复索引，请选择更具体的一层`, 409);
  if (!roots.includes(target) || options.kinds !== undefined) {
    const nextRoots = roots.includes(target) ? roots : [...roots, target];
    const rootKinds = next.library?.rootKinds && typeof next.library.rootKinds === "object" && !Array.isArray(next.library.rootKinds)
      ? { ...next.library.rootKinds }
      : {};
    rootKinds[target] = selectedKinds;
    next.library = { ...next.library, roots: nextRoots, rootKinds };
    writeConfig(next);
    invalidateStatus();
  }
  if (options.scan !== false) queueScan(target, next);
  const status = await librariesStatus(next);
  const library = status.libraries.find((entry) => entry.id === target);
  if (!library) throw new Error(`素材库注册失败：${target}`);
  return { ok: true, library, scanStarted: jobs.get(target)?.state === "queued" || jobs.get(target)?.state === "running" };
}

/** Unregister a library; vectors are only purged when purge is requested. */
export async function removeLibrary(assetPath: string, options: { purge?: boolean; config?: ReturnType<typeof loadConfig> } = {}): Promise<{ ok: true; removed: string; purged: boolean; scanCancelled: boolean }> {
  const target = path.resolve(String(assetPath || "").trim());
  const config = options.config ?? loadConfig();
  const roots = libraryRoots(config);
  if (!roots.includes(target)) throw new AssetAccessError(`素材库未注册：${target}`, 400);
  const cancellation = cancelScan(target);
  const activeRun = activeRuns.get(target);
  if (activeRun) await activeRun.catch(() => undefined);
  const rootKinds = config.library?.rootKinds && typeof config.library.rootKinds === "object" && !Array.isArray(config.library.rootKinds)
    ? { ...config.library.rootKinds }
    : {};
  delete rootKinds[target];
  const next = { ...config, library: { ...config.library, roots: roots.filter((root) => root !== target), rootKinds } };
  writeConfig(next);
  let purged = false;
  if (options.purge) {
    const ctx = context(next);
    const rows = await listLocalVectors(ctx.direct.assetIndex, ctx.direct, {
      filter: metadataFilter({ record_type: ASSET_RECORD_TYPE, embedding_space: ctx.space, library_root: target }),
      returnData: false,
    });
    const keys = rows.map((row) => row.key);
    for (let offset = 0; offset < keys.length; offset += 500) {
      await deleteLocalVectors(ctx.direct.assetIndex, keys.slice(offset, offset + 500), ctx.direct);
    }
    invalidateStatus();
    purged = true;
  }
  jobs.delete(target);
  return { ok: true, removed: target, purged, scanCancelled: Boolean(cancellation.cancelled.length || cancellation.stopping.length) };
}

/**
 * Guard serving local files: the real path must live inside a registered library
 * root, and only indexed media extensions may be read back.
 */
export function resolveAssetForServing(assetPath: string, config = loadConfig()): ServedAsset {
  const requested = String(assetPath || "").trim();
  if (!requested) throw new AssetAccessError("需要素材路径");
  if (!path.isAbsolute(requested)) throw new AssetAccessError("素材路径必须是绝对路径");
  const extension = path.extname(requested).toLowerCase();
  const kind = assetKindOf(extension);
  if (!kind) throw new AssetAccessError(`不支持的素材类型：${extension || "无扩展名"}`);
  const ctx = context(config);
  if (!ctx.roots.length) throw new AssetAccessError("尚未注册素材库目录");
  let real = "";
  try {
    real = fs.realpathSync(path.resolve(requested));
  } catch {
    throw new AssetAccessError(`素材文件不存在：${requested}`, 404);
  }
  const stat = fs.statSync(real);
  if (!stat.isFile()) throw new AssetAccessError(`不是文件：${requested}`, 404);
  const inside = ctx.roots.some((root) => {
    let realRoot = "";
    try { realRoot = fs.realpathSync(root); } catch { return false; }
    return real === realRoot || real.startsWith(`${realRoot}${path.sep}`);
  });
  if (!inside) throw new AssetAccessError("素材不在已注册的素材库目录内");
  return { path: real, name: path.basename(real), kind, extension, contentType: assetContentType(extension), size: stat.size };
}

/** Preview bytes for the dashboard: images pass through, video frames are extracted. */
export async function resolveAssetPreview(assetPath: string, atSeconds: number, config = loadConfig()): Promise<{ path: string; contentType: string }> {
  const asset = resolveAssetForServing(assetPath, config);
  if (asset.kind === "image") return { path: asset.path, contentType: asset.contentType };
  if (asset.kind !== "video") throw new AssetAccessError(`${asset.extension} 不支持预览抽帧`);
  const ctx = context(config);
  const frame = await extractFrame(asset.path, atSeconds, {
    previewDir: assetPreviewDirectory(ctx.direct),
    libraryRoots: ctx.roots,
  });
  return { path: frame, contentType: "image/jpeg" };
}

/* ------------------------------------------------------------------ *
 * Text assets: read and edit a note in place, then re-embed it.
 * ------------------------------------------------------------------ */

/**
 * Refuse anything that is not plain UTF-8 text rather than guess: a GBK note or a
 * binary file masquerading as `.txt` would come back mangled from a save.
 */
function editableTextOf(assetPath: string): string {
  let buffer: Buffer;
  try {
    buffer = fs.readFileSync(assetPath);
  } catch {
    throw new AssetAccessError(`无法读取文本素材：${assetPath}`);
  }
  if (buffer.length > MAX_EDITED_BYTES) {
    throw new AssetAccessError(`文本素材过大（${buffer.length} 字节），上限 ${MAX_EDITED_BYTES} 字节`);
  }
  if (buffer.includes(0)) throw new AssetAccessError("文本素材包含空字节，无法编辑");
  const text = buffer.toString("utf8");
  if (text.includes("�")) throw new AssetAccessError("文本素材不是 UTF-8 编码，无法编辑");
  // A browser textarea submits LF, so normalising here keeps read and write honest.
  return text.replace(/\r\n?/g, "\n");
}

/** Rows currently holding this one asset, whatever segment index they were written with. */
async function assetRows(ctx: AssetContext, assetPath: string): Promise<Array<{ key: string }>> {
  return listLocalVectors(ctx.direct.assetIndex, ctx.direct, {
    filter: metadataFilter({
      record_type: ASSET_RECORD_TYPE,
      embedding_space: ctx.space,
      asset_path: assetPath,
    }),
    returnData: false,
  });
}

export interface AssetDocument {
  path: string;
  name: string;
  extension: string;
  libraryPath: string;
  text: string;
  bytes: number;
  modifiedAt: number;
  /** Rows this file has in the asset index right now; 0 means it was never indexed. */
  vectors: number;
}

export interface AssetDocumentWrite {
  ok: true;
  path: string;
  name: string;
  bytes: number;
  modifiedAt: number;
  embeddingSpace: string;
  vectors: number;
  removed: number;
  /** Media rows whose copied note excerpt was refreshed alongside this file. */
  previews: number;
  /** What happened to the vector side, so a caller can tell "saved" from "saved and recallable". */
  notice: string;
}

/** The guard for text editing: indexed document, editable extension, inside a registered root. */
export function resolveAssetForEditing(assetPath: string, config = loadConfig()): ServedAsset {
  const asset = resolveAssetForServing(assetPath, config);
  if (!EDITABLE_ASSET_EXTENSIONS.has(asset.extension)) {
    throw new AssetAccessError(`${asset.extension} 不支持文本编辑，只有 ${[...EDITABLE_ASSET_EXTENSIONS].join(" ")} 可以`);
  }
  return asset;
}

/**
 * Media rows copy the note text beside them for their cards, so an edited sidecar
 * would otherwise leave a clip quoting words it no longer has. Their vectors
 * describe pixels rather than that text, so they are written back untouched with
 * only the copied preview replaced.
 */
async function refreshSidecarPreviews(
  ctx: ReturnType<typeof context>,
  partner: string,
  preview: string,
): Promise<number> {
  if (!partner) return 0;
  const rows = await listLocalVectors(ctx.direct.assetIndex, ctx.direct, {
    filter: metadataFilter({ record_type: "asset", embedding_space: ctx.space, asset_path: partner }),
    returnData: true,
  });
  if (!rows.length) return 0;
  const refreshed = rows.flatMap((row) => {
    const vector = row.data?.float32;
    if (!vector?.length) return [];
    return [{
      key: row.key,
      data: { float32: vector },
      metadata: { ...(row.metadata || {}), text_preview: preview },
    }];
  });
  if (!refreshed.length) return 0;
  await putLocalVectors(ctx.direct.assetIndex, refreshed, ctx.direct);
  return refreshed.length;
}

export async function readAssetDocument(assetPath: string, config = loadConfig()): Promise<AssetDocument> {
  const asset = resolveAssetForEditing(assetPath, config);
  const ctx = context(config);
  const text = editableTextOf(asset.path);
  const stat = fs.statSync(asset.path);
  const rows = await assetRows(ctx, indexedAssetPath(asset.path, ctx.roots));
  return {
    path: asset.path,
    name: asset.name,
    extension: asset.extension,
    libraryPath: libraryRootOf(asset.path, ctx.roots),
    text,
    bytes: stat.size,
    modifiedAt: Math.round(stat.mtimeMs),
    vectors: rows.length,
  };
}

/**
 * Save a text asset in place and refresh its vector.
 *
 * The file is the source of truth: the write is atomic (a hidden temporary that the
 * walker skips, then a rename on the same filesystem), and the deterministic
 * per-path vector key means the recall row is replaced rather than duplicated.
 * Stale rows for the same file are dropped, so a note that shrank cannot keep
 * answering searches with text it no longer contains.
 */
export async function writeAssetDocument(
  assetPath: string,
  content: string,
  options: { config?: ReturnType<typeof loadConfig> } = {},
): Promise<AssetDocumentWrite> {
  const config = options.config ?? loadConfig();
  const asset = resolveAssetForEditing(assetPath, config);
  if (typeof content !== "string") throw new AssetAccessError("需要文本内容");
  const text = content.replace(/\r\n?/g, "\n");
  const buffer = Buffer.from(text, "utf8");
  if (buffer.length > MAX_EDITED_BYTES) {
    throw new AssetAccessError(`文本过长（${buffer.length} 字节），上限 ${MAX_EDITED_BYTES} 字节`);
  }
  if (buffer.includes(0)) throw new AssetAccessError("文本内容包含空字节");

  const ctx = context(config);
  const before = fs.statSync(asset.path);
  const temporary = path.join(path.dirname(asset.path), `.${path.basename(asset.path)}.indexed-tmp`);
  try {
    // Create permissive-then-restore: a read-only note (mode 0444) could not be
    // written at all if the temporary inherited it.
    fs.writeFileSync(temporary, buffer, { mode: 0o600 });
    fs.chmodSync(temporary, before.mode & 0o777);
    fs.renameSync(temporary, asset.path);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw new AssetAccessError(`写入失败：${String((error as Error)?.message || error)}`);
  }

  const after = fs.statSync(asset.path);
  const fingerprint = `${after.size}:${Math.round(after.mtimeMs)}`;
  const indexedPath = indexedAssetPath(asset.path, ctx.roots);
  const written = await assetRows(ctx, indexedPath);
  const mediaPartner = mediaPartnerOf(asset.path);
  const partner = mediaPartner ? indexedAssetPath(mediaPartner, ctx.roots) : "";
  const trimmed = text.trim();
  const preview = trimmed.slice(0, PREVIEW_CHARACTERS);

  let vectors = 0;
  let previews = 0;
  let notice = "";
  if (trimmed) {
    try {
      const planned = chunkDocumentText(trimmed);
      const rows: Array<{ key: string; data: { float32: number[] }; metadata: Record<string, unknown> }> = [];
      for (const chunk of planned.chunks) {
        rows.push({
          key: await assetKey(indexedPath, chunk.index, ctx.space),
          data: { float32: await embedDocument(chunk.text, ctx.direct) },
          metadata: assetMetadata({
            assetPath: indexedPath,
            assetName: asset.name,
            kind: "document",
            extension: asset.extension,
            size: after.size,
            modifiedAt: Math.round(after.mtimeMs),
            libraryRoot: libraryRootOf(asset.path, ctx.roots),
            fingerprint,
            sidecarOf: partner,
            textPreview: chunk.text.slice(0, PREVIEW_CHARACTERS),
            embeddingModel: ctx.direct.embeddingModel,
            space: ctx.space,
            embeddingInputVersion: assetEmbeddingInputVersion("document", ctx.direct),
            overrides: {
              segment_index: chunk.index,
              character_start: chunk.startCharacter,
              character_end: chunk.endCharacter,
              document_chunk_version: DOCUMENT_CHUNK_VERSION,
              embedding_basis: partner ? "sidecar_text_v1" : "document_text_v1",
            },
          }),
        });
      }
      await putLocalVectors(ctx.direct.assetIndex, rows, ctx.direct);
      vectors = rows.length;
      notice = planned.truncated
        ? `已按段落更新 ${vectors} 个片段；超长尾部未索引`
        : `已按段落更新 ${vectors} 个片段`;
    } catch (error) {
      // The file is saved either way; the new fingerprint makes the next scan retry.
      notice = `文件已保存，但重新嵌入失败：${String((error as Error)?.message || error)}`;
    }
  } else {
    notice = "内容为空，已移除该素材的向量";
  }

  const freshKeys = new Set(vectors ? await Promise.all(Array.from({ length: vectors }, (_, index) => assetKey(indexedPath, index, ctx.space))) : []);
  const doomed = written.map((row) => row.key).filter((value) => !freshKeys.has(value));
  let removed = 0;
  if (doomed.length) {
    await deleteLocalVectors(ctx.direct.assetIndex, doomed, ctx.direct);
    removed = doomed.length;
  }
  // Cards for the media file beside a note quote its first lines, so that copy is
  // refreshed even when the note went empty. It never blocks the save itself.
  try {
    previews = await refreshSidecarPreviews(ctx, partner, preview);
  } catch (error) {
    notice += `；素材摘要未刷新：${String((error as Error)?.message || error)}`;
  }
  invalidateStatus();
  return {
    ok: true,
    path: asset.path,
    name: asset.name,
    bytes: after.size,
    modifiedAt: Math.round(after.mtimeMs),
    embeddingSpace: ctx.space,
    vectors,
    removed,
    previews,
    notice,
  };
}


export interface DirectoryEntry {
  name: string;
  path: string;
  /** This exact directory is already registered as an asset root. */
  registered: boolean;
  /** A registered root lives somewhere below this directory. */
  containsLibrary: boolean;
}

export interface DirectoryBrowseResult {
  path: string;
  parent: string | null;
  home: string;
  roots: string[];
  /** Kinds that a newly selected root will track, or this root's saved override. */
  kinds: AssetKind[];
  /** Format families shown under the simple three-kind picker. */
  documentFormats: ReturnType<typeof documentFormatCapabilities>;
  /** Display names for `roots`, parallel by index, disambiguating generic folder names. */
  rootLabels: string[];
  entries: DirectoryEntry[];
  truncated: boolean;
}

const BROWSE_LIMIT = 400;

/**
 * Read-only walk of one server directory, for the Dashboard's directory picker.
 * People point Indexed at footage by clicking through volumes, not by retyping a
 * path they read off a terminal. Only directory names are returned: no files, no
 * sizes, no contents.
 */
/**
 * A leaf called `assets` tells you nothing about which project it belongs to, so a
 * generic name gets qualified with its parent. Display-only: never a path.
 */
function directoryLabel(root: string): string {
  const leaf = path.basename(root);
  const parent = path.basename(path.dirname(root));
  const generic = /^(assets?|media|videos?|files?|renders?|output|out|clips?|素材)$/i;
  return parent && parent !== leaf && generic.test(leaf) ? `${parent}/${leaf}` : leaf;
}

/** Deepest directory that contains every registered root, so the picker opens nearby. */
function commonAncestor(roots: string[]): string {
  if (!roots.length) return "";
  let segments = (roots[0] || "").split(path.sep);
  for (const root of roots) {
    const current = root.split(path.sep);
    let index = 0;
    while (index < segments.length && current[index] === segments[index]) index += 1;
    segments = segments.slice(0, index);
  }
  const ancestor = segments.join(path.sep) || path.sep;
  return fs.existsSync(ancestor) && fs.statSync(ancestor).isDirectory() ? ancestor : path.parse(roots[0] || path.sep).root;
}

export function browseAssetDirectories(options: {
  path?: string;
  limit?: number;
  config?: ReturnType<typeof loadConfig>;
} = {}): DirectoryBrowseResult {
  const config = options.config || loadConfig();
  const ctx = context(config);
  const requested = String(options.path || "").trim();
  // Land where the libraries already are: someone adding a directory is looking
  // for its sibling, not for their home folder.
  const fallback = commonAncestor(ctx.roots) || (fs.existsSync(os.homedir()) ? os.homedir() : path.parse(process.cwd()).root);
  const target = requested ? path.resolve(requested) : fallback;
  if (!fs.existsSync(target) || !fs.statSync(target).isDirectory()) {
    throw new AssetAccessError(`目录不存在：${target}`, 400);
  }
  const limit = Math.max(1, Math.min(1000, Math.floor(options.limit || BROWSE_LIMIT)));
  let dirents: fs.Dirent[] = [];
  try {
    dirents = fs.readdirSync(target, { withFileTypes: true });
  } catch {
    throw new AssetAccessError(`无法读取目录：${target}`, 403);
  }
  const entries: DirectoryEntry[] = [];
  let truncated = false;
  for (const entry of dirents.slice().sort((left, right) => left.name.localeCompare(right.name))) {
    if (!entry.isDirectory() || entry.name.startsWith(".") || SKIPPED_DIRECTORIES.has(entry.name)) continue;
    if (entries.length >= limit) {
      truncated = true;
      break;
    }
    const child = path.join(target, entry.name);
    entries.push({
      name: entry.name,
      path: child,
      registered: ctx.roots.includes(child),
      containsLibrary: ctx.roots.some((root) => root.startsWith(`${child}${path.sep}`)),
    });
  }
  return {
    path: target,
    parent: target === path.dirname(target) ? null : path.dirname(target),
    home: os.homedir(),
    roots: ctx.roots,
    kinds: libraryKinds(config, target),
    documentFormats: documentFormatCapabilities(),
    rootLabels: ctx.roots.map(directoryLabel),
    entries,
    truncated,
  };
}

export interface LibraryCoverage {
  path: string;
  name: string;
  kinds: AssetKind[];
  /** Indexable files currently on disk. */
  files: number;
  /** Files whose indexed row still matches their size and mtime. */
  cached: number;
  /** Files the index has never seen. */
  pending: number;
  /** Files whose indexed row is behind the copy on disk. */
  stale: number;
  /** Indexed files no longer on disk, which search now hides. */
  missing: number;
  byKind: Record<AssetKind, { files: number; cached: number; pending: number; stale: number }>;
  truncated: boolean;
  state: AssetJobState;
}

export interface CoverageResult {
  embeddingSpace: string;
  kinds: string[];
  totals: { files: number; cached: number; pending: number; stale: number; missing: number };
  totalsByKind: Record<AssetKind, { files: number; cached: number; pending: number; stale: number }>;
  ingestEstimate: ReturnType<typeof estimateIngestPerformance>;
  libraries: LibraryCoverage[];
}

interface CoverageCache {
  at: number;
  value: LibraryCoverage;
}

const coverageCache = new Map<string, CoverageCache>();
const COVERAGE_IDLE_MILLISECONDS = 30_000;
const COVERAGE_BUSY_MILLISECONDS = 5_000;

function emptyKindCoverage(): Record<AssetKind, { files: number; cached: number; pending: number; stale: number }> {
  return {
    video: { files: 0, cached: 0, pending: 0, stale: 0 },
    image: { files: 0, cached: 0, pending: 0, stale: 0 },
    document: { files: 0, cached: 0, pending: 0, stale: 0 },
  };
}

/**
 * Per-directory accounting for the settings screen: what the disk holds versus what
 * the index has, so a background scan is legible without showing anyone the asset list.
 * Walking is the expensive part, so each root is cached - and refreshed often only
 * while that same root is scanning.
 */
export async function libraryCoverage(options: {
  library?: string;
  refresh?: boolean;
  config?: ReturnType<typeof loadConfig>;
} = {}): Promise<CoverageResult> {
  const config = options.config || loadConfig();
  const ctx = context(config);
  const requested = String(options.library || "").trim();
  const roots = requested ? [path.resolve(requested)] : ctx.roots;
  const indexed = await assetsByRoot(config);
  const libraries: LibraryCoverage[] = [];
  for (const root of roots) {
    const trackedKinds = libraryKinds(config, root);
    const kinds = new Set(trackedKinds);
    const state: AssetJobState = jobs.get(root)?.state || "idle";
    const busy = state === "queued" || state === "running";
    const key = `${localVectorDatabasePath(ctx.direct)}|${ctx.space}|${root}`;
    const held = coverageCache.get(key);
    const ttl = busy ? COVERAGE_BUSY_MILLISECONDS : COVERAGE_IDLE_MILLISECONDS;
    // Polling the Dashboard must not walk tens of thousands of files every five
    // seconds while the model is busy. The active job is the progress source.
    if (!options.refresh && held && (busy || Date.now() - held.at < ttl)) {
      libraries.push({ ...held.value, state });
      continue;
    }
    const bucket = indexed.get(root) || new Map<string, string>();
    let value: LibraryCoverage;
    if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
      value = {
        path: root,
        name: directoryLabel(root),
        kinds: trackedKinds,
        files: 0,
        cached: 0,
        pending: 0,
        stale: 0,
        missing: bucket.size,
        byKind: emptyKindCoverage(),
        truncated: false,
        state,
      };
    } else {
      const walked = walkLibrary(root, kinds, ctx.settings.maxFilesPerLibrary);
      const discovered = walked.files;
      const seen = new Set<string>();
      let cached = 0;
      let pending = 0;
      let stale = 0;
      const byKind = emptyKindCoverage();
      for (const file of discovered) {
        seen.add(file.path);
        byKind[file.kind].files += 1;
        const fingerprint = bucket.get(file.path);
        if (!fingerprint) {
          pending += 1;
          byKind[file.kind].pending += 1;
        } else if (fingerprint === file.fingerprint) {
          cached += 1;
          byKind[file.kind].cached += 1;
        } else {
          stale += 1;
          byKind[file.kind].stale += 1;
        }
      }
      // A capped walk cannot prove what is absent, so dead rows stay uncounted.
      const truncated = walked.truncated;
      value = {
        path: root,
        name: path.basename(root),
        kinds: trackedKinds,
        files: discovered.length,
        cached,
        pending,
        stale,
        missing: truncated ? 0 : [...bucket.keys()].filter((assetPath) => !seen.has(assetPath)).length,
        byKind,
        truncated,
        state,
      };
    }
    coverageCache.set(key, { at: Date.now(), value });
    libraries.push(value);
  }
  const totals = libraries.reduce(
    (sum, item) => ({
      files: sum.files + item.files,
      cached: sum.cached + item.cached,
      pending: sum.pending + item.pending,
      stale: sum.stale + item.stale,
      missing: sum.missing + item.missing,
    }),
    { files: 0, cached: 0, pending: 0, stale: 0, missing: 0 },
  );
  const totalsByKind = libraries.reduce((sum, item) => {
    for (const kind of ASSET_KIND_ORDER) {
      sum[kind].files += item.byKind[kind].files;
      sum[kind].cached += item.byKind[kind].cached;
      sum[kind].pending += item.byKind[kind].pending;
      sum[kind].stale += item.byKind[kind].stale;
    }
    return sum;
  }, emptyKindCoverage());
  const ingestEstimate = estimateIngestPerformance({
    config,
    pendingByKind: Object.fromEntries(ASSET_KIND_ORDER.map((kind) => [
      kind,
      totalsByKind[kind].pending + totalsByKind[kind].stale,
    ])) as KindCounters,
  });
  return { embeddingSpace: ctx.space, kinds: ctx.settings.kinds, totals, totalsByKind, ingestEstimate, libraries };
}
