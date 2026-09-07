// @ts-nocheck -- compatibility port preserving the existing on-disk schema.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  APPLE_EMBEDDING_DEFAULT_DIMENSION,
  PRODUCT_DEFAULTS,
  appleEmbeddingModel,
  readAppleExecutionMode,
  appleExecutionModeIssue,
  requireAppleExecutionMode,
} from "@indexed/contracts";

export const ASSET_KINDS = ["video", "image", "document"];

/** Resolve the optional single-root runtime directory used by local deployments. */
export function indexedHomePath() {
  const configured = String(process.env.INDEXED_HOME || "").trim();
  if (!configured) return "";
  const expanded = configured === "~"
    ? os.homedir()
    : configured.startsWith("~/")
      ? path.join(os.homedir(), configured.slice(2))
      : configured;
  return path.resolve(expanded);
}

export const DEFAULT_CONFIG = {
  version: 1,
  activeProfile: "default",
  server: { ...PRODUCT_DEFAULTS.server },
  // library.roots is the single source of truth for the asset library; a root id
  // is always its absolute path.
  library: {
    roots: [],
    // Optional per-root overrides preserve the stable string-only roots list.
    rootKinds: {},
    autoScan: true,
    scanIntervalSeconds: 900,
    maxAssetsPerScan: 400,
    // File discovery and embedding are separate budgets. Large libraries may
    // raise this without forcing every discovered file through the model.
    maxFilesPerLibrary: 100_000,
    kinds: [...ASSET_KINDS],
  },
  profiles: {
    default: {
      label: "Default",
      spaceId: "",
      embedding: {
        provider: "remote",
        baseUrl: "",
        apiKey: "",
        model: "",
        dimension: PRODUCT_DEFAULTS.vectorDimension,
        inputStyle: "auto",
        native: {
          binary: "",
          modelPackage: "",
          coreMLCache: "",
          executionMode: "b",
          mode: "fast",
          visionCompute: "ane",
          privateANE: {
            videoDownProjection: "q8",
            videoPipeline: 2,
            sequenceLength: 2112,
            mlpFraction: 0.75,
            mlpVariant: 8,
            mlpMaxLayers: 24,
            recurrenceProfile: "",
            recurrenceBlockSize: 8,
            recurrenceLayerSlots: [0],
            recurrenceQueryScale: 4096,
            recurrenceMaxTokens: 8192,
            recurrenceIODtype: "fp16",
            recurrenceVerifyReference: false,
          },
          maxQueuedRequests: 16,
          startupTimeoutSeconds: 300,
          maintenanceTimeoutSeconds: 600,
          autoRestart: true,
          maxRestarts: 3,
        },
      },
      reranker: {
        enabled: false,
        baseUrl: "",
        model: "",
        candidates: 20,
      },
      video: { chunkSeconds: 30, maxChunkSeconds: 60, fps: 2, width: 1280, maxSegments: 240 },
      storage: {
        provider: "local",
        path: "",
        region: PRODUCT_DEFAULTS.ossRegion,
        accountId: "",
        bucket: "",
        visualIndex: "",
        transcriptIndex: "",
        documentIndex: "local-documents",
        // The asset library writes a new table; the legacy documentIndex above stays read-only.
        assetIndex: "library-assets",
        accessKeyId: "",
        accessKeySecret: "",
        securityToken: "",
      },
    },
  },
};

export function configPath() {
  const home = indexedHomePath();
  return path.resolve(
    process.env.INDEXED_CONFIG
      || (home ? path.join(home, "config", "config.json") : path.join(os.homedir(), ".config", "indexed", "config.json")),
  );
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function merge(base, incoming) {
  if (!incoming || typeof incoming !== "object" || Array.isArray(incoming)) return clone(base);
  const output = clone(base);
  for (const [key, value] of Object.entries(incoming)) {
    if (value && typeof value === "object" && !Array.isArray(value)
      && output[key] && typeof output[key] === "object" && !Array.isArray(output[key])) {
      output[key] = merge(output[key], value);
    } else {
      output[key] = value;
    }
  }
  return output;
}

function wholeNumber(value: unknown, fallback: number) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : fallback;
}

function positiveNumber(value: unknown, fallback: number) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function normalizeLibrary(value: unknown) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
  const roots = Array.isArray(source.roots)
    ? [...new Set(source.roots.map((item) => String(item ?? "").trim()).filter(Boolean))]
    : [];
  const kinds = Array.isArray(source.kinds)
    ? [...new Set(source.kinds.map((item) => String(item ?? "").trim().toLowerCase()).filter((item) => ASSET_KINDS.includes(item)))]
    : [];
  const fallbackKinds = kinds.length ? kinds : [...ASSET_KINDS];
  const rawRootKinds = source.rootKinds && typeof source.rootKinds === "object" && !Array.isArray(source.rootKinds)
    ? source.rootKinds
    : {};
  const rootKinds = Object.fromEntries(roots.map((root) => {
    const configured = Array.isArray(rawRootKinds[root])
      ? [...new Set(rawRootKinds[root].map((item) => String(item ?? "").trim().toLowerCase()).filter((item) => ASSET_KINDS.includes(item)))]
      : [];
    return [root, configured.length ? configured : fallbackKinds];
  }));
  return {
    roots,
    rootKinds,
    autoScan: source.autoScan === undefined ? true : Boolean(source.autoScan),
    scanIntervalSeconds: wholeNumber(source.scanIntervalSeconds, DEFAULT_CONFIG.library.scanIntervalSeconds),
    maxAssetsPerScan: wholeNumber(source.maxAssetsPerScan, DEFAULT_CONFIG.library.maxAssetsPerScan),
    maxFilesPerLibrary: wholeNumber(source.maxFilesPerLibrary, DEFAULT_CONFIG.library.maxFilesPerLibrary),
    kinds: fallbackKinds,
  };
}

function normalizeVideoProfile(value: unknown) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
  const defaults = DEFAULT_CONFIG.profiles.default.video;
  const video = {
    chunkSeconds: wholeNumber(source.chunkSeconds, defaults.chunkSeconds),
    maxChunkSeconds: wholeNumber(source.maxChunkSeconds, defaults.maxChunkSeconds),
    fps: positiveNumber(source.fps, defaults.fps),
    width: wholeNumber(source.width, defaults.width),
    maxSegments: wholeNumber(source.maxSegments, defaults.maxSegments),
  };
  if (video.maxChunkSeconds < video.chunkSeconds) video.maxChunkSeconds = video.chunkSeconds;
  return video;
}

export function normalizeConfig(value = {}) {
  const output = merge(DEFAULT_CONFIG, value);
  if (value.profiles && Object.keys(value.profiles).length) output.profiles = clone(value.profiles);
  output.version = 1;
  output.library = normalizeLibrary(value.library);
  output.profiles ||= {};
  if (!Object.keys(output.profiles).length) output.profiles.default = clone(DEFAULT_CONFIG.profiles.default);
  if (!output.profiles[output.activeProfile]) output.activeProfile = Object.keys(output.profiles)[0];
  output.server.port = Number(output.server.port || PRODUCT_DEFAULTS.server.port);
  for (const [id, source] of Object.entries(output.profiles)) {
    const profile = merge(DEFAULT_CONFIG.profiles.default, source);
    const sourceNative = source?.embedding?.native || {};
    profile.label ||= id;
    profile.embedding.provider = String(profile.embedding.provider || "remote").trim().toLowerCase();
    if (!new Set(["remote", "apple-native"]).has(profile.embedding.provider)) {
      profile.embedding.provider = "remote";
    }
    profile.embedding.dimension = Number(profile.embedding.dimension || 0);
    if (profile.embedding.provider === "apple-native") {
      if (!Number.isFinite(profile.embedding.dimension) || profile.embedding.dimension <= 0) {
        profile.embedding.dimension = APPLE_EMBEDDING_DEFAULT_DIMENSION;
      }
      profile.embedding.dimension = Math.floor(profile.embedding.dimension);
      profile.embedding.baseUrl = "";
      profile.embedding.model = appleEmbeddingModel(profile.embedding.dimension);
      profile.embedding.inputStyle = "wemm";
    }
    profile.embedding.native ||= clone(DEFAULT_CONFIG.profiles.default.embedding.native);
    profile.embedding.native.executionMode = readAppleExecutionMode(sourceNative);
    profile.embedding.native.executionModeIssue = appleExecutionModeIssue(sourceNative);
    if (!profile.embedding.native.executionModeIssue) {
      profile.embedding.native.mode = "fast";
      profile.embedding.native.visionCompute = profile.embedding.native.executionMode === "a" ? "gpu" : "ane";
      for (const key of ["decoderSegment", "decoderBundles", "decoderLoading", "decoderMinimumTokens"]) delete profile.embedding.native[key];
    }
    profile.embedding.native.privateANE ||= clone(DEFAULT_CONFIG.profiles.default.embedding.native.privateANE);
    profile.embedding.native.privateANE.videoDownProjection = profile.embedding.native.privateANE.videoDownProjection === "fp16" ? "fp16" : "q8";
    profile.embedding.native.privateANE.videoPipeline = profile.embedding.native.privateANE.videoPipeline === 1 ? 1 : 2;
    profile.embedding.native.privateANE.sequenceLength = wholeNumber(
      profile.embedding.native.privateANE.sequenceLength, 2112,
    );
    profile.embedding.native.privateANE.mlpFraction = positiveNumber(
      profile.embedding.native.privateANE.mlpFraction, 0.75,
    );
    profile.embedding.native.privateANE.mlpVariant = wholeNumber(
      profile.embedding.native.privateANE.mlpVariant, 8,
    );
    profile.embedding.native.privateANE.mlpMaxLayers = wholeNumber(
      profile.embedding.native.privateANE.mlpMaxLayers, 24,
    );
    profile.embedding.native.privateANE.recurrenceBlockSize = wholeNumber(
      profile.embedding.native.privateANE.recurrenceBlockSize, 8,
    );
    profile.embedding.native.privateANE.recurrenceLayerSlots = Array.isArray(
      profile.embedding.native.privateANE.recurrenceLayerSlots,
    ) ? [...new Set(profile.embedding.native.privateANE.recurrenceLayerSlots
      .map(Number).filter((item) => Number.isInteger(item) && item >= 0 && item < 18))] : [];
    if (!profile.embedding.native.privateANE.recurrenceLayerSlots.length) {
      profile.embedding.native.privateANE.recurrenceLayerSlots = [0];
    }
    profile.embedding.native.privateANE.recurrenceQueryScale = positiveNumber(
      profile.embedding.native.privateANE.recurrenceQueryScale, 4096,
    );
    profile.embedding.native.privateANE.recurrenceMaxTokens = wholeNumber(
      profile.embedding.native.privateANE.recurrenceMaxTokens,
      profile.embedding.native.privateANE.sequenceLength,
    );
    profile.embedding.native.privateANE.recurrenceIODtype =
      profile.embedding.native.privateANE.recurrenceIODtype === "fp32" ? "fp32" : "fp16";
    profile.embedding.native.privateANE.recurrenceVerifyReference =
      profile.embedding.native.privateANE.recurrenceVerifyReference === true;
    profile.embedding.native.maxQueuedRequests = Math.max(0, Math.floor(Number(profile.embedding.native.maxQueuedRequests ?? 16)) || 0);
    profile.embedding.native.startupTimeoutSeconds = wholeNumber(profile.embedding.native.startupTimeoutSeconds, 300);
    profile.embedding.native.maintenanceTimeoutSeconds = wholeNumber(profile.embedding.native.maintenanceTimeoutSeconds, 600);
    profile.embedding.native.autoRestart = profile.embedding.native.autoRestart !== false;
    profile.embedding.native.maxRestarts = Math.max(0, Math.floor(Number(profile.embedding.native.maxRestarts ?? 3)) || 0);
    profile.reranker.candidates = Number(profile.reranker.candidates || 20);
    profile.video = normalizeVideoProfile(source.video);
    if (profile.storage.provider === "oss-vectors") profile.storage.provider = "aliyun";
    if (!["local", "aliyun"].includes(profile.storage.provider)) profile.storage.provider = "local";
    output.profiles[id] = profile;
  }
  return output;
}

export function writeConfig(value, target = configPath()) {
  const normalized = normalizeConfig(value);
  for (const profile of Object.values(normalized.profiles)) {
    if (profile.embedding.provider === "apple-native") requireAppleExecutionMode(profile.embedding.native);
  }
  fs.mkdirSync(path.dirname(target), { recursive: true, mode: 0o700 });
  const temporary = `${target}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(normalized, null, 2)}\n`, { mode: 0o600 });
  fs.renameSync(temporary, target);
  fs.chmodSync(target, 0o600);
  return normalized;
}

export function loadConfig({ create = true } = {}) {
  const target = configPath();
  if (!fs.existsSync(target)) return create ? writeConfig(DEFAULT_CONFIG, target) : normalizeConfig();
  return normalizeConfig(JSON.parse(fs.readFileSync(target, "utf8")));
}

export function activeProfile(config = loadConfig()) {
  const profile = config.profiles[config.activeProfile];
  if (!profile) throw new Error(`配置档案不存在：${config.activeProfile}`);
  return { id: config.activeProfile, ...profile };
}

export function spaceId(profile) {
  const explicit = String(profile.spaceId || "").trim();
  if (explicit) return explicit;
  const model = String(profile.embedding?.model || "unknown").trim().toLowerCase();
  const dimension = Number(profile.embedding?.dimension || 0);
  const style = String(profile.embedding?.inputStyle || "auto").trim().toLowerCase();
  return `${model}-${dimension}-${style}-indexed-v1`.replace(/[^a-z0-9._-]+/g, "-");
}

const runtimeEmbeddingOverrides = new Map();

/** Install process-local credentials and endpoint returned by a managed helper. */
export function setRuntimeEmbeddingOverride(profileId, value) {
  const id = String(profileId || "").trim();
  if (!id) throw new Error("运行时 embedding override 需要 profileId");
  runtimeEmbeddingOverrides.set(id, { ...value });
}

export function clearRuntimeEmbeddingOverride(profileId) {
  runtimeEmbeddingOverrides.delete(String(profileId || "").trim());
}

export function clearRuntimeEmbeddingOverrides() {
  runtimeEmbeddingOverrides.clear();
}

export function runtimeEmbeddingOverride(profileId) {
  const value = runtimeEmbeddingOverrides.get(String(profileId || "").trim());
  return value ? { ...value } : null;
}

export function directConfig(profile) {
  const local = profile.storage.provider === "local";
  const runtime = runtimeEmbeddingOverride(profile.id);
  const embeddingBaseUrl = String(runtime?.baseUrl || profile.embedding.baseUrl || "").replace(/\/+$/, "");
  const embeddingModel = String(runtime?.model || profile.embedding.model || "");
  const embeddingDimension = Number(runtime?.dimension || profile.embedding.dimension || 0);
  const embeddingInputStyle = String(runtime?.inputStyle || profile.embedding.inputStyle || "auto");
  return {
    embeddingProvider: String(profile.embedding.provider || "remote"),
    embeddingBaseUrl,
    embeddingApiKey: String(runtime?.apiKey || profile.embedding.apiKey || ""),
    embeddingModel,
    embeddingDimension,
    embeddingInputStyle,
    embeddingSpace: String(runtime?.embeddingSpace || "") || spaceId({
      ...profile,
      embedding: {
        ...profile.embedding,
        model: embeddingModel,
        dimension: embeddingDimension,
        inputStyle: embeddingInputStyle,
      },
    }),
    embeddingSpaceExplicit: Boolean(runtime?.embeddingSpace || String(profile.spaceId || "").trim()),
    storageProvider: local ? "local" : "aliyun",
    localStorePath: String(profile.storage.path || ""),
    ossRegion: String(profile.storage.region || ""),
    ossAccountId: String(profile.storage.accountId || ""),
    ossBucket: String(profile.storage.bucket || ""),
    ossVisualIndex: String(profile.storage.visualIndex || (local ? "video-visual" : "")),
    ossTranscriptIndex: String(profile.storage.transcriptIndex || (local ? "video-transcript" : "")),
    documentIndex: String(profile.storage.documentIndex || "local-documents"),
    assetIndex: String(profile.storage.assetIndex || "library-assets"),
    video: normalizeVideoProfile(profile.video),
    ossAccessKeyId: String(profile.storage.accessKeyId || ""),
    ossAccessKeySecret: String(profile.storage.accessKeySecret || ""),
    ossSecurityToken: String(profile.storage.securityToken || ""),
  };
}

export function publicConfig(config = loadConfig()) {
  const output = clone(config);
  for (const profile of Object.values(output.profiles)) {
    profile.embedding.hasApiKey = Boolean(profile.embedding.apiKey);
    profile.embedding.apiKey = "";
    const storage = profile.storage;
    storage.hasAccessKeyId = Boolean(storage.accessKeyId);
    storage.hasAccessKeySecret = Boolean(storage.accessKeySecret);
    storage.hasSecurityToken = Boolean(storage.securityToken);
    storage.accessKeyId = mask(storage.accessKeyId);
    storage.accessKeySecret = "";
    storage.securityToken = "";
    profile.resolvedSpaceId = spaceId(profile);
    profile.resolvedStoragePath = profile.storage.provider === "local"
      ? resolveLocalStoragePath(profile.storage.path)
      : "";
  }
  output.configPath = configPath();
  output.indexedHome = indexedHomePath();
  output.performanceHistoryPath = ingestPerformanceHistoryPath();
  return output;
}

function resolveUserPath(selected) {
  const value = String(selected || "").trim();
  const expanded = value === "~"
    ? os.homedir()
    : value.startsWith("~/")
      ? path.join(os.homedir(), value.slice(2))
      : value;
  return path.resolve(expanded);
}

/** Resolve the effective zvec directory without changing the persisted value. */
export function resolveLocalStoragePath(configured = "") {
  const home = indexedHomePath();
  const selected = String(configured || "").trim()
    || String(process.env.INDEXED_DATA_DIR || "").trim()
    || (home ? path.join(home, "data", "zvec") : path.join(os.homedir(), ".local", "share", "indexed", "zvec"));
  return resolveUserPath(selected);
}

/** Runtime measurements live outside every vector database so a new DB can reuse them. */
export function ingestPerformanceHistoryPath() {
  const configured = String(process.env.INDEXED_STATE_DIR || "").trim();
  const home = indexedHomePath();
  const base = configured
    || (home ? path.join(home, "state") : "")
    || (String(process.env.XDG_STATE_HOME || "").trim()
      ? path.join(String(process.env.XDG_STATE_HOME).trim(), "indexed")
      : path.join(os.homedir(), ".local", "state", "indexed"));
  return path.join(resolveUserPath(base), "ingest-performance.json");
}

export function mask(value) {
  const text = String(value || "");
  if (!text) return "";
  if (text.length <= 4) return "*".repeat(text.length);
  return `${text.slice(0, 2)}${"*".repeat(Math.min(12, text.length - 4))}${text.slice(-2)}`;
}

export function mergeProfile(existing, incoming) {
  const next = merge(existing, incoming);
  next.embedding ||= {};
  if (!String(next.embedding.apiKey || "").trim() && existing?.embedding?.apiKey) {
    next.embedding.apiKey = existing.embedding.apiKey;
  }
  const oldStorage = existing?.storage || {};
  next.storage ||= {};
  for (const key of ["accessKeyId", "accessKeySecret", "securityToken"]) {
    if (!String(next.storage[key] || "").trim() && oldStorage[key]) next.storage[key] = oldStorage[key];
  }
  if (String(next.embedding.provider || "").trim().toLowerCase() === "apple-native") {
    const selected = Number(next.embedding.dimension);
    next.embedding.dimension = Number.isFinite(selected) && selected > 0
      ? Math.floor(selected)
      : APPLE_EMBEDDING_DEFAULT_DIMENSION;
    next.embedding.baseUrl = "";
    next.embedding.model = appleEmbeddingModel(next.embedding.dimension);
    next.embedding.inputStyle = "wemm";
    next.spaceId = "";
  }
  return next;
}

export function setPath(config, dottedPath, rawValue) {
  const keys = String(dottedPath || "").split(".").filter(Boolean);
  if (!keys.length) throw new Error("配置路径不能为空");
  let target = config;
  for (const key of keys.slice(0, -1)) {
    if (!target[key] || typeof target[key] !== "object") target[key] = {};
    target = target[key];
  }
  let value: unknown = rawValue;
  // Only command line strings are inferred; typed callers keep their own value.
  if (typeof rawValue === "string") {
    if (/^(true|false)$/i.test(rawValue)) value = rawValue.toLowerCase() === "true";
    else if (/^-?\d+(\.\d+)?$/.test(rawValue)) value = Number(rawValue);
  }
  target[keys.at(-1)] = value;
  return config;
}
