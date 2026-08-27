// @ts-nocheck -- behavior-preserving MV3 port; queue modules will be typed independently.
import {
  DIRECT_DEFAULT_CONFIG,
  deleteVectors,
  discoverEmbeddingConfig,
  embeddingSpace,
  embedImage,
  embedTranscript,
  embedVideo,
  embedVisualQuery,
  getVectors,
  listVectors,
  metadataFilter,
  putVectors,
  queryVectors,
  scoreFromDistance,
  testDirectConfig,
  uuid5Url,
  vectorEndpoint,
} from "@indexed/clients/direct-cloud";
import { OSS_VECTOR_PRICING, PRODUCT_DEFAULTS, SEGMENT_SCHEMAS } from "@indexed/contracts";
import { fixWebmDuration } from "./webm-duration.js";

const DEFAULT_CONFIG = { ...DIRECT_DEFAULT_CONFIG };
const SYNC_CONFIG_KEYS = [
  "backendMode",
  "controlPlaneUrl",
  "embeddingBaseUrl",
  "embeddingModel",
  "embeddingDimension",
  "embeddingInputStyle",
  "embeddingSpace",
  "ossRegion",
  "ossAccountId",
  "ossBucket",
  "ossVisualIndex",
  "ossTranscriptIndex",
];
const LOCAL_CREDENTIAL_KEYS = [
  "ossAccessKeyId",
  "ossAccessKeySecret",
  "ossSecurityToken",
];
const DB_NAME = "scene-memory-cache";
const DB_VERSION = 3;
const QUEUE_STORE = "queue";
const PROCESSED_STORE = "processed";
const PREVIEW_STORE = "previews";
const SESSION_STORE = "video_sessions";
const DEFAULT_SEGMENT_SECONDS = PRODUCT_DEFAULTS.capture.segmentSeconds;
const QUEUE_CONCURRENCY = PRODUCT_DEFAULTS.queue.concurrency;
const MAX_QUEUE_ATTEMPTS = PRODUCT_DEFAULTS.queue.maxAttempts;
const VECTOR_STORAGE_CNY_PER_GIB_MONTH = OSS_VECTOR_PRICING.storageCnyPerGibMonth;
const VECTOR_QUERY_CNY_PER_TIB = OSS_VECTOR_PRICING.queryCnyPerTib;
const OSS_GET_CNY_PER_10000 = OSS_VECTOR_PRICING.getCnyPer10k;
const FREE_VECTOR_WRITE_GIB_PER_MONTH = OSS_VECTOR_PRICING.freeWriteGibPerMonth;
const REBUILD_JOB_KEY = "youtubeRebuildJob";
const PROCESSING_PAUSED_KEY = "processingPaused";
const MAX_REPAIR_ATTEMPTS = PRODUCT_DEFAULTS.queue.maxRepairAttempts;
const lastFrames = new Map();
const activeQueueItems = new Map();
const deletingVideos = new Set();
let libraryRowsCache = null;
let draining = null;

function pickConfig(source, keys) {
  return Object.fromEntries(keys
    .filter((key) => Object.prototype.hasOwnProperty.call(source || {}, key))
    .map((key) => [key, source[key]]));
}

function maskCredential(value) {
  const text = String(value || "");
  if (!text) return "";
  if (text.length <= 4) return "*".repeat(text.length);
  return `${text.slice(0, 2)}${"*".repeat(text.length - 4)}${text.slice(-2)}`;
}

async function saveSyncedConfig(config) {
  const values = pickConfig(config, SYNC_CONFIG_KEYS);
  await chrome.storage.sync.set(values);
  return values;
}

async function restrictSyncedConfigAccess() {
  try {
    await chrome.storage.sync.setAccessLevel({ accessLevel: "TRUSTED_CONTEXTS" });
  } catch {}
}

async function readConfig() {
  const localKeys = [...SYNC_CONFIG_KEYS, ...LOCAL_CREDENTIAL_KEYS];
  const [local, synced] = await Promise.all([
    chrome.storage.local.get(localKeys),
    chrome.storage.sync.get(SYNC_CONFIG_KEYS).catch(() => ({})),
  ]);
  const migration = {};
  for (const key of SYNC_CONFIG_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(synced, key)
      && Object.prototype.hasOwnProperty.call(local, key)) {
      migration[key] = local[key];
    }
  }
  if (Object.keys(migration).length) {
    try { await chrome.storage.sync.set(migration); } catch {}
  }
  return {
    ...DEFAULT_CONFIG,
    ...pickConfig(local, SYNC_CONFIG_KEYS),
    ...synced,
    ...migration,
    ...pickConfig(local, LOCAL_CREDENTIAL_KEYS),
  };
}

async function localServerRequest(config, pathname, options = {}) {
  const baseUrl = String(config.controlPlaneUrl || "").trim().replace(/\/+$/, "");
  if (!baseUrl) throw new Error("请填写本地 Indexed 服务地址");
  const response = await fetch(`${baseUrl}${pathname}`, {
    headers: { "content-type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `本地服务返回 ${response.status}`);
  return payload;
}

function localResult(item, modality) {
  return {
    id: item.id,
    modality,
    score: Number(item.score || 0),
    distance: Number(item.distance || 0),
    video_id: item.videoId,
    source_site: item.sourceSite,
    title: item.title,
    channel_name: item.channel,
    source_url: item.sourceUrl,
    youtube_url: item.sourceUrl,
    start_time: Number(item.startTime || 0),
    end_time: Number(item.endTime || 0),
    transcript_text: item.transcript || "",
    thumbnail_url: item.preview || "",
    embedding_model: item.embeddingModel || "",
    embedding_space: item.embeddingSpace || "",
  };
}

async function processingPaused() {
  const stored = await chrome.storage.local.get({ [PROCESSING_PAUSED_KEY]: false });
  return Boolean(stored[PROCESSING_PAUSED_KEY]);
}

function openDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(QUEUE_STORE)) db.createObjectStore(QUEUE_STORE, { keyPath: "id" });
      if (!db.objectStoreNames.contains(PROCESSED_STORE)) db.createObjectStore(PROCESSED_STORE, { keyPath: "id" });
      if (!db.objectStoreNames.contains(PREVIEW_STORE)) db.createObjectStore(PREVIEW_STORE, { keyPath: "id" });
      if (!db.objectStoreNames.contains(SESSION_STORE)) db.createObjectStore(SESSION_STORE, { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function storeGet(storeName, key) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, "readonly");
    const request = tx.objectStore(storeName).get(key);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
    tx.oncomplete = () => db.close();
  });
}

async function storePut(storeName, value) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, "readwrite");
    tx.objectStore(storeName).put(value);
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => reject(tx.error);
  });
}

async function storeDelete(storeName, key) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, "readwrite");
    tx.objectStore(storeName).delete(key);
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => reject(tx.error);
  });
}

async function storeDeleteMatching(storeName, predicate) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    let deleted = 0;
    const tx = db.transaction(storeName, "readwrite");
    const request = tx.objectStore(storeName).openCursor();
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) return;
      if (predicate(cursor.value)) {
        cursor.delete();
        deleted += 1;
      }
      cursor.continue();
    };
    request.onerror = () => reject(request.error);
    tx.oncomplete = () => {
      db.close();
      resolve(deleted);
    };
    tx.onerror = () => reject(tx.error);
  });
}

async function purgeAbandonedQueueItems() {
  return storeDeleteMatching(QUEUE_STORE, (item) => Boolean(item?.terminal));
}

async function storeAll(storeName) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, "readonly");
    const request = tx.objectStore(storeName).getAll();
    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => reject(request.error);
    tx.oncomplete = () => db.close();
  });
}

async function sha256(value) {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value;
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function dataUrlBase64(dataUrl) {
  return dataUrl.slice(dataUrl.indexOf(",") + 1);
}

function hamming(left, right) {
  if (!left || !right) return 64;
  let value = BigInt(`0x${left}`) ^ BigInt(`0x${right}`);
  let count = 0;
  while (value) {
    count += Number(value & 1n);
    value >>= 1n;
  }
  return count;
}

async function cropFrame(dataUrl, rect, viewport) {
  const sourceBlob = await (await fetch(dataUrl)).blob();
  const bitmap = await createImageBitmap(sourceBlob);
  const scaleX = bitmap.width / Math.max(1, viewport.width);
  const scaleY = bitmap.height / Math.max(1, viewport.height);
  const sx = Math.max(0, Math.round(rect.left * scaleX));
  const sy = Math.max(0, Math.round(rect.top * scaleY));
  const sw = Math.max(1, Math.min(bitmap.width - sx, Math.round(rect.width * scaleX)));
  const sh = Math.max(1, Math.min(bitmap.height - sy, Math.round(rect.height * scaleY)));
  const targetWidth = Math.min(960, sw);
  const targetHeight = Math.max(1, Math.round((sh / sw) * targetWidth));
  const canvas = new OffscreenCanvas(targetWidth, targetHeight);
  const context = canvas.getContext("2d", { alpha: false });
  context.drawImage(bitmap, sx, sy, sw, sh, 0, 0, targetWidth, targetHeight);

  const hashCanvas = new OffscreenCanvas(9, 8);
  const hashContext = hashCanvas.getContext("2d", { willReadFrequently: true });
  hashContext.drawImage(canvas, 0, 0, 9, 8);
  const pixels = hashContext.getImageData(0, 0, 9, 8).data;
  let bits = 0n;
  let bitIndex = 0n;
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      const leftIndex = (y * 9 + x) * 4;
      const rightIndex = leftIndex + 4;
      const leftLuma = pixels[leftIndex] * 0.299 + pixels[leftIndex + 1] * 0.587 + pixels[leftIndex + 2] * 0.114;
      const rightLuma = pixels[rightIndex] * 0.299 + pixels[rightIndex + 1] * 0.587 + pixels[rightIndex + 2] * 0.114;
      if (leftLuma > rightLuma) bits |= 1n << bitIndex;
      bitIndex += 1n;
    }
  }
  bitmap.close();
  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.78 });
  const buffer = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let offset = 0; offset < buffer.length; offset += 32768) {
    binary += String.fromCharCode(...buffer.subarray(offset, offset + 32768));
  }
  return {
    base64: btoa(binary),
    hash: bits.toString(16).padStart(16, "0"),
    width: targetWidth,
    height: targetHeight,
  };
}

async function enqueue(item) {
  // The backend owns model-aware idempotency. A processed record in this browser
  // must not prevent re-indexing after switching model, service, or vector store.
  const existing = await storeGet(QUEUE_STORE, item.id);
  if (existing && !existing.terminal) return { cached: true };
  await storePut(QUEUE_STORE, { ...item, attempts: 0, createdAt: Date.now(), nextAttempt: 0 });
  if (!(await processingPaused())) void drainQueue();
  return { queued: true };
}

async function recordVideoSession(payload = {}) {
  const videoId = String(payload.videoId || payload.video_id || "").trim();
  const sourceSite = String(payload.sourceSite || payload.source_site || "youtube").trim().toLowerCase();
  if (!videoId) return { skipped: true, reason: "missing_video_id" };
  const updatedAt = Date.now();
  await storePut(SESSION_STORE, {
    id: `${sourceSite}:${videoId}`,
    videoId,
    sourceSite,
    title: String(payload.title || ""),
    channelId: String(payload.channelId || payload.channel_id || ""),
    channelName: String(payload.channelName || payload.channel_name || ""),
    sourceUrl: String(payload.sourceUrl || payload.source_url || ""),
    thumbnailUrl: String(payload.thumbnailUrl || payload.thumbnail_url || ""),
    duration: Math.max(0, Number(payload.duration || 0)),
    segmentInterval: Math.max(1, Number(payload.segmentInterval || payload.segment_interval || DEFAULT_SEGMENT_SECONDS)),
    optedIn: Boolean(payload.optedIn),
    running: Boolean(payload.running),
    playing: Boolean(payload.playing),
    recordingSegment: Number.isInteger(Number(payload.recordingSegment))
      ? Number(payload.recordingSegment)
      : -1,
    watchedSegments: Array.from(new Set((payload.watchedSegments || [])
      .map(Number)
      .filter((value) => Number.isInteger(value) && value >= 0))),
    updatedAt,
  });
  return { ok: true, updatedAt };
}

function mimeForFilename(filename, fallback) {
  const suffix = String(filename || "").toLowerCase().split(".").pop();
  if (suffix === "png") return "image/png";
  if (suffix === "webp") return "image/webp";
  if (suffix === "gif") return "image/gif";
  if (suffix === "mp4" || suffix === "m4v") return "video/mp4";
  if (suffix === "webm") return "video/webm";
  return fallback;
}

async function resizedPreviewDataUri(base64, filename, width, height, quality, maxBase64) {
  const value = String(base64 || "");
  if (!value) return "";
  const source = await (await fetch(`data:${mimeForFilename(filename, "image/jpeg")};base64,${value}`)).blob();
  const bitmap = await createImageBitmap(source);
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { alpha: false });
  context.fillStyle = "#000";
  context.fillRect(0, 0, width, height);
  const scale = Math.min(width / bitmap.width, height / bitmap.height);
  const drawWidth = Math.max(1, Math.round(bitmap.width * scale));
  const drawHeight = Math.max(1, Math.round(bitmap.height * scale));
  context.drawImage(
    bitmap,
    Math.round((width - drawWidth) / 2),
    Math.round((height - drawHeight) / 2),
    drawWidth,
    drawHeight
  );
  const blob = await canvas.convertToBlob({ type: "image/webp", quality });
  bitmap.close();
  const encoded = bytesToBase64(await blob.arrayBuffer());
  return encoded.length <= maxBase64 ? `data:image/webp;base64,${encoded}` : "";
}

async function tinyPreviewDataUri(base64, filename) {
  return (
    await resizedPreviewDataUri(base64, filename, 96, 54, 0.4, 720)
    || await resizedPreviewDataUri(base64, filename, 64, 36, 0.32, 720)
  );
}

async function localPreviewDataUri(base64, filename) {
  return (
    await resizedPreviewDataUri(base64, filename, 320, 180, 0.62, 120000)
    || await tinyPreviewDataUri(base64, filename)
  );
}

function usesClearCloudPreview(indexName) {
  return String(indexName || "").toLowerCase().endsWith("v2");
}

async function cloudAndLocalPreviews(base64, filename, indexName) {
  const localPromise = localPreviewDataUri(base64, filename);
  const cloudPromise = usesClearCloudPreview(indexName)
    ? localPromise
    : tinyPreviewDataUri(base64, filename);
  const [cloud, local] = await Promise.all([cloudPromise, localPromise]);
  return [cloud, local];
}

function segmentPreviewId(site, videoId, segmentIndex) {
  return `segment:${String(site || "youtube")}:${String(videoId || "")}:${Number(segmentIndex ?? -1)}`;
}

async function saveLocalPreview(recordId, body, dataUri) {
  if (!dataUri) return;
  const value = {
    id: String(recordId),
    dataUri,
    videoId: String(body.video_id || ""),
    sourceSite: String(body.source_site || "youtube"),
    segmentIndex: Number(body.segment_index ?? -1),
    updatedAt: Date.now(),
  };
  await Promise.all([
    storePut(PREVIEW_STORE, value),
    storePut(PREVIEW_STORE, {
      ...value,
      id: segmentPreviewId(body.source_site, body.video_id, body.segment_index),
    }),
  ]);
}

function withPreview(metadata, previewDataUri, indexName) {
  if (!previewDataUri) return metadata;
  const clear = usesClearCloudPreview(indexName);
  const candidate = {
    ...metadata,
    preview_data_uri: previewDataUri,
    ...(clear ? {
      preview_width: 320,
      preview_height: 180,
      preview_version: "clear-v2",
    } : {}),
  };
  const maxMetadataBytes = clear ? 200 * 1024 : 1900;
  return new TextEncoder().encode(JSON.stringify(candidate)).length <= maxMetadataBytes
    ? candidate
    : metadata;
}

async function existingVector(indexName, key, config) {
  const result = await getVectors(indexName, [key], config, {
    returnData: true,
    returnMetadata: true,
  });
  return result.vectors?.[0] || null;
}

async function enrichPreview(indexName, existing, previewDataUri, config) {
  if (!existing || !previewDataUri || existing.metadata?.preview_data_uri || !existing.data) return;
  const metadata = withPreview(existing.metadata || {}, previewDataUri, indexName);
  if (!metadata.preview_data_uri) return;
  await putVectors(
    indexName,
    [{
      key: String(existing.key),
      data: existing.data,
      metadata,
    }],
    config
  );
}

function stableSegmentIdentity(modality, body, space) {
  const site = String(body.source_site || "youtube");
  return `${site}:${modality}:${body.video_id}:${body.segment_schema}:${body.segment_index}:${space}`;
}

function sourceNamespace(body) {
  return String(body.source_site || "youtube");
}

function identitySpace(config) {
  // Keep deterministic IDs compatible with existing Indexed records until a
  // user explicitly chooses a semantic-space ID in a profile-aware client.
  return String(config.embeddingSpace || config.embeddingModel || embeddingSpace(config));
}

async function ingestVisualImage(body, config) {
  const digest = await sha256(Uint8Array.from(atob(body.image_base64), (value) => value.charCodeAt(0)));
  const identity = body.segment_index !== null && body.segment_index !== undefined
    ? stableSegmentIdentity("visual", body, identitySpace(config))
    : `${sourceNamespace(body)}:visual:${body.video_id}:legacy:${Math.round(Number(body.timestamp || 0) * 2)}:${digest}:${identitySpace(config)}`;
  const recordId = await uuid5Url(identity);
  const existing = await existingVector(config.ossVisualIndex, recordId, config);
  const [previewDataUri, localPreview] = await cloudAndLocalPreviews(
    body.image_base64,
    body.image_filename,
    config.ossVisualIndex
  );
  if (existing) {
    await enrichPreview(config.ossVisualIndex, existing, previewDataUri, config);
    await saveLocalPreview(recordId, body, localPreview);
    return { ok: true, cached: true, record_id: recordId };
  }
  const vector = await embedImage(
    body.image_base64,
    mimeForFilename(body.image_filename, "image/jpeg"),
    "",
    config
  );
  const metadata = withPreview({
    record_type: `${String(body.source_site || "youtube")}_visual`,
    source_site: String(body.source_site || "youtube"),
    video_id: body.video_id,
    title: String(body.title || ""),
    channel_id: String(body.channel_id || ""),
    channel_name: String(body.channel_name || ""),
    source_url: String(body.source_url || ""),
    thumbnail_url: String(body.thumbnail_url || ""),
    timestamp: Number(Number(body.timestamp || 0).toFixed(3)),
    duration: Number(Number(body.duration || 0).toFixed(3)),
    segment_index: body.segment_index ?? -1,
    segment_interval: Number(body.segment_interval || DEFAULT_SEGMENT_SECONDS),
    segment_schema: String(body.segment_schema || SEGMENT_SCHEMAS.visual),
    content_hash: digest,
    embedding_model: config.embeddingModel,
    embedding_space: embeddingSpace(config),
    indexed_at_ms: Date.now(),
  }, previewDataUri, config.ossVisualIndex);
  await putVectors(
    config.ossVisualIndex,
    [{ key: recordId, data: { float32: vector }, metadata }],
    config
  );
  await saveLocalPreview(recordId, body, localPreview);
  return { ok: true, cached: false, record_id: recordId };
}

async function ingestVideo(body, config) {
  const recordId = await uuid5Url(stableSegmentIdentity("video", body, identitySpace(config)));
  const existing = await existingVector(config.ossVisualIndex, recordId, config);
  const [previewDataUri, localPreview] = await cloudAndLocalPreviews(
    body.preview_base64,
    body.preview_filename,
    config.ossVisualIndex
  );
  if (existing) {
    await enrichPreview(config.ossVisualIndex, existing, previewDataUri, config);
    await saveLocalPreview(recordId, body, localPreview);
    return { ok: true, cached: true, record_id: recordId, embedding_basis: "native_video_10s" };
  }
  const mimeType = mimeForFilename(body.video_filename, "video/webm");
  let videoBase64 = body.video_base64;
  if (mimeType === "video/webm" && Number(body.captured_seconds || 0) > 0) {
    const source = Uint8Array.from(atob(videoBase64), (character) => character.charCodeAt(0));
    const repaired = await fixWebmDuration(
      new Blob([source], { type: mimeType }),
      Number(body.captured_seconds) * 1000
    );
    videoBase64 = bytesToBase64(await repaired.arrayBuffer());
  }
  const vector = await embedVideo(videoBase64, mimeType, config);
  const metadata = withPreview({
    record_type: `${String(body.source_site || "youtube")}_video`,
    source_site: String(body.source_site || "youtube"),
    video_id: body.video_id,
    title: String(body.title || ""),
    channel_id: String(body.channel_id || ""),
    channel_name: String(body.channel_name || ""),
    source_url: String(body.source_url || ""),
    thumbnail_url: String(body.thumbnail_url || ""),
    timestamp: Number(Number(body.start_time || 0).toFixed(3)),
    start_time: Number(Number(body.start_time || 0).toFixed(3)),
    end_time: Number(Number(body.end_time || 0).toFixed(3)),
    duration: Number(Number(body.duration || 0).toFixed(3)),
    captured_seconds: Number(Number(body.captured_seconds || 0).toFixed(3)),
    segment_index: Number(body.segment_index || 0),
    segment_interval: Number(body.segment_interval || DEFAULT_SEGMENT_SECONDS),
    segment_schema: String(body.segment_schema || SEGMENT_SCHEMAS.video),
    frame_count: Number(body.frame_count || 0),
    capture_fps: Number(body.capture_fps || 0),
    capture_width: Number(body.width || 0),
    capture_height: Number(body.height || 0),
    embedding_model: config.embeddingModel,
    embedding_space: embeddingSpace(config),
    embedding_basis: "native_video_10s",
    mime_type: mimeType,
    indexed_at_ms: Date.now(),
  }, previewDataUri, config.ossVisualIndex);
  await putVectors(
    config.ossVisualIndex,
    [{ key: recordId, data: { float32: vector }, metadata }],
    config
  );
  await saveLocalPreview(recordId, body, localPreview);
  return { ok: true, cached: false, record_id: recordId, embedding_basis: "native_video_10s" };
}

async function ingestTranscript(body, config) {
  const text = String(body.text || "").replace(/\s+/g, " ").trim();
  if (!text) return { skipped: true };
  const digest = await sha256(text);
  const identity = body.segment_index !== null && body.segment_index !== undefined
    ? stableSegmentIdentity("transcript", body, identitySpace(config))
    : `${sourceNamespace(body)}:transcript:${body.video_id}:legacy:${Number(body.start_time || 0).toFixed(1)}:${Number(body.end_time || 0).toFixed(1)}:${digest}:${identitySpace(config)}`;
  const recordId = await uuid5Url(identity);
  if (await existingVector(config.ossTranscriptIndex, recordId, config)) {
    return { ok: true, cached: true, record_id: recordId };
  }
  const vector = await embedTranscript(text, config);
  const metadata = {
    record_type: `${String(body.source_site || "youtube")}_transcript`,
    source_site: String(body.source_site || "youtube"),
    video_id: body.video_id,
    title: String(body.title || ""),
    channel_id: String(body.channel_id || ""),
    channel_name: String(body.channel_name || ""),
    source_url: String(body.source_url || ""),
    thumbnail_url: String(body.thumbnail_url || ""),
    start_time: Number(Number(body.start_time || 0).toFixed(3)),
    end_time: Number(Number(body.end_time || 0).toFixed(3)),
    segment_index: body.segment_index ?? -1,
    segment_interval: Number(body.segment_interval || DEFAULT_SEGMENT_SECONDS),
    segment_schema: String(body.segment_schema || SEGMENT_SCHEMAS.transcript),
    transcript_text: text,
    content_hash: digest,
    embedding_model: config.embeddingModel,
    embedding_space: embeddingSpace(config),
    indexed_at_ms: Date.now(),
  };
  await putVectors(
    config.ossTranscriptIndex,
    [{ key: recordId, data: { float32: vector }, metadata }],
    config
  );
  return { ok: true, cached: false, record_id: recordId };
}

async function directIngest(path, body) {
  const config = await readConfig();
  const kind = String(path || "").split("/").pop().replace(/-upsert$/, "");
  if (config.backendMode === "local") {
    return localServerRequest(config, `/api/ingest/${encodeURIComponent(kind)}`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  }
  if (kind === "video") return ingestVideo(body, config);
  if (kind === "visual") return ingestVisualImage(body, config);
  if (kind === "transcript") return ingestTranscript(body, config);
  throw new Error(`无后端模式不支持 ${path}`);
}

async function saveRebuildJob(job) {
  job.updatedAt = Date.now();
  await chrome.storage.local.set({ [REBUILD_JOB_KEY]: job });
  await chrome.action.setBadgeBackgroundColor({ color: job.active ? "#6d5dfc" : "#2f9e73" });
  await chrome.action.setBadgeText({
    text: job.active ? `${Math.min(99, Number(job.index || 0) + 1)}` : "",
  });
  return job;
}

async function segmentStatus(entry) {
  const config = await readConfig();
  const rows = await listVectors(config.ossVisualIndex, config, {
    filter: metadataFilter({ video_id: entry.id }),
  });
  const existing = new Set(
    rows
      .map((row) => Number(row.metadata?.segment_index))
      .filter((value) => Number.isInteger(value) && value >= 0)
  );
  const expected = Number(entry.expected_segments || Math.ceil(Number(entry.duration || 0) / DEFAULT_SEGMENT_SECONDS));
  const missing = [];
  for (let index = 0; index < expected; index += 1) {
    if (!existing.has(index)) missing.push(index);
  }
  return { expected, existing: existing.size, missing };
}

function rebuildVideoUrl(entry, startSegment = 0) {
  const base = entry.url || `https://www.youtube.com/watch?v=${entry.id}`;
  if (!startSegment) return base;
  return `${base}${base.includes("?") ? "&" : "?"}t=${startSegment * 10}s`;
}

async function advanceRebuildJob(job, tabId) {
  while (job.active && job.index < job.entries.length) {
    const entry = job.entries[job.index];
    const status = await segmentStatus(entry);
    if (!status.missing.length) {
      job.completedVideoIds = Array.from(new Set([...(job.completedVideoIds || []), entry.id]));
      job.index += 1;
      continue;
    }
    job.status = status.existing ? "repairing" : "playing";
    job.currentVideoId = entry.id;
    job.currentTitle = entry.title;
    job.currentExpectedSegments = status.expected;
    job.currentExistingSegments = status.existing;
    job.currentMissingSegments = status.missing.length;
    await saveRebuildJob(job);
    await chrome.tabs.update(tabId, { url: rebuildVideoUrl(entry, status.missing[0]) });
    return job;
  }
  job.active = false;
  job.status = "complete";
  job.completedAt = Date.now();
  job.currentVideoId = "";
  job.currentTitle = "";
  job.currentMissingSegments = 0;
  await saveRebuildJob(job);
  return job;
}

async function startChannelRebuild(payload, sender) {
  void payload;
  void sender;
  throw new Error("无后端插件不抓取频道清单；打开视频后会按实际观看的十秒片段直接写入阿里云");
}

async function waitForVideoQueue(videoId, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await drainQueue();
    const remaining = (await storeAll(QUEUE_STORE)).filter((item) => item.videoId === videoId);
    if (!remaining.length) return true;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  return false;
}

async function completeRebuildVideo(payload, sender) {
  const stored = await chrome.storage.local.get({ [REBUILD_JOB_KEY]: null });
  const job = stored[REBUILD_JOB_KEY];
  if (!job?.active || !sender.tab?.id || sender.tab.id !== job.tabId) return { skipped: true };
  const entry = job.entries?.[job.index];
  if (!entry || entry.id !== payload.videoId) return { skipped: true };
  job.status = "verifying";
  await saveRebuildJob(job);
  const drained = await waitForVideoQueue(entry.id);
  const status = await segmentStatus(entry);
  if (status.missing.length) {
    const attempts = Number(job.repairs?.[entry.id] || 0) + 1;
    job.repairs = { ...(job.repairs || {}), [entry.id]: attempts };
    if (attempts <= MAX_REPAIR_ATTEMPTS) {
      job.status = "repairing";
      job.currentExistingSegments = status.existing;
      job.currentMissingSegments = status.missing.length;
      await saveRebuildJob(job);
      await chrome.tabs.update(sender.tab.id, {
        url: rebuildVideoUrl(entry, status.missing[0]),
      });
      return { ok: true, repaired: true, drained, missing: status.missing };
    }
    job.errors = [
      ...(job.errors || []),
      { videoId: entry.id, title: entry.title, missing: status.missing, at: Date.now() },
    ];
  }
  job.completedVideoIds = Array.from(new Set([...(job.completedVideoIds || []), entry.id]));
  job.index += 1;
  job.status = "advancing";
  await saveRebuildJob(job);
  await advanceRebuildJob(job, sender.tab.id);
  return { ok: true, repaired: false, drained, missing: status.missing };
}

async function drainQueueInternal() {
  // Older versions kept media fragments that the model could never decode as
  // terminal errors. They are disposable inputs, not damaged cloud records.
  await purgeAbandonedQueueItems();
  while (true) {
    if (await processingPaused()) break;
    const items = (await storeAll(QUEUE_STORE))
      .filter((item) => !item.terminal && item.nextAttempt <= Date.now())
      .sort((a, b) => a.createdAt - b.createdAt);
    if (!items.length) break;
    const batch = items.slice(0, QUEUE_CONCURRENCY);
    const completed = await Promise.all(
      batch.map(async (item) => {
        activeQueueItems.set(item.id, {
          id: item.id,
          video_id: String(item.videoId || item.body?.video_id || ""),
          title: String(item.body?.title || "未命名视频"),
          source_site: String(item.body?.source_site || "youtube"),
          source_url: String(item.body?.source_url || ""),
          segment_index: Number(item.body?.segment_index ?? -1),
          kind: String(item.kind || ""),
        });
        try {
          const response = await directIngest(item.path, item.body);
          libraryRowsCache = null;
          await storeDelete(QUEUE_STORE, item.id);
          await storePut(PROCESSED_STORE, {
            id: item.id,
            processedAt: Date.now(),
            recordId: response.record_id || "",
            videoId: item.videoId,
            sourceSite: String(item.body?.source_site || "youtube"),
            kind: item.kind,
          });
          return true;
        } catch (error) {
          const attempts = Number(item.attempts || 0) + 1;
          const detail = String(error.message || error);
          const unrecoverableMedia = item.kind === "video"
            && attempts >= MAX_QUEUE_ATTEMPTS
            && /Embedding 服务返回 400|Failed to apply Qwen3VLProcessor|Could not open video|Number of samples/i.test(detail);
          if (unrecoverableMedia) {
            await storeDelete(QUEUE_STORE, item.id);
            return true;
          }
          await storePut(QUEUE_STORE, {
            ...item,
            attempts,
            terminal: false,
            lastError: detail,
            lastErrorAt: Date.now(),
            nextAttempt: Date.now() + Math.min(300000, 5000 * 2 ** Math.min(attempts, 6)),
          });
          return false;
        } finally {
          activeQueueItems.delete(item.id);
        }
      })
    );
    if (!completed.some(Boolean)) break;
  }
  const remaining = (await storeAll(QUEUE_STORE)).length;
  await chrome.action.setBadgeText({ text: remaining ? String(Math.min(99, remaining)) : "" });
  await chrome.action.setBadgeBackgroundColor({ color: "#7c6df2" });
}

function bytesToBase64(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 32768) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
  }
  return btoa(binary);
}

function parseStoryboardSpec(spec) {
  const parts = String(spec || "").split("|");
  if (parts.length < 2) throw new Error("这个视频没有可用的 Storyboard");
  const levelIndex = parts.length - 2;
  const fields = parts[levelIndex + 1].split("#");
  const [width, height, frameCount, columns, rows, intervalMs] = fields.slice(0, 6).map(Number);
  const filenameTemplate = fields[6] || "M$M";
  const signature = String(fields[7] || "").replace(/^rs\$/, "");
  if (![width, height, frameCount, columns, rows, intervalMs].every((value) => value > 0)) {
    throw new Error("无法解析 Storyboard 时间轴")
  }
  return {
    base: parts[0],
    levelIndex,
    width,
    height,
    frameCount,
    columns,
    rows,
    intervalMs,
    filenameTemplate,
    signature,
  };
}

function storyboardSheetUrl(storyboard, sheetIndex) {
  const filename = storyboard.filenameTemplate.replace("$M", String(sheetIndex));
  let url = storyboard.base
    .replace("$L", String(storyboard.levelIndex))
    .replace("$N", filename);
  if (storyboard.signature) url += `${url.includes("?") ? "&" : "?"}rs=${storyboard.signature}`;
  return url;
}

function captionChunks(body, maxSeconds = 25, maxCharacters = 180) {
  const cues = (body.events || [])
    .map((event) => ({
      time: Number(event.tStartMs || 0) / 1000,
      end: (Number(event.tStartMs || 0) + Number(event.dDurationMs || 0)) / 1000,
      text: (event.segs || []).map((segment) => segment.utf8 || "").join("").replace(/\s+/g, " ").trim(),
    }))
    .filter((cue) => cue.text);
  const chunks = [];
  let current = [];
  let characters = 0;
  const flush = () => {
    if (!current.length) return;
    chunks.push({
      startTime: current[0].time,
      endTime: Math.max(current[0].time, current[current.length - 1].end),
      text: current.map((cue) => cue.text).join(" ").replace(/\s+/g, " ").trim(),
    });
    current = [];
    characters = 0;
  };
  for (const cue of cues) {
    if (current.length && (cue.time - current[0].time >= maxSeconds || characters + cue.text.length > maxCharacters)) flush();
    current.push(cue);
    characters += cue.text.length;
  }
  flush();
  return chunks;
}

async function saveStoryboardJob(job) {
  await chrome.storage.local.set({ lastStoryboardJob: job });
  return job;
}

async function ingestStoryboard(payload) {
  const storyboard = parseStoryboardSpec(payload.storyboardSpec);
  const job = {
    job_id: crypto.randomUUID(),
    video_id: payload.videoId,
    status: "running",
    stage: "storyboard",
    message: "正在读取整条视频的预览时间轴",
    visual_total: storyboard.frameCount,
    visual_done: 0,
    transcript_total: 0,
    transcript_done: 0,
    started_at: Date.now(),
  };
  await saveStoryboardJob(job);
  try {
    const framesPerSheet = storyboard.columns * storyboard.rows;
    const sheetCount = Math.ceil(storyboard.frameCount / framesPerSheet);
    for (let sheetIndex = 0; sheetIndex < sheetCount; sheetIndex += 1) {
      const response = await fetch(storyboardSheetUrl(storyboard, sheetIndex));
      if (!response.ok) throw new Error(`Storyboard 返回 HTTP ${response.status}`);
      const bitmap = await createImageBitmap(await response.blob());
      for (let tile = 0; tile < framesPerSheet; tile += 1) {
        const frameIndex = sheetIndex * framesPerSheet + tile;
        if (frameIndex >= storyboard.frameCount) break;
        const canvas = new OffscreenCanvas(storyboard.width, storyboard.height);
        const context = canvas.getContext("2d", { alpha: false });
        context.drawImage(
          bitmap,
          (tile % storyboard.columns) * storyboard.width,
          Math.floor(tile / storyboard.columns) * storyboard.height,
          storyboard.width,
          storyboard.height,
          0,
          0,
          storyboard.width,
          storyboard.height
        );
        const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.84 });
        const timestamp = (frameIndex * storyboard.intervalMs) / 1000;
        await enqueue({
          id: `storyboard:${payload.videoId}:${frameIndex}`,
          videoId: payload.videoId,
          kind: "visual",
          path: "visual",
          body: {
            video_id: payload.videoId,
            title: payload.title || "",
            channel_id: payload.channelId || "",
            channel_name: payload.channelName || "",
            source_url: payload.sourceUrl,
            timestamp,
            duration: Number(payload.duration || 0),
            image_base64: bytesToBase64(await blob.arrayBuffer()),
            image_filename: `storyboard-${frameIndex}.jpg`,
            retain_preview: payload.retainPreview !== false,
          },
        });
        job.visual_done = frameIndex + 1;
      }
      job.message = `已读取 ${job.visual_done}/${job.visual_total} 个视觉时间点`;
      await saveStoryboardJob(job);
      bitmap.close();
    }

    if (payload.captionUrl) {
      job.stage = "captions";
      job.message = "正在读取完整字幕时间轴";
      await saveStoryboardJob(job);
      const separator = payload.captionUrl.includes("?") ? "&" : "?";
      const response = await fetch(`${payload.captionUrl}${separator}fmt=json3`);
      if (response.ok) {
        const chunks = captionChunks(await response.json());
        job.transcript_total = chunks.length;
        for (let index = 0; index < chunks.length; index += 1) {
          const chunk = chunks[index];
          const digest = await sha256(chunk.text);
          await enqueue({
            id: `storyboard-text:${payload.videoId}:${index}:${digest}`,
            videoId: payload.videoId,
            kind: "transcript",
            path: "transcript",
            body: {
              video_id: payload.videoId,
              title: payload.title || "",
              channel_id: payload.channelId || "",
              channel_name: payload.channelName || "",
              source_url: payload.sourceUrl,
              start_time: chunk.startTime,
              end_time: chunk.endTime,
              text: chunk.text,
            },
          });
          job.transcript_done = index + 1;
        }
      } else {
        job.warning = `字幕返回 HTTP ${response.status}`;
      }
    }

    job.stage = "embedding";
    job.message = "已预读整条视频，正在并发写入向量库";
    await saveStoryboardJob(job);
    await drainQueue();
    const queue = await storeAll(QUEUE_STORE);
    const remaining = queue.filter((item) => item.videoId === payload.videoId).length;
    job.status = remaining ? "queued" : "complete";
    job.stage = remaining ? "queued" : "complete";
    job.message = remaining ? `还有 ${remaining} 个时间点等待服务恢复` : "整条视频已经完成预索引";
    job.elapsed_seconds = Math.round((Date.now() - job.started_at) / 100) / 10;
    job.speed_multiple = job.elapsed_seconds ? Math.round((Number(payload.duration || 0) / job.elapsed_seconds) * 10) / 10 : 0;
    return saveStoryboardJob(job);
  } catch (error) {
    job.status = "failed";
    job.stage = "failed";
    job.error = String(error.message || error);
    job.message = "快速预索引失败";
    await saveStoryboardJob(job);
    return job;
  }
}

function drainQueue() {
  if (!draining) {
    draining = processingPaused().then((paused) => paused ? undefined : drainQueueInternal()).finally(() => {
      draining = null;
    });
  }
  return draining;
}

async function captureAndQueue(payload, sender) {
  if (!sender.tab?.active) return { skipped: true, reason: "inactive_tab" };
  const screenshot = await chrome.tabs.captureVisibleTab(sender.tab.windowId, {
    format: "jpeg",
    quality: 88,
  });
  const frame = await cropFrame(screenshot, payload.rect, payload.viewport);
  const key = `${payload.videoId}:${sender.tab.id}`;
  const previous = lastFrames.get(key);
  const elapsed = Number(payload.currentTime || 0) - Number(previous?.time || 0);
  if (previous && hamming(previous.hash, frame.hash) < 7 && elapsed >= 0 && elapsed < 20) {
    return { skipped: true, reason: "visual_cache" };
  }
  lastFrames.set(key, { hash: frame.hash, time: payload.currentTime });
  const id = `visual:${payload.sourceSite || "youtube"}:${payload.videoId}:${Math.round(payload.currentTime * 2)}:${frame.hash}`;
  return enqueue({
    id,
    videoId: payload.videoId,
    kind: "visual",
    path: "visual",
    body: {
      video_id: payload.videoId,
      source_site: payload.sourceSite || "youtube",
      title: payload.title,
      channel_id: payload.channelId,
      channel_name: payload.channelName,
      source_url: payload.sourceUrl,
      thumbnail_url: payload.thumbnailUrl || "",
      timestamp: payload.currentTime,
      duration: payload.duration,
      image_base64: frame.base64,
      image_filename: `youtube-${payload.videoId}-${Math.round(payload.currentTime)}.jpg`,
      retain_preview: payload.retainPreview !== false,
    },
  });
}

async function queueVideoFrame(payload, sender) {
  if (!payload.imageBase64 || !payload.imageHash) return { skipped: true, reason: "empty_frame" };
  const segmentIndex = Math.max(0, Number(payload.segmentIndex || 0));
  const segmentSchema = String(payload.segmentSchema || SEGMENT_SCHEMAS.visual);
  const id = `visual:${payload.sourceSite || "youtube"}:${payload.videoId}:${segmentSchema}:${segmentIndex}`;
  return enqueue({
    id,
    videoId: payload.videoId,
    kind: "visual",
    path: "visual",
    body: {
      video_id: payload.videoId,
      source_site: payload.sourceSite || "youtube",
      title: payload.title,
      channel_id: payload.channelId,
      channel_name: payload.channelName,
      source_url: payload.sourceUrl,
      thumbnail_url: payload.thumbnailUrl || "",
      timestamp: payload.currentTime,
      duration: payload.duration,
      segment_index: segmentIndex,
      segment_interval: Number(payload.segmentInterval || DEFAULT_SEGMENT_SECONDS),
      segment_schema: segmentSchema,
      image_base64: payload.imageBase64,
      image_filename: `youtube-${payload.videoId}-${Math.round(payload.currentTime)}.jpg`,
      retain_preview: payload.retainPreview !== false,
    },
  });
}

async function queueVideoSegment(payload) {
  if (!payload.videoBase64) return { error: "视频分片不能为空" };
  const segmentIndex = Math.max(0, Number(payload.segmentIndex || 0));
  const videoSegmentSchema = String(payload.videoSegmentSchema || SEGMENT_SCHEMAS.video);
  const transcriptSegmentSchema = String(payload.transcriptSegmentSchema || SEGMENT_SCHEMAS.transcript);
  const video = await enqueue({
    id: `video:${payload.sourceSite || "youtube"}:${payload.videoId}:${videoSegmentSchema}:${segmentIndex}`,
    videoId: payload.videoId,
    kind: "video",
    path: "video",
    body: {
      video_id: payload.videoId,
      source_site: payload.sourceSite || "youtube",
      title: payload.title,
      channel_id: payload.channelId,
      channel_name: payload.channelName,
      source_url: payload.sourceUrl,
      thumbnail_url: payload.thumbnailUrl || "",
      start_time: payload.startTime,
      end_time: payload.endTime,
      duration: payload.duration,
      segment_index: segmentIndex,
      segment_interval: Number(payload.segmentInterval || DEFAULT_SEGMENT_SECONDS),
      segment_schema: videoSegmentSchema,
      video_base64: payload.videoBase64,
      video_filename: payload.videoFilename || `${payload.sourceSite || "youtube"}-${payload.videoId}-${segmentIndex}.webm`,
      preview_base64: payload.previewBase64 || "",
      preview_filename: payload.previewFilename || `${payload.sourceSite || "youtube"}-${payload.videoId}-${segmentIndex}.jpg`,
      captured_seconds: Number(payload.capturedSeconds || 0),
      frame_count: Math.max(0, Number(payload.frameCount || 0)),
      capture_fps: Math.max(0, Number(payload.captureFps || 0)),
      width: Math.max(0, Number(payload.width || 0)),
      height: Math.max(0, Number(payload.height || 0)),
      retain_preview: payload.retainPreview !== false,
      retain_clip: false,
    },
  });

  let transcript = { skipped: true };
  const text = String(payload.captionText || "").replace(/\s+/g, " ").trim();
  if (text) {
    transcript = await enqueue({
      id: `text:${payload.sourceSite || "youtube"}:${payload.videoId}:${transcriptSegmentSchema}:${segmentIndex}`,
      videoId: payload.videoId,
      kind: "transcript",
      path: "transcript",
      body: {
        video_id: payload.videoId,
        source_site: payload.sourceSite || "youtube",
        title: payload.title,
        channel_id: payload.channelId,
        channel_name: payload.channelName,
        source_url: payload.sourceUrl,
        thumbnail_url: payload.thumbnailUrl || "",
        start_time: payload.startTime,
        end_time: payload.endTime,
        segment_index: segmentIndex,
        segment_interval: Number(payload.segmentInterval || DEFAULT_SEGMENT_SECONDS),
        segment_schema: transcriptSegmentSchema,
        text,
      },
    });
  }
  return { video, transcript };
}

async function enqueueTranscript(payload) {
  const normalized = String(payload.text || "").replace(/\s+/g, " ").trim();
  if (!normalized) return { skipped: true };
  const segmentIndex = Math.max(0, Number(payload.segmentIndex || 0));
  const segmentSchema = String(payload.segmentSchema || SEGMENT_SCHEMAS.transcriptLegacy);
  const id = `text:${payload.sourceSite || "youtube"}:${payload.videoId}:${segmentSchema}:${segmentIndex}`;
  return enqueue({
    id,
    videoId: payload.videoId,
    kind: "transcript",
    path: "transcript",
    body: {
      video_id: payload.videoId,
      source_site: payload.sourceSite || "youtube",
      title: payload.title,
      channel_id: payload.channelId,
      channel_name: payload.channelName,
      source_url: payload.sourceUrl,
      thumbnail_url: payload.thumbnailUrl || "",
      start_time: payload.startTime,
      end_time: payload.endTime,
      segment_index: segmentIndex,
      segment_interval: Number(payload.segmentInterval || DEFAULT_SEGMENT_SECONDS),
      segment_schema: segmentSchema,
      text: normalized,
    },
  });
}

function sourceAt(metadata, seconds) {
  const videoId = String(metadata.video_id || "");
  const site = String(metadata.source_site || "youtube");
  const fallback = site === "bilibili"
    ? `https://www.bilibili.com/video/${encodeURIComponent(videoId)}/`
    : `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}`;
  const position = Math.max(0, Math.floor(seconds));
  try {
    const url = new URL(String(metadata.source_url || fallback));
    url.searchParams.set("t", site === "bilibili" ? String(position) : `${position}s`);
    return url.toString();
  } catch {
    return `${fallback}${fallback.includes("?") ? "&" : "?"}t=${site === "bilibili" ? position : `${position}s`}`;
  }
}

function directResult(row, kind) {
  const metadata = row.metadata || {};
  const cloudPreview = String(metadata.preview_data_uri || "");
  const playbackTime = Number(
    kind === "transcript"
      ? (metadata.start_time ?? metadata.timestamp ?? 0)
      : (metadata.timestamp ?? metadata.start_time ?? 0)
  );
  const videoId = String(metadata.video_id || "");
  const sourceSite = String(metadata.source_site || "youtube");
  const openUrl = sourceAt(metadata, playbackTime);
  return {
    id: String(row.key || ""),
    score: scoreFromDistance(row.distance),
    vector_distance: Number(row.distance || 0),
    video_id: videoId,
    source_site: sourceSite,
    title: String(metadata.title || "未命名视频"),
    channel_id: String(metadata.channel_id || ""),
    channel_name: String(metadata.channel_name || "未知频道"),
    playback_time: playbackTime,
    timestamp: playbackTime,
    start_time: Number(metadata.start_time ?? playbackTime),
    end_time: Number(metadata.end_time ?? playbackTime + Number(metadata.segment_interval || DEFAULT_SEGMENT_SECONDS)),
    transcript_text: String(metadata.transcript_text || ""),
    thumbnail_url: String(
      cloudPreview
      || metadata.thumbnail_url
      || (sourceSite === "youtube" && videoId ? `https://i.ytimg.com/vi/${encodeURIComponent(videoId)}/hqdefault.jpg` : "")
    ),
    preview_source: cloudPreview.startsWith("data:image/") ? "oss_metadata" : "source",
    clip_url: "",
    open_url: openUrl,
    youtube_url: openUrl,
    embedding_model: String(metadata.embedding_model || ""),
    embedding_space: String(metadata.embedding_space || ""),
    embedding_basis: String(metadata.embedding_basis || ""),
    segment_index: Number(metadata.segment_index ?? -1),
  };
}

async function attachLocalPreviews(items) {
  return Promise.all(
    items.map(async (item) => {
      if (item.preview_source === "oss_metadata") return item;
      const direct = await storeGet(PREVIEW_STORE, item.id);
      const segment = direct || await storeGet(
        PREVIEW_STORE,
        segmentPreviewId(item.source_site, item.video_id, item.segment_index)
      );
      return segment?.dataUri
        ? { ...item, thumbnail_url: segment.dataUri, preview_source: "indexeddb" }
        : item;
    })
  );
}

async function directSearch(payload) {
  const query = String(payload.query || "").trim();
  if (!query) throw new Error("搜索文字不能为空");
  const config = await readConfig();
  if (config.backendMode === "local") {
    const response = await localServerRequest(config, "/api/search", {
      method: "POST",
      body: JSON.stringify({ query, limit: payload.limit, videoId: payload.video_id || payload.videoId }),
    });
    return {
      query,
      retrieval_mode: "local_server_dual_vector",
      reranked: Boolean(response.reranked),
      embedding_model: response.embeddingModel,
      embedding_space: response.embeddingSpace,
      storage_provider: "local_lancedb",
      storage_label: "本地 LanceDB",
      visual: (response.visual || []).map((item) => localResult(item, "visual")),
      transcript: (response.transcript || []).map((item) => localResult(item, "transcript")),
    };
  }
  const filter = metadataFilter({
    embedding_model: config.embeddingModel,
    embedding_space: embeddingSpace(config),
    video_id: payload.video_id || payload.videoId,
    channel_id: payload.channel_id || payload.channelId,
  });
  const [visualVector, transcriptVector] = await Promise.all([
    embedVisualQuery(query, config),
    embedTranscript(query, config),
  ]);
  const [visual, transcript] = await Promise.all([
    queryVectors(config.ossVisualIndex, visualVector, config, { limit: payload.limit, filter }),
    queryVectors(config.ossTranscriptIndex, transcriptVector, config, { limit: payload.limit, filter }),
  ]);
  const [visualResults, transcriptResults] = await Promise.all([
    attachLocalPreviews((visual.vectors || []).map((row) => directResult(row, "visual"))),
    attachLocalPreviews((transcript.vectors || []).map((row) => directResult(row, "transcript"))),
  ]);
  return {
    query,
    retrieval_mode: "direct_dual_vector_raw_score",
    reranked: false,
    embedding_model: config.embeddingModel,
    embedding_space: embeddingSpace(config),
    embedding_dimension: Number(config.embeddingDimension),
    storage_provider: "oss_vectors_direct",
    storage_label: "阿里云 OSS Vector Bucket（插件直连）",
    visual: visualResults,
    transcript: transcriptResults,
  };
}

async function directImageSearch(payload) {
  const config = await readConfig();
  let encoded = String(payload.image_base64 || "").trim();
  let mimeType = mimeForFilename(payload.image_filename, "image/png");
  if (encoded.startsWith("data:") && encoded.includes(",")) {
    mimeType = encoded.slice(5, encoded.indexOf(";")) || mimeType;
    encoded = encoded.slice(encoded.indexOf(",") + 1);
  }
  if (!encoded) throw new Error("查询图片不能为空");
  if (config.backendMode === "local") {
    const response = await localServerRequest(config, "/api/search/image", {
      method: "POST",
      body: JSON.stringify({
        imageBase64: encoded,
        mimeType,
        query: payload.query || "",
        limit: payload.limit,
        videoId: payload.video_id || payload.videoId,
      }),
    });
    return {
      query: String(payload.query || "").trim(),
      query_modality: "image",
      retrieval_mode: "local_server_image_vector",
      reranked: Boolean(response.reranked),
      embedding_model: response.embeddingModel,
      embedding_space: response.embeddingSpace,
      storage_provider: "local_lancedb",
      storage_label: "本地 LanceDB",
      visual: (response.visual || []).map((item) => localResult(item, "visual")),
      transcript: [],
    };
  }
  const vector = await embedImage(encoded, mimeType, payload.query || "", config);
  const filter = metadataFilter({
    embedding_model: config.embeddingModel,
    embedding_space: embeddingSpace(config),
    video_id: payload.video_id || payload.videoId,
    channel_id: payload.channel_id || payload.channelId,
  });
  const visual = await queryVectors(config.ossVisualIndex, vector, config, {
    limit: payload.limit,
    filter,
  });
  const visualResults = await attachLocalPreviews(
    (visual.vectors || []).map((row) => directResult(row, "visual"))
  );
  return {
    query: String(payload.query || "").trim(),
    query_modality: "image",
    retrieval_mode: "direct_image_to_video_vector_raw_score",
    reranked: false,
    embedding_model: config.embeddingModel,
    embedding_space: embeddingSpace(config),
    embedding_dimension: Number(config.embeddingDimension),
    storage_provider: "oss_vectors_direct",
    storage_label: "阿里云 OSS Vector Bucket（插件直连）",
    visual: visualResults,
    transcript: [],
  };
}

async function directStats(videoId = "") {
  const config = await readConfig();
  if (config.backendMode === "local") {
    const library = await localServerRequest(config, "/api/extension-library");
    const videos = videoId ? (library.videos || []).filter((item) => String(item.video_id) === String(videoId)) : (library.videos || []);
    return {
      video_id: videoId,
      visual_count: videos.reduce((sum, item) => sum + Number(item.visual_count || 0), 0),
      transcript_count: videos.reduce((sum, item) => sum + Number(item.transcript_count || 0), 0),
      video_count: videos.length,
      record_count: videos.reduce((sum, item) => sum + Number(item.visual_count || 0) + Number(item.transcript_count || 0), 0),
      storage_provider: "local_lancedb",
      storage_label: "本地 LanceDB",
    };
  }
  const filter = metadataFilter({
    embedding_model: config.embeddingModel,
    embedding_space: embeddingSpace(config),
    video_id: videoId,
  });
  const [visualRows, transcriptRows] = await Promise.all([
    listVectors(config.ossVisualIndex, config, { filter }),
    listVectors(config.ossTranscriptIndex, config, { filter }),
  ]);
  const matches = (row) => !videoId || String(row.metadata?.video_id || "") === String(videoId);
  const visual = visualRows.filter(matches);
  const transcript = transcriptRows.filter(matches);
  const segmentIndexes = (rows) => Array.from(new Set(
    rows
      .map((row) => Number(row.metadata?.segment_index))
      .filter((value) => Number.isInteger(value) && value >= 0)
  )).sort((a, b) => a - b);
  const videos = new Set(
    [...visual, ...transcript]
      .map((row) => String(row.metadata?.video_id || ""))
      .filter(Boolean)
  );
  return {
    video_id: videoId,
    visual_count: visual.length,
    transcript_count: transcript.length,
    visual_segments: segmentIndexes(visual),
    transcript_segments: segmentIndexes(transcript),
    video_count: videos.size,
    record_count: visual.length + transcript.length,
    storage_provider: "oss_vectors_direct",
    storage_label: "阿里云 OSS Vector Bucket（插件直连）",
    embedding_model: config.embeddingModel,
    embedding_space: embeddingSpace(config),
    embedding_dimension: Number(config.embeddingDimension),
  };
}

function sourceSiteFromMetadata(metadata = {}) {
  const explicit = String(metadata.source_site || "").toLowerCase();
  if (explicit) return explicit;
  const recordType = String(metadata.record_type || "").toLowerCase();
  return recordType.startsWith("bilibili") ? "bilibili" : "youtube";
}

function videoLibrarySummary(visualRows, transcriptRows, queue, sessions = []) {
  const videos = new Map();
  const ensure = (videoId, source = {}) => {
    const id = String(videoId || source.video_id || "");
    if (!id) return null;
    if (!videos.has(id)) {
      videos.set(id, {
        video_id: id,
        title: "未命名视频",
        channel_name: "",
        source_site: "youtube",
        source_url: "",
        thumbnail_url: "",
        duration: 0,
        segment_interval: DEFAULT_SEGMENT_SECONDS,
        visual_count: 0,
        transcript_count: 0,
        saved_visual_segments: new Set(),
        saved_transcript_segments: new Set(),
        pending_visual_segments: new Set(),
        pending_transcript_segments: new Set(),
        failed_visual_segments: new Set(),
        processing_segments: new Set(),
        last_indexed_at: 0,
        last_session_at: 0,
        active_capture: false,
        capture_playing: false,
        current_segment: -1,
      });
    }
    const video = videos.get(id);
    if (source.title) video.title = String(source.title);
    if (source.channel_name) video.channel_name = String(source.channel_name);
    if (source.source_url) video.source_url = String(source.source_url);
    if (source.thumbnail_url) video.thumbnail_url = String(source.thumbnail_url);
    if (source.source_site || source.record_type) video.source_site = sourceSiteFromMetadata(source);
    video.duration = Math.max(video.duration, Number(source.duration || source.end_time || 0));
    video.segment_interval = Math.max(1, Number(source.segment_interval || video.segment_interval || DEFAULT_SEGMENT_SECONDS));
    video.last_indexed_at = Math.max(video.last_indexed_at, Number(source.indexed_at_ms || 0));
    return video;
  };
  const addSegment = (set, value) => {
    const index = Number(value);
    if (Number.isInteger(index) && index >= 0) set.add(index);
  };

  for (const row of visualRows) {
    const metadata = row.metadata || {};
    const video = ensure(metadata.video_id, metadata);
    if (!video) continue;
    video.visual_count += 1;
    addSegment(video.saved_visual_segments, metadata.segment_index);
  }
  for (const row of transcriptRows) {
    const metadata = row.metadata || {};
    const video = ensure(metadata.video_id, metadata);
    if (!video) continue;
    video.transcript_count += 1;
    addSegment(video.saved_transcript_segments, metadata.segment_index);
  }
  for (const item of queue) {
    if (item.terminal) continue;
    const body = item.body || {};
    const video = ensure(item.videoId || body.video_id, body);
    if (!video) continue;
    if (item.kind === "transcript") {
      addSegment(video.pending_transcript_segments, body.segment_index);
    } else {
      addSegment(video.pending_visual_segments, body.segment_index);
    }
  }
  for (const item of activeQueueItems.values()) {
    const video = ensure(item.video_id, item);
    if (video && item.kind !== "transcript") addSegment(video.processing_segments, item.segment_index);
  }
  const sessionFreshAfter = Date.now() - 15000;
  for (const session of sessions) {
    const video = ensure(session.videoId, {
      video_id: session.videoId,
      source_site: session.sourceSite,
      title: session.title,
      channel_id: session.channelId,
      channel_name: session.channelName,
      source_url: session.sourceUrl,
      thumbnail_url: session.thumbnailUrl,
      duration: session.duration,
      segment_interval: session.segmentInterval,
    });
    if (!video) continue;
    video.last_session_at = Math.max(video.last_session_at, Number(session.updatedAt || 0));
    const active = Boolean(
      session.optedIn
      && session.running
      && Number(session.updatedAt || 0) >= sessionFreshAfter
    );
    video.active_capture = video.active_capture || active;
    video.capture_playing = video.capture_playing || (active && Boolean(session.playing));
    if (active && Number.isInteger(Number(session.recordingSegment)) && Number(session.recordingSegment) >= 0) {
      const segment = Number(session.recordingSegment);
      video.current_segment = segment;
      addSegment(video.processing_segments, segment);
    }
  }

  const list = Array.from(videos.values()).map((video) => {
    const arrays = {};
    for (const key of [
      "saved_visual_segments",
      "saved_transcript_segments",
      "pending_visual_segments",
      "pending_transcript_segments",
      "failed_visual_segments",
      "processing_segments",
    ]) arrays[key] = Array.from(video[key]).sort((a, b) => a - b);
    const lastKnown = Math.max(-1, ...Object.values(arrays).flat());
    const totalBlocks = Math.max(
      lastKnown + 1,
      video.duration > 0 ? Math.ceil(video.duration / video.segment_interval) : 0
    );
    return {
      ...video,
      ...arrays,
      total_blocks: totalBlocks,
      pending_count: arrays.pending_visual_segments.length + arrays.pending_transcript_segments.length,
      failed_count: arrays.failed_visual_segments.length,
      processing: arrays.processing_segments.length > 0 || video.active_capture,
    };
  });
  list.sort((left, right) =>
    Number(right.processing) - Number(left.processing)
    || right.pending_count - left.pending_count
    || right.visual_count - left.visual_count
    || left.title.localeCompare(right.title, "zh-CN")
  );
  return list;
}

function videoIdentity(sourceSite, videoId) {
  return `${String(sourceSite || "youtube").toLowerCase()}:${String(videoId || "")}`;
}

function encodedBytes(value) {
  return new TextEncoder().encode(String(value || "")).length;
}

function estimatedVectorRowBytes(row, dimension) {
  return Number(dimension || 0) * 4
    + encodedBytes(row?.key)
    + encodedBytes(JSON.stringify(row?.metadata || {}));
}

function billingEstimate(visualRows, transcriptRows, dimension) {
  const visualBytes = visualRows.reduce(
    (total, row) => total + estimatedVectorRowBytes(row, dimension),
    0
  );
  const transcriptBytes = transcriptRows.reduce(
    (total, row) => total + estimatedVectorRowBytes(row, dimension),
    0
  );
  const totalBytes = visualBytes + transcriptBytes;
  const totalGib = totalBytes / 1024 ** 3;
  const scanCnyPer1000 = (totalGib / 1024) * VECTOR_QUERY_CNY_PER_TIB * 1000;
  // A text search scans the visual and transcript indexes once each.
  const requestCnyPer1000 = (2000 / 10000) * OSS_GET_CNY_PER_10000;
  return {
    estimated_bytes: totalBytes,
    estimated_gib: totalGib,
    visual_gib: visualBytes / 1024 ** 3,
    transcript_gib: transcriptBytes / 1024 ** 3,
    storage_month_cny: totalGib * VECTOR_STORAGE_CNY_PER_GIB_MONTH,
    text_search_1000_cny: scanCnyPer1000 + requestCnyPer1000,
    text_search_scan_1000_cny: scanCnyPer1000,
    text_search_request_1000_cny: requestCnyPer1000,
    current_write_cny_after_free: Math.max(0, totalGib - FREE_VECTOR_WRITE_GIB_PER_MONTH) * 0.5,
    free_write_gib_per_month: FREE_VECTOR_WRITE_GIB_PER_MONTH,
    rates: {
      storage_cny_per_gib_month: VECTOR_STORAGE_CNY_PER_GIB_MONTH,
      query_cny_per_tib: VECTOR_QUERY_CNY_PER_TIB,
      get_cny_per_10000: OSS_GET_CNY_PER_10000,
    },
  };
}

function tabVideoIdentity(urlValue) {
  try {
    const url = new URL(urlValue);
    if (url.hostname.endsWith("bilibili.com")) {
      return videoIdentity("bilibili", url.pathname.match(/\/video\/(BV[A-Za-z0-9]+|av\d+)/i)?.[1] || "");
    }
    if (url.hostname.endsWith("youtube.com")) {
      return videoIdentity("youtube", url.searchParams.get("v") || "");
    }
  } catch {}
  return "";
}

function itemMatchesVideo(item, sourceSite, videoId) {
  const body = item?.body || item || {};
  const itemVideoId = String(item?.videoId || item?.video_id || body.video_id || "");
  const itemSourceSite = String(item?.sourceSite || item?.source_site || body.source_site || sourceSite || "youtube");
  return itemVideoId === videoId && (!sourceSite || itemSourceSite === sourceSite);
}

async function stopOpenVideoTabs(sourceSite, videoId) {
  const identity = videoIdentity(sourceSite, videoId);
  const tabs = await chrome.tabs.query({});
  const matches = tabs.filter((tab) => tab.id && tabVideoIdentity(tab.url) === identity);
  await Promise.all(matches.map(async (tab) => {
    try {
      await chrome.tabs.sendMessage(tab.id, { type: "STOP_INDEXING" });
    } catch {}
  }));
  return matches.length;
}

async function waitForActiveVideo(sourceSite, videoId, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const active = Array.from(activeQueueItems.values()).some((item) =>
      itemMatchesVideo(item, sourceSite, videoId)
    );
    if (!active) return true;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return false;
}

async function deleteVectorKeys(indexName, keys, config) {
  const uniqueKeys = Array.from(new Set(keys.map((key) => String(key || "")).filter(Boolean)));
  for (let offset = 0; offset < uniqueKeys.length; offset += 500) {
    await deleteVectors(indexName, uniqueKeys.slice(offset, offset + 500), config);
  }
  return uniqueKeys.length;
}

function resetTarget(config) {
  return [
    String(config.ossRegion || ""),
    String(config.ossAccountId || ""),
    String(config.ossBucket || ""),
    String(config.ossVisualIndex || ""),
    String(config.ossTranscriptIndex || ""),
  ].join("/");
}

async function stopAllOpenVideoTabs() {
  const tabs = await chrome.tabs.query({
    url: ["https://www.youtube.com/watch*", "https://www.bilibili.com/video/*"],
  });
  await Promise.all(tabs.map(async (tab) => {
    if (!tab.id) return;
    try { await chrome.tabs.sendMessage(tab.id, { type: "STOP_INDEXING" }); } catch {}
  }));
  return tabs.length;
}

async function waitForQueueIdle(timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!activeQueueItems.size) return true;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return false;
}

async function resetVideoMemoryInfo() {
  const config = await readConfig();
  if (config.backendMode === "local") throw new Error("清空云端索引仅适用于阿里云模式");
  const [visualRows, transcriptRows] = await Promise.all([
    listVectors(config.ossVisualIndex, config, { returnMetadata: false }),
    listVectors(config.ossTranscriptIndex, config, { returnMetadata: false }),
  ]);
  return {
    ok: true,
    target: resetTarget(config),
    bucket: config.ossBucket,
    visual_index: config.ossVisualIndex,
    transcript_index: config.ossTranscriptIndex,
    visual_count: visualRows.length,
    transcript_count: transcriptRows.length,
  };
}

async function resetAllVideoMemory(payload = {}) {
  const config = await readConfig();
  if (config.backendMode === "local") throw new Error("清空云端索引仅适用于阿里云模式");
  const target = resetTarget(config);
  if (!target || String(payload.target || "") !== target) throw new Error("索引目标已变化，请重新确认");
  await chrome.storage.local.set({ [PROCESSING_PAUSED_KEY]: true });
  const stoppedTabs = await stopAllOpenVideoTabs();
  try {
    if (!await waitForQueueIdle()) throw new Error("仍有分片正在写入，请稍后重试");
    const [visualRows, transcriptRows] = await Promise.all([
      listVectors(config.ossVisualIndex, config, { returnMetadata: false }),
      listVectors(config.ossTranscriptIndex, config, { returnMetadata: false }),
    ]);
    const visualDeleted = await deleteVectorKeys(config.ossVisualIndex, visualRows.map((row) => row.key), config);
    const transcriptDeleted = await deleteVectorKeys(config.ossTranscriptIndex, transcriptRows.map((row) => row.key), config);
    const [queueDeleted, processedDeleted, previewsDeleted, sessionsDeleted] = await Promise.all([
      storeDeleteMatching(QUEUE_STORE, () => true),
      storeDeleteMatching(PROCESSED_STORE, () => true),
      storeDeleteMatching(PREVIEW_STORE, () => true),
      storeDeleteMatching(SESSION_STORE, () => true),
    ]);
    await chrome.storage.local.set({
      [REBUILD_JOB_KEY]: null,
      videoIndexingOptIns: {},
      autoIndexWatched: false,
    });
    libraryRowsCache = null;
    return {
      ok: true,
      target,
      visual_deleted: visualDeleted,
      transcript_deleted: transcriptDeleted,
      local_deleted: queueDeleted + processedDeleted + previewsDeleted + sessionsDeleted,
      stopped_tabs: stoppedTabs,
    };
  } finally {
    await chrome.storage.local.set({ [PROCESSING_PAUSED_KEY]: false });
  }
}

async function deleteVideoMemory(payload = {}) {
  const videoId = String(payload.video_id || payload.videoId || "").trim();
  const sourceSite = String(payload.source_site || payload.sourceSite || "youtube").trim().toLowerCase();
  if (!videoId) throw new Error("缺少要删除的视频 ID");
  const identity = videoIdentity(sourceSite, videoId);
  if (deletingVideos.has(identity)) throw new Error("这个视频正在删除，请稍候");
  deletingVideos.add(identity);
  try {
    const config = await readConfig();
    if (config.backendMode === "local") {
      const stoppedTabs = await stopOpenVideoTabs(sourceSite, videoId);
      const result = await localServerRequest(config, `/api/videos/${encodeURIComponent(sourceSite)}/${encodeURIComponent(videoId)}`, { method: "DELETE" });
      const [queueDeleted, processedDeleted, previewsDeleted] = await Promise.all([
        storeDeleteMatching(QUEUE_STORE, (item) => itemMatchesVideo(item, sourceSite, videoId)),
        storeDeleteMatching(PROCESSED_STORE, (item) => itemMatchesVideo(item, sourceSite, videoId)),
        storeDeleteMatching(PREVIEW_STORE, (item) => itemMatchesVideo(item, sourceSite, videoId)),
      ]);
      libraryRowsCache = null;
      return { ...result, local_deleted: queueDeleted + processedDeleted + previewsDeleted, stopped_tabs: stoppedTabs };
    }
    const stoppedTabs = await stopOpenVideoTabs(sourceSite, videoId);
    const queueBefore = await storeDeleteMatching(QUEUE_STORE, (item) =>
      itemMatchesVideo(item, sourceSite, videoId)
    );
    if (!await waitForActiveVideo(sourceSite, videoId)) {
      throw new Error("仍有分片正在写入，请稍后重试删除");
    }
    const queueAfter = await storeDeleteMatching(QUEUE_STORE, (item) =>
      itemMatchesVideo(item, sourceSite, videoId)
    );
    const filter = metadataFilter({
      video_id: videoId,
      embedding_model: config.embeddingModel,
      embedding_space: embeddingSpace(config),
    });
    const [visualRows, transcriptRows] = await Promise.all([
      listVectors(config.ossVisualIndex, config, { filter }),
      listVectors(config.ossTranscriptIndex, config, { filter }),
    ]);
    const matches = (row) => {
      const metadata = row.metadata || {};
      return String(metadata.video_id || "") === videoId
        && String(metadata.source_site || "youtube").toLowerCase() === sourceSite;
    };
    const visualKeys = visualRows.filter(matches).map((row) => row.key);
    const transcriptKeys = transcriptRows.filter(matches).map((row) => row.key);
    const visualDeleted = await deleteVectorKeys(config.ossVisualIndex, visualKeys, config);
    const transcriptDeleted = await deleteVectorKeys(config.ossTranscriptIndex, transcriptKeys, config);
    const [processedDeleted, previewsDeleted] = await Promise.all([
      storeDeleteMatching(PROCESSED_STORE, (item) => itemMatchesVideo(item, sourceSite, videoId)),
      storeDeleteMatching(PREVIEW_STORE, (item) => itemMatchesVideo(item, sourceSite, videoId)),
    ]);
    libraryRowsCache = null;
    return {
      ok: true,
      video_id: videoId,
      source_site: sourceSite,
      visual_deleted: visualDeleted,
      transcript_deleted: transcriptDeleted,
      local_deleted: queueBefore + queueAfter + processedDeleted + previewsDeleted,
      stopped_tabs: stoppedTabs,
    };
  } finally {
    deletingVideos.delete(identity);
  }
}

async function videoLibraryStatus() {
  const config = await readConfig();
  await purgeAbandonedQueueItems();
  if (config.backendMode === "local") {
    const [library, queue] = await Promise.all([
      localServerRequest(config, "/api/extension-library"),
      storeAll(QUEUE_STORE),
    ]);
    return {
      ...library,
      queued_count: queue.length,
      capturing_count: 0,
      processing_items: Array.from(activeQueueItems.values()),
      billing_estimate: { total_gib: 0, storage_month_cny: 0, query_1000_cny: 0 },
      updated_at: Date.now(),
    };
  }
  if (!libraryRowsCache || Date.now() - libraryRowsCache.updated_at > 12000) {
    const filter = metadataFilter({
      embedding_model: config.embeddingModel,
      embedding_space: embeddingSpace(config),
    });
    const [visual, transcript] = await Promise.all([
      listVectors(config.ossVisualIndex, config, { filter }),
      listVectors(config.ossTranscriptIndex, config, { filter }),
    ]);
    libraryRowsCache = { visual, transcript, updated_at: Date.now() };
  }
  const [queue, processed, sessions] = await Promise.all([
    storeAll(QUEUE_STORE),
    storeAll(PROCESSED_STORE),
    storeAll(SESSION_STORE),
  ]);
  const videos = videoLibrarySummary(libraryRowsCache.visual, libraryRowsCache.transcript, queue, sessions);
  const activity = new Map();
  for (const item of processed) {
    const videoId = String(item.videoId || "");
    if (videoId) activity.set(videoId, Math.max(activity.get(videoId) || 0, Number(item.processedAt || 0)));
  }
  for (const item of queue) {
    const videoId = String(item.videoId || item.body?.video_id || "");
    if (videoId) activity.set(videoId, Math.max(activity.get(videoId) || 0, Number(item.createdAt || 0)));
  }
  for (const video of videos) {
    video.last_processed_at = Math.max(
      activity.get(video.video_id) || 0,
      Number(video.last_indexed_at || 0),
      Number(video.last_session_at || 0)
    );
  }
  videos.sort((left, right) =>
    Number(right.processing) - Number(left.processing)
    || Number(right.last_processed_at || 0) - Number(left.last_processed_at || 0)
    || right.pending_count - left.pending_count
    || right.visual_count - left.visual_count
  );
  return {
    videos,
    video_count: videos.length,
    visual_count: libraryRowsCache.visual.length,
    transcript_count: libraryRowsCache.transcript.length,
    billing_estimate: billingEstimate(
      libraryRowsCache.visual,
      libraryRowsCache.transcript,
      Number(config.embeddingDimension)
    ),
    queued_count: queue.length,
    capturing_count: videos.filter((video) => video.active_capture).length,
    processing_items: Array.from(activeQueueItems.values()),
    storage_provider: "oss_vectors_direct",
    storage_label: "阿里云 OSS Vector Bucket（插件直连）",
    updated_at: libraryRowsCache.updated_at,
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const run = async () => {
    const payload = message?.payload || {};
    const incomingVideoId = String(payload.videoId || payload.video_id || "");
    const incomingSourceSite = String(payload.sourceSite || payload.source_site || "youtube").toLowerCase();
    if (incomingVideoId && deletingVideos.has(videoIdentity(incomingSourceSite, incomingVideoId))) {
      return { skipped: true, reason: "video_deleting" };
    }
    if (message?.type === "FRAME_TICK") return captureAndQueue(message.payload, sender);
    if (message?.type === "VIDEO_FRAME") return queueVideoFrame(message.payload, sender);
    if (message?.type === "VIDEO_SEGMENT") return queueVideoSegment(message.payload);
    if (message?.type === "TRANSCRIPT_CHUNK") return enqueueTranscript(message.payload);
    if (message?.type === "VIDEO_SESSION") return recordVideoSession(message.payload || {});
    if (message?.type === "START_CHANNEL_REBUILD") return startChannelRebuild(message.payload || {}, sender);
    if (message?.type === "CHANNEL_VIDEO_COMPLETE") return completeRebuildVideo(message.payload || {}, sender);
    if (message?.type === "GET_REBUILD_STATUS") {
      const stored = await chrome.storage.local.get({ [REBUILD_JOB_KEY]: null });
      return stored[REBUILD_JOB_KEY];
    }
    if (message?.type === "QUEUE_STATUS") {
      await purgeAbandonedQueueItems();
      const queue = await storeAll(QUEUE_STORE);
      const paused = await processingPaused();
      const videoId = String(message.videoId || "");
      const scoped = videoId ? queue.filter((item) => String(item.videoId || "") === videoId) : queue;
      const activeQueue = scoped;
      const activeQueueTotal = queue;
      const failed = scoped.filter((item) => item.lastError);
      const indexes = (items) => Array.from(new Set(
        items
          .map((item) => Number(item.body?.segment_index))
          .filter((value) => Number.isInteger(value) && value >= 0)
      )).sort((a, b) => a - b);
      const visualItems = scoped.filter((item) => item.kind === "video" || item.kind === "visual");
      const transcriptItems = scoped.filter((item) => item.kind === "transcript");
      const latestFailure = failed.sort((left, right) => Number(left.lastErrorAt || 0) - Number(right.lastErrorAt || 0)).at(-1);
      return {
        queued: activeQueue.length,
        queued_total: activeQueueTotal.length,
        processing: Boolean(draining) && !paused,
        paused,
        errors: failed.length,
        needs_recapture: 0,
        failed_items: [],
        last_error: String(latestFailure?.lastError || ""),
        visual_segments: indexes(visualItems),
        transcript_segments: indexes(transcriptItems),
        failed_visual_segments: [],
      };
    }
    if (message?.type === "SET_PROCESSING_PAUSED") {
      const paused = Boolean(message.paused);
      await chrome.storage.local.set({ [PROCESSING_PAUSED_KEY]: paused });
      if (!paused) {
        await purgeAbandonedQueueItems();
        const queue = await storeAll(QUEUE_STORE);
        await Promise.all(queue.map((item) => storePut(QUEUE_STORE, {
          ...item,
          terminal: false,
          lastError: "",
          lastErrorAt: 0,
          nextAttempt: 0,
        })));
        void drainQueue();
      }
      return { ok: true, paused };
    }
    if (message?.type === "SEARCH") return directSearch(message.payload || {});
    if (message?.type === "IMAGE_SEARCH") return directImageSearch(message.payload || {});
    if (message?.type === "DELETE_VIDEO") return deleteVideoMemory(message.payload || {});
    if (message?.type === "STATS") return directStats(String(message.videoId || ""));
    if (message?.type === "VIDEO_LIBRARY") return videoLibraryStatus();
    if (message?.type === "GET_CONFIG") {
      const config = await readConfig();
      return {
        ...config,
        ossAccessKeySecret: "",
        ossSecurityToken: "",
        maskedAccessKeyId: maskCredential(config.ossAccessKeyId),
        maskedAccessKeySecret: maskCredential(config.ossAccessKeySecret),
        hasAccessKeySecret: Boolean(config.ossAccessKeySecret),
        hasSecurityToken: Boolean(config.ossSecurityToken),
      };
    }
    if (message?.type === "SAVE_CONFIG") {
      const current = await readConfig();
      const incoming = message.config || {};
      const nextEmbeddingBaseUrl = String(incoming.embeddingBaseUrl || current.embeddingBaseUrl).replace(/\/+$/, "");
      const nextEmbeddingModel = String(incoming.embeddingModel || current.embeddingModel);
      const nextEmbeddingDimension = Number(incoming.embeddingDimension || current.embeddingDimension);
      const nextEmbeddingInputStyle = String(incoming.embeddingInputStyle || current.embeddingInputStyle || "auto");
      const embeddingIdentityChanged = nextEmbeddingBaseUrl !== String(current.embeddingBaseUrl || "").replace(/\/+$/, "")
        || nextEmbeddingModel !== String(current.embeddingModel || "")
        || nextEmbeddingDimension !== Number(current.embeddingDimension || 0)
        || nextEmbeddingInputStyle !== String(current.embeddingInputStyle || "auto");
      const hasIncomingEmbeddingSpace = Object.prototype.hasOwnProperty.call(incoming, "embeddingSpace");
      const next = {
        ...current,
        backendMode: incoming.backendMode
          ? (incoming.backendMode === "local" ? "local" : "cloud")
          : current.backendMode,
        controlPlaneUrl: String(incoming.controlPlaneUrl || current.controlPlaneUrl || DIRECT_DEFAULT_CONFIG.controlPlaneUrl).replace(/\/+$/, ""),
        embeddingBaseUrl: nextEmbeddingBaseUrl,
        embeddingModel: nextEmbeddingModel,
        embeddingDimension: nextEmbeddingDimension,
        embeddingInputStyle: nextEmbeddingInputStyle,
        embeddingSpace: hasIncomingEmbeddingSpace
          ? String(incoming.embeddingSpace || "")
          : (embeddingIdentityChanged ? "" : String(current.embeddingSpace || "")),
        ossRegion: String(incoming.ossRegion || current.ossRegion),
        ossAccountId: String(incoming.ossAccountId || current.ossAccountId),
        ossBucket: String(incoming.ossBucket || current.ossBucket),
        ossVisualIndex: String(incoming.ossVisualIndex || current.ossVisualIndex),
        ossTranscriptIndex: String(incoming.ossTranscriptIndex || current.ossTranscriptIndex),
        ossAccessKeyId: String(incoming.ossAccessKeyId || current.ossAccessKeyId),
      };
      if (incoming.ossAccessKeySecret) next.ossAccessKeySecret = String(incoming.ossAccessKeySecret);
      if (incoming.ossSecurityToken) next.ossSecurityToken = String(incoming.ossSecurityToken);
      if (incoming.clearAccessKeySecret) next.ossAccessKeySecret = "";
      if (incoming.clearSecurityToken) next.ossSecurityToken = "";
      await saveSyncedConfig(next);
      await chrome.storage.local.set(pickConfig(next, LOCAL_CREDENTIAL_KEYS));
      libraryRowsCache = null;
      return {
        ok: true,
        mode: next.backendMode,
        endpoint: next.backendMode === "local" ? next.controlPlaneUrl : vectorEndpoint(next),
        maskedAccessKeyId: maskCredential(next.ossAccessKeyId),
        maskedAccessKeySecret: maskCredential(next.ossAccessKeySecret),
        hasAccessKeySecret: Boolean(next.ossAccessKeySecret),
        hasSecurityToken: Boolean(next.ossSecurityToken),
      };
    }
    if (message?.type === "TEST_CONFIG") {
      const current = await readConfig();
      const incoming = message.config || {};
      const proposed = { ...current, ...incoming };
      if (!incoming.ossAccessKeySecret) proposed.ossAccessKeySecret = current.ossAccessKeySecret;
      if (!incoming.ossSecurityToken) proposed.ossSecurityToken = current.ossSecurityToken;
      return proposed.backendMode === "local"
        ? localServerRequest(proposed, "/api/status")
        : testDirectConfig(proposed);
    }
    if (message?.type === "DISCOVER_EMBEDDING") {
      const current = await readConfig();
      return { ok: true, ...(await discoverEmbeddingConfig({ ...current, ...(message.config || {}) })) };
    }
    if (message?.type === "RESET_VIDEO_MEMORY_INFO") return resetVideoMemoryInfo();
    if (message?.type === "RESET_ALL_VIDEO_MEMORY") return resetAllVideoMemory(message.payload || {});
    if (message?.type === "API_INFO") {
      const config = await readConfig();
      return { mode: config.backendMode, baseUrl: config.backendMode === "local" ? config.controlPlaneUrl : "" };
    }
    return null;
  };
  void run().then(sendResponse).catch((error) => sendResponse({ error: String(error.message || error) }));
  return true;
});

async function injectPlayerControlsIntoOpenTabs() {
  const tabs = await chrome.tabs.query({
    url: [
      "https://www.youtube.com/watch*",
      "https://www.bilibili.com/video/*",
    ],
  });
  await Promise.all(tabs.map(async (tab) => {
    if (!tab.id) return;
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content-script.js"],
      });
    } catch {}
  }));
}

chrome.alarms.create("drain-index-queue", { periodInMinutes: 1 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "drain-index-queue") void drainQueue();
});
chrome.runtime.onStartup.addListener(() => {
  void restrictSyncedConfigAccess();
  void injectPlayerControlsIntoOpenTabs();
  void drainQueue();
});
chrome.runtime.onInstalled.addListener(() => {
  void (async () => {
    const config = await readConfig();
    await restrictSyncedConfigAccess();
    try { await saveSyncedConfig(config); } catch {}
    await chrome.storage.local.set(pickConfig(config, LOCAL_CREDENTIAL_KEYS));
    await injectPlayerControlsIntoOpenTabs();
    await purgeAbandonedQueueItems();
    if (await processingPaused()) return;
    const queue = await storeAll(QUEUE_STORE);
    await Promise.all(queue.map((item) => storePut(QUEUE_STORE, {
      ...item,
      terminal: false,
      lastError: "",
      lastErrorAt: 0,
      nextAttempt: 0,
    })));
    await drainQueue();
  })();
});
