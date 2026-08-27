// @ts-nocheck -- compatibility port; typed public contracts are being introduced around this proven client.
import { PRODUCT_DEFAULTS } from "@indexed/contracts";

const encoder = new TextEncoder();

export const DIRECT_DEFAULT_CONFIG = {
  backendMode: "cloud",
  controlPlaneUrl: `http://${PRODUCT_DEFAULTS.server.host}:${PRODUCT_DEFAULTS.server.port}`,
  embeddingBaseUrl: "",
  embeddingModel: "",
  embeddingDimension: PRODUCT_DEFAULTS.vectorDimension,
  embeddingInputStyle: "auto",
  embeddingSpace: "",
  ossRegion: PRODUCT_DEFAULTS.ossRegion,
  ossAccountId: "",
  ossBucket: "",
  ossVisualIndex: "",
  ossTranscriptIndex: "",
  ossAccessKeyId: "",
  ossAccessKeySecret: "",
  ossSecurityToken: "",
};

export function embeddingInputStyle(config) {
  const configured = String(config.embeddingInputStyle || "auto").trim().toLowerCase();
  if (configured && configured !== "auto") return configured;
  return String(config.embeddingModel || "").toLowerCase().includes("wemm") ? "wemm" : "qwen";
}

export function embeddingSpace(config) {
  const explicit = String(config.embeddingSpace || "").trim();
  if (explicit) return explicit;
  const model = String(config.embeddingModel || "unknown").trim().toLowerCase();
  const dimension = Number(config.embeddingDimension || 0);
  return `${model}-${dimension}-${embeddingInputStyle(config)}-indexed-v1`.replace(/[^a-z0-9._-]+/g, "-");
}

const VISUAL_QUERY_INSTRUCTION =
  "Represent this text query for retrieving matching screenshots, software interfaces, diagrams, demonstrations, scenes, and video frames.";
const VISUAL_CONTENT_INSTRUCTION =
  "Represent the input for retrieving visually and semantically similar video shots.";
const TRANSCRIPT_INSTRUCTION =
  "Represent the spoken content of this video segment for semantic retrieval.";
const DOCUMENT_INSTRUCTION =
  "Represent this local document or file description for semantic retrieval and project organization.";

function normalizeUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function bytesToHex(value) {
  return Array.from(value, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function digest(algorithm, value) {
  const bytes = typeof value === "string" ? encoder.encode(value) : value;
  return new Uint8Array(await crypto.subtle.digest(algorithm, bytes));
}

async function hmac(key, value) {
  const cryptoKey = await crypto.subtle.importKey(
    "raw",
    typeof key === "string" ? encoder.encode(key) : key,
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  return new Uint8Array(await crypto.subtle.sign("HMAC", cryptoKey, encoder.encode(value)));
}

function utcStamp(now = new Date()) {
  return now.toISOString().replace(/[:-]|\.\d{3}/g, "");
}

function canonicalQuery(operation) {
  return encodeURIComponent(operation).replace(/%[0-9a-f]{2}/g, (value) => value.toUpperCase());
}

export function vectorEndpoint(config) {
  const bucket = String(config.ossBucket || "").trim();
  const accountId = String(config.ossAccountId || "").trim();
  const region = String(config.ossRegion || "").trim();
  if (!bucket || !accountId || !region) throw new Error("请填写阿里云 Region、账号 ID 和 Vector Bucket");
  return `${bucket}-${accountId}.${region}.oss-vectors.aliyuncs.com`;
}

function vectorCanonicalUri(config) {
  const resource = `acs:ossvector:${String(config.ossRegion).trim()}:${String(config.ossAccountId).trim()}:${String(config.ossBucket).trim()}`;
  return `/${encodeURIComponent(resource).replace(/%[0-9a-f]{2}/g, (value) => value.toUpperCase())}/`;
}

export async function signVectorRequest(operation, config, now = new Date()) {
  const accessKeyId = String(config.ossAccessKeyId || "").trim();
  const accessKeySecret = String(config.ossAccessKeySecret || "");
  if (!accessKeyId || !accessKeySecret) throw new Error("请填写阿里云 AccessKey ID 和 Secret");
  const timestamp = utcStamp(now);
  const date = timestamp.slice(0, 8);
  const region = String(config.ossRegion || "").trim();
  const scope = `${date}/${region}/oss/aliyun_v4_request`;
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "x-oss-content-sha256": "UNSIGNED-PAYLOAD",
    "x-oss-date": timestamp,
  };
  const securityToken = String(config.ossSecurityToken || "").trim();
  if (securityToken) headers["x-oss-security-token"] = securityToken;
  const canonicalHeaders = Object.entries(headers)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}:${String(value).trim()}\n`)
    .join("");
  const canonicalRequest = [
    "POST",
    vectorCanonicalUri(config),
    canonicalQuery(operation),
    canonicalHeaders,
    "",
    "UNSIGNED-PAYLOAD",
  ].join("\n");
  const requestHash = bytesToHex(await digest("SHA-256", canonicalRequest));
  const stringToSign = `OSS4-HMAC-SHA256\n${timestamp}\n${scope}\n${requestHash}`;
  const dateKey = await hmac(`aliyun_v4${accessKeySecret}`, date);
  const regionKey = await hmac(dateKey, region);
  const serviceKey = await hmac(regionKey, "oss");
  const signingKey = await hmac(serviceKey, "aliyun_v4_request");
  const signature = bytesToHex(await hmac(signingKey, stringToSign));
  headers.authorization = `OSS4-HMAC-SHA256 Credential=${accessKeyId}/${scope}, Signature=${signature}`;
  return {
    url: `https://${vectorEndpoint(config)}/?${operation}`,
    headers,
    canonicalRequest,
    stringToSign,
    signature,
  };
}

export async function vectorRequest(operation, body, config) {
  const signed = await signVectorRequest(operation, config);
  const response = await fetch(signed.url, {
    method: "POST",
    headers: signed.headers,
    body: JSON.stringify(body || {}),
  });
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = {};
  }
  if (!response.ok) {
    const detail = payload.message || payload.Message || payload.code || payload.Code || text.slice(0, 500);
    throw new Error(`OSS Vectors ${operation} 返回 ${response.status}: ${detail}`);
  }
  return payload;
}

export async function putVectors(indexName, vectors, config) {
  if (!vectors.length) return {};
  return vectorRequest("putVectors", { indexName, vectors }, config);
}

export async function deleteVectors(indexName, keys, config) {
  const uniqueKeys = Array.from(new Set((keys || []).map((key) => String(key || "")).filter(Boolean)));
  if (!uniqueKeys.length) return {};
  if (uniqueKeys.length > 500) throw new Error("DeleteVectors 单次最多删除 500 个 key");
  return vectorRequest("deleteVectors", { indexName, keys: uniqueKeys }, config);
}

export async function getVectors(indexName, keys, config, options = {}) {
  if (!keys.length) return { vectors: [] };
  return vectorRequest(
    "getVectors",
    {
      indexName,
      keys,
      returnData: Boolean(options.returnData),
      returnMetadata: Boolean(options.returnMetadata),
    },
    config
  );
}

export async function queryVectors(indexName, vector, config, options = {}) {
  const body = {
    indexName,
    queryVector: { float32: vector },
    topK: Math.min(500, Math.max(1, Number(options.limit || 60))),
    returnDistance: true,
    returnMetadata: true,
  };
  if (options.filter && Object.keys(options.filter).length) body.filter = options.filter;
  return vectorRequest("queryVectors", body, config);
}

export async function listVectors(indexName, config, options = {}) {
  const output = [];
  const maxItems = Number(options.maxItems || 0);
  let nextToken = "";
  do {
    const body = {
      indexName,
      maxResults: Math.min(1000, Math.max(1, Number(options.pageSize || 500))),
      returnData: false,
      returnMetadata: options.returnMetadata !== false,
    };
    if (nextToken) body.nextToken = nextToken;
    if (options.filter && Object.keys(options.filter).length) body.filter = options.filter;
    const page = await vectorRequest("listVectors", body, config);
    output.push(...(page.vectors || []));
    if (maxItems && output.length >= maxItems) return output.slice(0, maxItems);
    nextToken = String(page.nextToken || "");
  } while (nextToken);
  return output;
}

function extractVector(body) {
  if (Array.isArray(body)) {
    if (body.length && body.every((value) => typeof value === "number")) return body.map(Number);
    for (const item of body) {
      const found = extractVector(item);
      if (found.length) return found;
    }
    return [];
  }
  if (!body || typeof body !== "object") return [];
  for (const key of ["embedding", "embeddings", "pooled_hidden_states", "last_hidden_state", "data", "output"]) {
    if (!(key in body)) continue;
    const found = extractVector(body[key]);
    if (found.length) return found;
  }
  return [];
}

async function embeddingRequest(messages, config) {
  const baseUrl = normalizeUrl(config.embeddingBaseUrl);
  if (!baseUrl) throw new Error("请填写 Embedding 服务地址");
  const response = await fetch(`${baseUrl}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: config.embeddingModel, messages }),
  });
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = {};
  }
  if (!response.ok) throw new Error(`Embedding 服务返回 ${response.status}: ${text.slice(0, 500)}`);
  const vector = extractVector(payload);
  const expected = Number(config.embeddingDimension || 0);
  if (!vector.length) throw new Error("Embedding 响应中没有向量");
  if (expected && vector.length !== expected) throw new Error(`向量维度 ${vector.length}，预期 ${expected}`);
  return vector;
}

export function textMessages(text, instruction) {
  return [
    { role: "system", content: [{ type: "text", text: instruction }] },
    { role: "user", content: [{ type: "text", text: String(text || "").trim() }] },
  ];
}

export async function embedVisualQuery(text, config) {
  return embeddingRequest(textMessages(text, VISUAL_QUERY_INSTRUCTION), config);
}

export async function embedTranscript(text, config) {
  return embeddingRequest(textMessages(text, TRANSCRIPT_INSTRUCTION), config);
}

export async function embedDocument(text, config) {
  return embeddingRequest(textMessages(text, DOCUMENT_INSTRUCTION), config);
}

export async function embedImage(imageBase64, mimeType, text, config) {
  const content = [
    { type: "image_url", image_url: { url: `data:${mimeType || "image/jpeg"};base64,${imageBase64}` } },
  ];
  if (String(text || "").trim()) content.push({ type: "text", text: String(text).trim() });
  if (embeddingInputStyle(config) === "wemm") {
    if (!String(text || "").trim()) content.push({ type: "text", text: "Represent this image." });
    return embeddingRequest([{ role: "user", content }], config);
  }
  return embeddingRequest([
    { role: "system", content: [{ type: "text", text: VISUAL_CONTENT_INSTRUCTION }] },
    { role: "user", content },
  ], config);
}

export async function embedVideo(videoBase64, mimeType, config) {
  const content = [
    { type: "video_url", video_url: { url: `data:${mimeType || "video/webm"};base64,${videoBase64}` } },
  ];
  if (embeddingInputStyle(config) === "wemm") {
    content.push({ type: "text", text: "Represent this video." });
    return embeddingRequest([{ role: "user", content }], config);
  }
  return embeddingRequest([
    { role: "system", content: [{ type: "text", text: VISUAL_CONTENT_INSTRUCTION }] },
    { role: "user", content },
  ], config);
}

export async function uuid5Url(value) {
  const namespace = "6ba7b8119dad11d180b400c04fd430c8";
  const namespaceBytes = new Uint8Array(namespace.match(/../g).map((part) => parseInt(part, 16)));
  const valueBytes = encoder.encode(value);
  const merged = new Uint8Array(namespaceBytes.length + valueBytes.length);
  merged.set(namespaceBytes);
  merged.set(valueBytes, namespaceBytes.length);
  const hash = await digest("SHA-1", merged);
  const bytes = hash.slice(0, 16);
  bytes[6] = (bytes[6] & 0x0f) | 0x50;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = bytesToHex(bytes);
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function scoreFromDistance(distance) {
  return Math.max(-1, Math.min(1, 1 - Number(distance || 0)));
}

export function metadataFilter(values = {}) {
  const filters = Object.entries(values)
    .filter(([, value]) => String(value || "").trim())
    .map(([key, value]) => ({ [key]: { $eq: String(value).trim() } }));
  if (!filters.length) return null;
  return filters.length === 1 ? filters[0] : { $and: filters };
}

export async function testDirectConfig(config) {
  const baseUrl = normalizeUrl(config.embeddingBaseUrl);
  const modelResponse = await fetch(`${baseUrl}/v1/models`);
  if (!modelResponse.ok) throw new Error(`Embedding 健康检查返回 ${modelResponse.status}`);
  const modelBody = await modelResponse.json().catch(() => ({}));
  const rows = await listVectors(config.ossVisualIndex, config, {
    pageSize: 1,
    maxItems: 1,
    returnMetadata: false,
  });
  return {
    ok: true,
    embedding: { model: modelBody.data?.[0]?.id || config.embeddingModel },
    storage: { provider: "oss_vectors_direct", provider_label: "阿里云 OSS Vector Bucket（插件直连）" },
    sampleCount: rows.length,
  };
}

export async function testEmbeddingConfig(config) {
  const baseUrl = normalizeUrl(config.embeddingBaseUrl);
  if (!baseUrl) throw new Error("请填写 Embedding 服务地址");
  const response = await fetch(`${baseUrl}/v1/models`);
  if (!response.ok) throw new Error(`Embedding 健康检查返回 ${response.status}`);
  const body = await response.json().catch(() => ({}));
  return { model: body.data?.[0]?.id || config.embeddingModel };
}
