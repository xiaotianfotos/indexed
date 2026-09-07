import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zvec, { type ZVecCollection, type ZVecDoc, type ZVecDocInput } from "@zvec/zvec";

export type VectorRecord = {
  key: string;
  data?: { float32?: number[] };
  metadata?: Record<string, unknown>;
  distance?: number;
};

export type LocalVectorConfig = {
  localStorePath?: string;
  embeddingSpace?: string;
};

export interface LocalVectorSessionTimings {
  opens: number;
  openMs: number;
  writeMs: number;
  closeMs: number;
}

type CollectionSession = {
  references: number;
  collections: Map<string, ZVecCollection>;
  timings: LocalVectorSessionTimings;
};
const collectionSessions = new Map<string, CollectionSession>();
const leasedCollections = new WeakSet<ZVecCollection>();

/** One process owns a bounded scan session; concurrent reads share its handles.
 * All native calls below are synchronous, so no iterator can span a write.
 * Closing in finally releases handles on success, cancellation and failure.
 */
export async function withLocalVectorSession<T>(
  config: LocalVectorConfig,
  task: () => Promise<T>,
  onTimings?: (timings: LocalVectorSessionTimings) => void,
): Promise<T> {
  const root = localVectorDatabasePath(config);
  const session = collectionSessions.get(root) || {
    references: 0, collections: new Map<string, ZVecCollection>(),
    timings: { opens: 0, openMs: 0, writeMs: 0, closeMs: 0 },
  };
  session.references += 1;
  collectionSessions.set(root, session);
  try {
    return await task();
  } finally {
    session.references -= 1;
    if (!session.references) {
      collectionSessions.delete(root);
      const started = performance.now();
      let failure: unknown;
      for (const collection of session.collections.values()) {
        leasedCollections.delete(collection);
        try { collection.closeSync(); } catch (error) { failure ??= error; }
      }
      session.timings.closeMs += performance.now() - started;
      onTimings?.({ ...session.timings });
      if (failure) throw failure;
    } else {
      onTimings?.({ ...session.timings });
    }
  }
}

function retainCollection(root: string, name: string, open: () => ZVecCollection): ZVecCollection {
  const session = collectionSessions.get(root);
  const existing = session?.collections.get(name);
  if (existing) return existing;
  const started = performance.now();
  const collection = open();
  if (session) {
    session.collections.set(name, collection);
    leasedCollections.add(collection);
    session.timings.opens += 1;
    session.timings.openMs += performance.now() - started;
  }
  return collection;
}

function releaseCollection(collection: ZVecCollection): void {
  if (!leasedCollections.has(collection)) collection.closeSync();
}

export type LocalVectorIdentity = { indexName: string; embeddingSpace: string };

type CollectionManifest = {
  dimension: number;
  updatedAt: string;
  identity?: LocalVectorIdentity;
};

type StoreManifest = {
  format: "indexed-zvec";
  version: 1 | 2;
  collections: Record<string, CollectionManifest>;
};

const MANIFEST_FILE = ".indexed-zvec.json";
const VECTOR_FIELD = "vector";
const INDEX_SCHEMA_VERSION = 2;
const STRING_FIELDS = [
  "metadata_json",
  "embedding_model",
  "embedding_space",
  "source_site",
  "video_id",
  "channel_id",
  "record_type",
  "file_kind",
  "library_root",
  "asset_path",
  "source_path",
  "modality",
] as const;
const FILTER_FIELDS = new Set<string>(STRING_FIELDS.filter((field) => field !== "metadata_json"));

export function localVectorDatabasePath(config: LocalVectorConfig) {
  const configured = String(config.localStorePath || "").trim();
  const indexedHome = String(process.env.INDEXED_HOME || "").trim();
  const selected = configured
    || process.env.INDEXED_DATA_DIR
    || (indexedHome ? path.join(indexedHome, "data", "zvec") : "")
    || path.join(os.homedir(), ".local", "share", "indexed", "zvec");
  const expanded = selected === "~"
    ? os.homedir()
    : selected.startsWith("~/")
      ? path.join(os.homedir(), selected.slice(2))
      : selected;
  return path.resolve(expanded);
}

function safeName(value: string, fallback: string) {
  return String(value || fallback).trim().replace(/[^a-zA-Z0-9_]+/g, "_") || fallback;
}

export function localVectorCollectionName(indexName: string, config: LocalVectorConfig) {
  const identity = vectorIdentity(indexName, config);
  const digest = crypto.createHash("sha256").update(JSON.stringify([identity.indexName, identity.embeddingSpace])).digest("hex");
  return `v2_${safeName(indexName, "vectors").slice(0, 24)}_${digest}`;
}

function vectorIdentity(indexName: string, config: LocalVectorConfig): LocalVectorIdentity {
  if (!indexName || !indexName.trim()) throw new Error("逻辑索引名不能为空");
  return { indexName, embeddingSpace: config.embeddingSpace ?? "" };
}

function sameIdentity(left: LocalVectorIdentity | undefined, right: LocalVectorIdentity) {
  return left?.indexName === right.indexName && left.embeddingSpace === right.embeddingSpace;
}

function assertCurrentFormat(root: string, manifest = readManifest(root)): void {
  if (manifest.version === 1 && Object.keys(manifest.collections).length) {
    throw new Error(`zvec 测试索引格式已过期：${root}；请将 storage.path 设置为新的空目录，再重新扫描建立索引；旧目录未被修改`);
  }
}

function logicalCollection(root: string, indexName: string, config: LocalVectorConfig): string {
  const manifest = readManifest(root);
  assertCurrentFormat(root, manifest);
  const name = localVectorCollectionName(indexName, config);
  const existing = manifest.collections[name];
  if (existing && !sameIdentity(existing.identity, vectorIdentity(indexName, config))) {
    throw new Error(`zvec 集合的原始身份与请求不一致：${name}`);
  }
  return name;
}

function manifestPath(root: string) {
  return path.join(root, MANIFEST_FILE);
}

function emptyManifest(): StoreManifest {
  return { format: "indexed-zvec", version: 2, collections: {} };
}

function readManifest(root: string): StoreManifest {
  const target = manifestPath(root);
  if (!fs.existsSync(target)) {
    const legacyLanceDb = fs.existsSync(root)
      && fs.readdirSync(root).some((entry) => entry.endsWith(".lance"));
    if (legacyLanceDb) {
      throw new Error(`配置路径仍是 LanceDB，不能当作 zvec 打开：${root}；请将 storage.path 设置为新的空目录，再重新扫描建立索引；旧目录未被修改`);
    }
    return emptyManifest();
  }
  const parsed = JSON.parse(fs.readFileSync(target, "utf8")) as Partial<StoreManifest>;
  if (parsed.format !== "indexed-zvec" || (parsed.version !== 1 && parsed.version !== 2)
    || !parsed.collections || typeof parsed.collections !== "object" || Array.isArray(parsed.collections)) {
    throw new Error(`不支持的 zvec 数据库清单：${target}`);
  }
  for (const [name, entry] of Object.entries(parsed.collections)) {
    if (!/^[a-zA-Z0-9_]{1,255}$/.test(name) || !entry || !Number.isInteger(entry.dimension) || entry.dimension < 1) {
      throw new Error(`zvec 数据库清单包含无效集合：${name}`);
    }
    if (parsed.version === 2 && (!entry.identity || typeof entry.identity.indexName !== "string"
      || typeof entry.identity.embeddingSpace !== "string"
      || localVectorCollectionName(entry.identity.indexName, entry.identity) !== name)) {
      throw new Error(`zvec 集合身份校验失败：${name}`);
    }
  }
  return parsed as StoreManifest;
}

function writeManifest(root: string, manifest: StoreManifest) {
  fs.mkdirSync(root, { recursive: true });
  const target = manifestPath(root);
  const temporary = `${target}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(manifest, null, 2)}\n`, { mode: 0o600 });
  fs.renameSync(temporary, target);
}

function collectionPath(root: string, collectionName: string) {
  if (!/^[a-zA-Z0-9_]{1,255}$/.test(collectionName)) throw new Error(`无效物理集合名：${collectionName}`);
  return path.join(root, collectionName);
}

function schemaName(collectionName: string) {
  const readable = safeName(collectionName, "vectors").slice(0, 36);
  const identity = crypto.createHash("sha256").update(collectionName).digest("hex").slice(0, 16);
  // Zvec validates schema names more narrowly than collection directory paths.
  return `indexed_${readable}_${identity}`;
}

function collectionSchema(collectionName: string, dimension: number) {
  const indexedString = (name: string) => ({
    name,
    dataType: zvec.ZVecDataType.STRING,
    indexParams: { indexType: zvec.ZVecIndexType.INVERT },
  });
  return new zvec.ZVecCollectionSchema({
    // Keep the stable physical collection name on disk while satisfying the
    // shorter internal schema-name limit for long logical index names.
    name: schemaName(collectionName),
    vectors: {
      name: VECTOR_FIELD,
      dataType: zvec.ZVecDataType.VECTOR_FP32,
      dimension,
      // Exact cosine search is preferable for the current local corpus. The
      // on-disk schema is versioned so a future HNSW/DiskANN migration is explicit.
      indexParams: {
        indexType: zvec.ZVecIndexType.FLAT,
        metricType: zvec.ZVecMetricType.COSINE,
      },
    },
    fields: [
      indexedString("embedding_model"),
      indexedString("embedding_space"),
      indexedString("source_site"),
      indexedString("video_id"),
      indexedString("channel_id"),
      indexedString("record_type"),
      indexedString("file_kind"),
      indexedString("library_root"),
      indexedString("asset_path"),
      indexedString("source_path"),
      indexedString("modality"),
      { name: "metadata_json", dataType: zvec.ZVecDataType.STRING },
      // Preserve exact FP32 values independently of cosine normalization in the engine.
      { name: "vector_f32", dataType: zvec.ZVecDataType.STRING },
      { name: "indexed_at_ms", dataType: zvec.ZVecDataType.INT64 },
      { name: "index_schema_version", dataType: zvec.ZVecDataType.INT32 },
    ],
  });
}

function openCollection(root: string, collectionName: string): ZVecCollection | null {
  const manifest = readManifest(root);
  assertCurrentFormat(root, manifest);
  if (!Object.hasOwn(manifest.collections, collectionName)) return null;
  return retainCollection(root, collectionName, () => zvec.ZVecOpen(collectionPath(root, collectionName)));
}

function openOrCreateCollection(root: string, collectionName: string, dimension: number, identity: LocalVectorIdentity): ZVecCollection {
  if (!Number.isInteger(dimension) || dimension < 1) {
    throw new Error(`zvec collection ${collectionName} 的向量维度无效：${dimension}`);
  }
  fs.mkdirSync(root, { recursive: true });
  const manifest = readManifest(root);
  assertCurrentFormat(root, manifest);
  const registered = manifest.collections[collectionName];
  if (registered) {
    if (!sameIdentity(registered.identity, identity)) throw new Error(`zvec 集合身份不一致：${collectionName}`);
    if (registered.dimension !== dimension) {
      throw new Error(
        `zvec collection ${collectionName} 的向量维度为 ${registered.dimension}，不能写入 ${dimension} 维向量`,
      );
    }
    return retainCollection(root, collectionName, () => zvec.ZVecOpen(collectionPath(root, collectionName)));
  }
  const target = collectionPath(root, collectionName);
  if (fs.existsSync(target)) {
    throw new Error(`zvec collection 目录已存在但未登记，拒绝覆盖：${target}`);
  }
  const collection = zvec.ZVecCreateAndOpen(target, collectionSchema(collectionName, dimension));
  manifest.version = 2;
  manifest.collections[collectionName] = { dimension, updatedAt: new Date().toISOString(), identity };
  try {
    writeManifest(root, manifest);
  } catch (error) {
    collection.closeSync();
    throw error;
  }
  return retainCollection(root, collectionName, () => collection);
}

function fieldsOf(record: VectorRecord) {
  const metadata = record.metadata || {};
  const vector = record.data?.float32 || [];
  const bytes = Buffer.allocUnsafe(vector.length * 4);
  vector.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
  return {
    metadata_json: JSON.stringify(metadata),
    vector_f32: bytes.toString("base64"),
    embedding_model: String(metadata.embedding_model || ""),
    embedding_space: String(metadata.embedding_space || ""),
    source_site: String(metadata.source_site || ""),
    video_id: String(metadata.video_id || ""),
    channel_id: String(metadata.channel_id || ""),
    record_type: String(metadata.record_type || ""),
    file_kind: String(metadata.file_kind || ""),
    library_root: String(metadata.library_root || ""),
    asset_path: String(metadata.asset_path || ""),
    source_path: String(metadata.source_path || ""),
    modality: String(metadata.modality || ""),
    indexed_at_ms: Number(metadata.indexed_at_ms || Date.now()),
    index_schema_version: INDEX_SCHEMA_VERSION,
  };
}

function input(record: VectorRecord): ZVecDocInput {
  const vector = Array.from(record.data?.float32 || [], Number);
  return {
    id: String(record.key),
    vectors: { [VECTOR_FIELD]: vector },
    fields: fieldsOf(record),
  };
}

function metadataOf(doc: ZVecDoc) {
  try {
    const value: unknown = JSON.parse(String(doc.fields.metadata_json || "{}"));
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("metadata must be an object");
    return value as Record<string, unknown>;
  } catch {
    throw new Error(`zvec 记录 ${doc.id} 的 metadata_json 无效`);
  }
}

function record(doc: ZVecDoc, returnData = true): VectorRecord {
  const stored = doc.vectors[VECTOR_FIELD];
  const exact = returnData && typeof doc.fields.vector_f32 === "string" && doc.fields.vector_f32
    ? Buffer.from(doc.fields.vector_f32, "base64") : undefined;
  if (exact && exact.length % 4) throw new Error(`zvec 记录 ${doc.id} 的 FP32 原值无效`);
  const values = exact ? Array.from({ length: exact.length / 4 }, (_, index) => exact.readFloatLE(index * 4))
    : returnData && stored ? Array.from(stored as ArrayLike<number>, Number) : undefined;
  return {
    key: doc.id,
    ...(values ? { data: { float32: values } } : {}),
    metadata: metadataOf(doc),
    ...(Number.isFinite(doc.score) ? { distance: Number(doc.score) } : {}),
  };
}

function valuesFromFilter(filter: unknown, output: Record<string, string> = {}) {
  if (!filter || typeof filter !== "object") return output;
  if (Array.isArray(filter)) {
    filter.forEach((entry) => valuesFromFilter(entry, output));
    return output;
  }
  for (const [key, value] of Object.entries(filter as Record<string, unknown>)) {
    if (key === "$and") valuesFromFilter(value, output);
    else if (value && typeof value === "object" && "$eq" in value) {
      output[key] = String((value as { $eq: unknown }).$eq);
    }
  }
  return output;
}

function matches(metadata: Record<string, unknown>, filter: unknown) {
  return Object.entries(valuesFromFilter(filter))
    .every(([key, value]) => String(metadata[key] ?? "") === value);
}

function escapeFilterString(value: string) {
  return value.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
}

function zvecFilter(filter: unknown) {
  const values = valuesFromFilter(filter);
  const unsupported = Object.keys(values).some((field) => !FILTER_FIELDS.has(field));
  const clauses = Object.entries(values)
    .filter(([field]) => FILTER_FIELDS.has(field))
    .map(([field, value]) => `${field} = '${escapeFilterString(value)}'`);
  return { expression: clauses.join(" AND "), fullySupported: !unsupported };
}

function assertStatuses(statuses: ReturnType<ZVecCollection["upsertSync"]>) {
  const values = Array.isArray(statuses) ? statuses : [statuses];
  const failed = values.find((status) => !status.ok);
  if (failed) throw new Error(`zvec 写入失败：${failed.code} ${failed.message}`.trim());
}

/** Internal physical write; public callers must resolve a logical identity first. */
async function writeCollection(
  collectionName: string,
  vectors: VectorRecord[],
  config: LocalVectorConfig,
  identity: LocalVectorIdentity,
) {
  if (!vectors.length) return { inserted: 0 };
  if (collectionName !== localVectorCollectionName(identity.indexName, identity)) throw new Error("物理集合名与原始身份不一致");
  for (const row of vectors) {
    if (String(row.metadata?.embedding_space ?? "") !== identity.embeddingSpace) {
      throw new Error(`向量 ${row.key} 的 embedding_space 与目标空间不一致`);
    }
    if (!row.key || !row.data?.float32?.every((value) => Number.isFinite(value) && Number.isFinite(Math.fround(value)))) {
      throw new Error("向量 key 不能为空，数值必须在有限 Float32 范围内");
    }
  }
  const dimensions = new Set(vectors.map((item) => item.data?.float32?.length || 0));
  if (dimensions.size !== 1 || dimensions.has(0)) {
    throw new Error(`zvec collection ${collectionName} 的导入批次包含空向量或混合维度`);
  }
  const root = localVectorDatabasePath(config);
  const target = openOrCreateCollection(root, collectionName, [...dimensions][0] as number, identity);
  try {
    const started = performance.now();
    assertStatuses(target.upsertSync(vectors.map(input)));
    const session = collectionSessions.get(root);
    if (session) session.timings.writeMs += performance.now() - started;
    return { inserted: vectors.length };
  } finally {
    releaseCollection(target);
  }
}

export async function putLocalVectors(indexName: string, vectors: VectorRecord[], config: LocalVectorConfig) {
  const name = logicalCollection(localVectorDatabasePath(config), indexName, config);
  return writeCollection(name, vectors, config, vectorIdentity(indexName, config));
}

export async function deleteLocalVectors(indexName: string, keys: string[], config: LocalVectorConfig) {
  const unique = [...new Set(keys.map(String).filter(Boolean))];
  if (!unique.length) return { deleted: 0 };
  const root = localVectorDatabasePath(config);
  const target = openCollection(root, logicalCollection(root, indexName, config));
  if (!target) return { deleted: 0 };
  try {
    const documents = target.fetchSync({ ids: unique, outputFields: ["metadata_json"], includeVector: false });
    const scoped = unique.filter((key) => documents[key]
      && String(metadataOf(documents[key]).embedding_space ?? "") === (config.embeddingSpace ?? ""));
    if (scoped.length) assertStatuses(target.deleteSync(scoped));
    return { deleted: scoped.length };
  } finally {
    releaseCollection(target);
  }
}

export async function getLocalVectors(
  indexName: string,
  keys: string[],
  config: LocalVectorConfig,
  options: { returnData?: boolean } = {},
) {
  const name = logicalCollection(localVectorDatabasePath(config), indexName, config);
  const result = await getLocalVectorCollection(name, keys, config, options);
  return { vectors: result.vectors.filter((row) => String(row.metadata?.embedding_space ?? "") === (config.embeddingSpace ?? "")) };
}

/** Internal physical fetch; public callers validate identity and filter the space. */
async function getLocalVectorCollection(
  collectionName: string,
  keys: string[],
  config: LocalVectorConfig,
  options: { returnData?: boolean } = {},
) {
  const unique = [...new Set(keys.map(String).filter(Boolean))];
  if (!unique.length) return { vectors: [] as VectorRecord[] };
  const target = openCollection(localVectorDatabasePath(config), collectionName);
  if (!target) return { vectors: [] as VectorRecord[] };
  try {
    const documents = target.fetchSync({
      ids: unique,
      outputFields: options.returnData !== false ? ["metadata_json", "vector_f32"] : ["metadata_json"],
      includeVector: options.returnData !== false,
    });
    return {
      vectors: unique.flatMap((key) => documents[key] ? [record(documents[key], options.returnData !== false)] : []),
    };
  } finally {
    releaseCollection(target);
  }
}

export async function listLocalVectors(
  indexName: string,
  config: LocalVectorConfig,
  options: { filter?: unknown; maxItems?: number; returnData?: boolean } = {},
) {
  const root = localVectorDatabasePath(config);
  const target = openCollection(root, logicalCollection(root, indexName, config));
  if (!target) return [];
  const rows: VectorRecord[] = [];
  try {
    const iterator = target.iterDocsSync({
      outputFields: options.returnData ? ["metadata_json", "vector_f32"] : ["metadata_json"],
      includeVector: Boolean(options.returnData),
    });
    for (const doc of iterator) {
      const item = record(doc, Boolean(options.returnData));
      if (String(item.metadata?.embedding_space ?? "") !== (config.embeddingSpace ?? "")) continue;
      if (!matches(item.metadata || {}, options.filter)) continue;
      rows.push(item);
      if (options.maxItems && rows.length >= options.maxItems) break;
    }
    return rows;
  } finally {
    releaseCollection(target);
  }
}

export async function queryLocalVectors(
  indexName: string,
  vector: number[],
  config: LocalVectorConfig,
  options: { filter?: unknown; limit?: number } = {},
) {
  const root = localVectorDatabasePath(config);
  const target = openCollection(root, logicalCollection(root, indexName, config));
  if (!target) return { vectors: [] as VectorRecord[] };
  const requested = Math.min(500, Math.max(1, Number(options.limit || 60)));
  const filter = zvecFilter(options.filter);
  const candidates = filter.fullySupported ? requested : Math.min(5000, Math.max(requested * 20, 500));
  try {
    const rows = target.querySync({
      fieldName: VECTOR_FIELD,
      vector,
      topk: candidates,
      filter: `embedding_space = '${escapeFilterString(config.embeddingSpace ?? "")}'${filter.expression ? ` AND (${filter.expression})` : ""}`,
      includeVector: false,
      outputFields: ["metadata_json"],
    });
    return {
      vectors: rows.map((item) => record(item, false))
        .filter((item) => String(item.metadata?.embedding_space ?? "") === (config.embeddingSpace ?? ""))
        .filter((item) => matches(item.metadata || {}, options.filter))
        .slice(0, requested),
    };
  } finally {
    releaseCollection(target);
  }
}

export async function localVectorStatus(config: LocalVectorConfig) {
  const root = localVectorDatabasePath(config);
  fs.mkdirSync(root, { recursive: true });
  const manifest = readManifest(root);
  const tables = Object.keys(manifest.collections).sort();
  const collections = [];
  for (const name of tables) {
    if (manifest.version === 1) {
      // Status must not open a legacy native collection or imply an empty library.
      collections.push({ name, dimension: manifest.collections[name]?.dimension || 0, count: null });
      continue;
    }
    const target = openCollection(root, name);
    if (!target) continue;
    try {
      collections.push({ name, dimension: manifest.collections[name]?.dimension || 0, count: target.stats.docCount });
    } finally {
      releaseCollection(target);
    }
  }
  return {
    provider: "zvec",
    provider_label: "本地 zvec",
    path: root,
    schema_version: INDEX_SCHEMA_VERSION,
    storage_format_version: manifest.version,
    rebuild_required: manifest.version === 1 && tables.length > 0,
    tables,
    collections,
  };
}
