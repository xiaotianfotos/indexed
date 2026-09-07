import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { putLocalVectors } from "@indexed/clients/local-vectors";
import { librariesStatus, listAssets, pruneMissingAssets, readIngestPerformanceHistory, scanLibrary, searchAllAssets, searchAssets } from "@indexed/core";

const SPACE = "test-asset-space-v1";

function assetConfig(directory: string, library: string, profile: Record<string, unknown> = {}) {
  return {
    version: 1,
    activeProfile: "default",
    library: {
      roots: [library], rootKinds: {}, autoScan: false, scanIntervalSeconds: 900,
      maxAssetsPerScan: 400, maxFilesPerLibrary: 100_000,
      kinds: ["video", "image", "document"],
    },
    profiles: {
      default: {
        label: "Default",
        spaceId: SPACE,
        embedding: { baseUrl: "http://embedding.test", model: "test-model", dimension: 2, inputStyle: "auto" },
        reranker: { enabled: false, baseUrl: "", model: "", candidates: 20 },
        storage: { provider: "local", path: directory, assetIndex: "library-assets", documentIndex: "local-documents" },
        ...profile,
      },
    },
  };
}

/**
 * The embedding service answers with one fixed unit vector, so every score a case
 * sees is decided by the row vectors it wrote - which is exactly what the merge
 * rules have to be tested against.
 */
async function startEmbeddingStub(context: { after: (fn: () => unknown) => void }, empty = false) {
  const stub = http.createServer((_request, response) => {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify(empty ? { data: [] } : { data: [{ embedding: [1, 0] }] }));
  });
  await new Promise<void>((resolve) => stub.listen(0, "127.0.0.1", resolve));
  context.after(() => stub.close());
  const address = stub.address();
  return `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
}

/** Unit vectors whose cosine similarity to [1, 0] is the first component. */
const NEAR = (score: number) => [score, Math.sqrt(Math.max(0, 1 - score * score))];

test("dashboard search combines local files with extension web-link assets", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-all-assets-"));
  const library = path.join(directory, "library");
  fs.mkdirSync(library, { recursive: true });
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const baseUrl = await startEmbeddingStub(context);
  const localPath = path.join(library, "local.mp4");
  fs.writeFileSync(localPath, "video");
  const config = assetConfig(directory, library, {
    embedding: { baseUrl, model: "test-model", dimension: 2, inputStyle: "auto" },
  });
  const direct = { localStorePath: directory, embeddingSpace: SPACE };
  await putLocalVectors("library-assets", [{
    key: "local",
    data: { float32: NEAR(0.8) },
    metadata: {
      record_type: "asset", embedding_space: SPACE, embedding_model: "test-model",
      library_root: library, asset_path: localPath, asset_name: "local.mp4", file_kind: "video",
    },
  }], direct);
  await putLocalVectors("video-visual", [{
    key: "plugin",
    data: { float32: NEAR(0.95) },
    metadata: {
      embedding_space: SPACE, embedding_model: "test-model", source_site: "youtube", video_id: "abc123",
      title: "插件视频", channel_name: "演示频道", source_url: "https://www.youtube.com/watch?v=abc123",
      start_time: 42, end_time: 52,
    },
  }], direct);

  const result = await searchAllAssets("演示", { kind: "video", limit: 10, config });
  assert.equal(result.sources.local, 1);
  assert.equal(result.sources.web, 1);
  assert.deepEqual(result.hits.map((hit) => hit.source), ["web", "local"]);
  assert.match(result.hits[0]?.openUrl || "", /^https:\/\/www\.youtube\.com\/watch\?v=abc123&t=42s$/);
  assert.equal(result.hits[1]?.path, localPath);
});

test("a library tracks only its selected file kinds and removes excluded rows on rescan", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-kinds-"));
  const library = path.join(directory, "library");
  fs.mkdirSync(library, { recursive: true });
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const baseUrl = await startEmbeddingStub(context);
  const image = path.join(library, "poster.png");
  const note = path.join(library, "notes.md");
  fs.writeFileSync(image, "image");
  fs.writeFileSync(note, "document");
  const config = assetConfig(directory, library, {
    embedding: { baseUrl, model: "test-model", dimension: 2, inputStyle: "auto" },
  });
  config.library.rootKinds = { [library]: ["image"] };
  await putLocalVectors("library-assets", [{
    key: "old-document",
    data: { float32: [1, 0] },
    metadata: {
      record_type: "asset", embedding_space: SPACE, embedding_model: "test-model", library_root: library,
      asset_path: note, asset_name: "notes.md", file_kind: "document", extension: ".md",
      indexed_at_ms: 1_700_000_000_000,
    },
  }], { localStorePath: directory, embeddingSpace: SPACE });

  const historyPath = path.join(directory, "state", "history.json");
  const scanned = await scanLibrary(library, { config, limit: 10, recordPerformance: true, performanceHistoryPath: historyPath });
  assert.equal(scanned.discovered, 1);
  assert.equal(scanned.deleted, 1);
  assert.deepEqual(scanned.indexedByKind, { video: 0, image: 1, document: 0 });
  assert.equal(readIngestPerformanceHistory(historyPath).samples[0]?.indexedByKind.image, 1);
  assert.deepEqual((await librariesStatus(config)).libraries[0]?.kinds, ["image"]);
  assert.deepEqual((await listAssets({ limit: 10, includeMissing: true, config })).assets.map((asset) => asset.name), ["poster.png"]);
});

test("dry-run scans compute vectors without touching existing storage and sign identical inputs", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-comparison-"));
  const library = path.join(directory, "library");
  fs.mkdirSync(library);
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  fs.writeFileSync(path.join(library, "one.md"), "A comparison document");
  const baseUrl = await startEmbeddingStub(context);
  const config = assetConfig(directory, library, { embedding: { baseUrl, model: "test-model", dimension: 2, inputStyle: "auto" } });
  assert.equal((await scanLibrary(library, { config })).indexed, 1);
  assert.equal((await scanLibrary(library, { config })).skipped, 1);
  let key = "";
  const scan = () => scanLibrary(library, { config, dryRun: true, benchmark: true, onProgress: (job) => {
    if (job.benchmarkKey) key = job.benchmarkKey;
    if (job.storageTimings) assert.equal(job.storageTimings.opens, 0);
  } });
  assert.equal((await scan()).indexed, 1);
  const first = key;
  assert.match(first, /^[a-f0-9]{64}$/);
  assert.equal((await scan()).indexed, 1);
  assert.equal(key, first);
  Object.assign(config.profiles.default, { video: { chunkSeconds: 30, maxChunkSeconds: 60, fps: 2, width: 640, maxSegments: 240 } });
  await scan();
  assert.notEqual(key, first);
  const emptyStore = path.join(directory, "never-created-db");
  config.profiles.default.storage.path = emptyStore;
  const result = await scan();
  assert.equal(result.indexed, 1);
  assert.equal(fs.existsSync(emptyStore), false);
});

test("scan progress counts real files and keeps the next-run backlog visible", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-progress-"));
  const library = path.join(directory, "library");
  fs.mkdirSync(library, { recursive: true });
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  for (const name of ["one.md", "two.md", "three.md"]) {
    fs.writeFileSync(path.join(library, name), `document ${name}`);
  }
  const baseUrl = await startEmbeddingStub(context);
  const config = assetConfig(directory, library, {
    embedding: { baseUrl, model: "test-model", dimension: 2, inputStyle: "auto" },
  });
  config.library = {
    ...config.library,
    maxAssetsPerScan: 2,
    maxFilesPerLibrary: 100,
    rootKinds: { [library]: ["document"] },
  };
  const progress: Array<Record<string, unknown>> = [];

  const result = await scanLibrary(library, {
    config,
    limit: 2,
    background: true,
    onProgress: (job) => progress.push({
      done: job.done,
      total: job.total,
      discovered: job.discovered,
      remaining: job.remaining,
      processingMode: job.processingMode,
      executionMode: job.executionMode,
      concurrency: job.concurrency,
      currentUnitsDone: job.currentUnitsDone,
      currentUnitsTotal: job.currentUnitsTotal,
      currentUnitLabel: job.currentUnitLabel,
      currentCompletedUnitMs: job.currentCompletedUnitMs,
      lastFileElapsedMs: job.lastFileElapsedMs,
    }),
  });

  assert.equal(result.discovered, 3);
  assert.equal(result.indexed, 2);
  assert.equal(result.remaining, 1);
  assert.ok(progress.some((job) => job.done === 0 && job.total === 2 && job.remaining === 3));
  assert.ok(progress.some((job) => job.done === 1 && job.remaining === 2));
  assert.ok(progress.some((job) => job.done === 2 && job.remaining === 1));
  assert.ok(progress.every((job) => job.processingMode === "background" && job.concurrency === 1));
  assert.ok(progress.every((job) => job.executionMode === "remote"));
  const finished = progress.at(-1);
  assert.equal(finished?.currentUnitsDone, 1);
  assert.equal(finished?.currentUnitsTotal, 1);
  assert.equal(finished?.currentUnitLabel, "文本块");
  assert.ok(Number(finished?.currentCompletedUnitMs) >= 1);
  assert.ok(Number(finished?.lastFileElapsedMs) >= 1);
});

/**
 * The dashboard reads the asset table straight back, so these cases pin the key
 * and metadata contract the scanner writes: record type, semantic space and the
 * distinct-asset counting a person sees on a card.
 */
test("asset listing keeps record types and embedding spaces apart", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-list-"));
  const library = path.join(directory, "library");
  fs.mkdirSync(library, { recursive: true });
  // Asset rows stand for real files: results are filtered against the disk.
  fs.writeFileSync(path.join(library, "clip.mp4"), "clip");
  fs.writeFileSync(path.join(library, "note.md"), "hello");
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));

  const row = (key: string, metadata: Record<string, unknown>) => ({
    key,
    data: { float32: [1, 0] },
    metadata: {
      record_type: "asset",
      embedding_space: SPACE,
      embedding_model: "test-model",
      indexed_at_ms: 1_700_000_000_000,
      library_root: library,
      ...metadata,
    },
  });
  const clip = (index: number) => ({
    asset_path: path.join(library, "clip.mp4"),
    asset_name: "clip.mp4",
    file_kind: "video",
    extension: ".mp4",
    segment_index: index,
    start_time: index * 30,
    end_time: index * 30 + 30,
    duration: 30,
    embedding_basis: "native_video_chunk_v1",
  });
  await putLocalVectors("library-assets", [
    row("clip-0", clip(0)),
    row("clip-1", clip(1)),
    row("note-1", {
      asset_path: path.join(library, "note.md"),
      asset_name: "note.md",
      file_kind: "document",
      extension: ".md",
      text_preview: "hello",
      embedding_basis: "document_text_v1",
    }),
    // Same table and space, but not an asset record: legacy file rows must never
    // surface as assets.
    row("legacy-1", { record_type: "file", asset_path: "/elsewhere/legacy.mp4", file_kind: "video" }),
  ], { localStorePath: directory, embeddingSpace: SPACE });
  // The same index name under another space lives in another table and stays there.
  await putLocalVectors("library-assets", [
    row("foreign-space", { embedding_space: "other-embedding-space-v1", asset_path: path.join(library, "foreign.png"), file_kind: "image" }),
  ], { localStorePath: directory, embeddingSpace: "other-embedding-space-v1" });

  const config = assetConfig(directory, library);
  const listed = await listAssets({ limit: 50, config });
  assert.equal(listed.count, 3);
  assert.equal(listed.assets.every((asset) => asset.legacy === false), true);

  const videos = await listAssets({ limit: 50, kind: "video", config });
  assert.deepEqual(videos.assets.map((asset) => asset.segmentIndex).sort(), [0, 1]);
  assert.equal(videos.assets.map((asset) => asset.startSeconds).join(","), "0,30");
  assert.equal(
    videos.assets[0]?.previewUrl,
    `/api/assets/preview?asset=${encodeURIComponent(path.join(library, "clip.mp4"))}&at=0`,
  );

  const documents = await listAssets({ limit: 50, kind: "document", config });
  assert.equal(documents.count, 1);
  assert.equal(documents.assets[0]?.sidecarText, "hello");
  assert.equal(documents.assets[0]?.embeddingBasis, "document_text_v1");

  const otherLibrary = await listAssets({ limit: 50, library: path.join(directory, "elsewhere"), config });
  assert.equal(otherLibrary.count, 0);
  await assert.rejects(() => listAssets({ limit: 50, kind: "audio", config }), /不支持的素材类型/);

  const status = await librariesStatus(config);
  assert.equal(status.autoScan, false);
  assert.equal(status.scanIntervalSeconds, 900);
  assert.equal(status.maxAssetsPerScan, 400);
  assert.equal(status.libraries.length, 1);
  // Chunks of one clip are reported as one asset, which is what a card shows.
  assert.equal(status.libraries[0]?.assetCount, 2);
  assert.equal(status.libraries[0]?.path, library);
  assert.equal(status.libraries[0]?.state, "idle");
  assert.equal(status.libraries[0]?.lastScanAt, 1_700_000_000_000);
});

test("rows for files that no longer exist stay out of results until they are pruned", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-prune-"));
  const library = path.join(directory, "library");
  fs.mkdirSync(library, { recursive: true });
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const baseUrl = await startEmbeddingStub(context);
  fs.writeFileSync(path.join(library, "clip.mp4"), "clip");
  fs.writeFileSync(path.join(library, "brief.md"), "brief");
  // A render folder the person cleaned by hand: the vectors are still there.
  const gone = path.join(library, "cut-4k.mov");
  const config = assetConfig(directory, library, {
    embedding: { baseUrl, model: "test-model", dimension: 2, inputStyle: "auto" },
  });

  const row = (key: string, score: number, metadata: Record<string, unknown>) => ({
    key,
    data: { float32: NEAR(score) },
    metadata: {
      record_type: "asset",
      embedding_space: SPACE,
      embedding_model: "test-model",
      library_root: library,
      indexed_at_ms: 1_700_000_000_000,
      ...metadata,
    },
  });
  await putLocalVectors("library-assets", [
    row("clip-0", 0.98, {
      asset_path: path.join(library, "clip.mp4"),
      asset_name: "clip.mp4",
      file_kind: "video",
      extension: ".mp4",
      segment_index: 0,
      embedding_basis: "native_video_chunk_v1",
    }),
    row("brief-0", 0.9, {
      asset_path: path.join(library, "brief.md"),
      asset_name: "brief.md",
      file_kind: "document",
      extension: ".md",
      embedding_basis: "document_text_v1",
    }),
    // Two rows of one deleted file, and the best match on any query is among them.
    row("gone-0", 1, { asset_path: gone, asset_name: "cut-4k.mov", file_kind: "video", extension: ".mov", segment_index: 0 }),
    row("gone-1", 0.99, { asset_path: gone, asset_name: "cut-4k.mov", file_kind: "video", extension: ".mov", segment_index: 1 }),
  ], { localStorePath: directory, embeddingSpace: SPACE });

  const listed = await listAssets({ limit: 50, config });
  assert.deepEqual(listed.assets.map((asset) => asset.name).sort(), ["brief.md", "clip.mp4"]);
  assert.equal((await listAssets({ limit: 50, includeMissing: true, config })).count, 4);

  const searched = await searchAssets("成片", { limit: 10, config });
  assert.equal(searched.hiddenMissing, 2);
  assert.deepEqual(searched.hits.map((hit) => hit.id), ["clip-0", "brief-0"]);
  assert.equal(searched.hits.some((hit) => hit.path === gone), false);

  const preview = await pruneMissingAssets({ dryRun: true, config });
  assert.equal(preview.scanned, 4);
  assert.equal(preview.removedRows, 0);
  assert.deepEqual(preview.missingAssets.map((item) => [item.path, item.kind, item.rows]), [[gone, "video", 2]]);
  // A dry run changes nothing, not even the row a person can still ask for.
  assert.equal((await listAssets({ limit: 50, includeMissing: true, config })).count, 4);

  const otherRoot = await pruneMissingAssets({ library: path.join(directory, "elsewhere"), dryRun: true, config });
  assert.deepEqual([otherRoot.scanned, otherRoot.missingAssets.length], [0, 0]);

  const pruned = await pruneMissingAssets({ config });
  assert.deepEqual([pruned.scanned, pruned.removedRows], [4, 2]);
  assert.equal((await listAssets({ limit: 50, includeMissing: true, config })).count, 2);
  assert.equal((await searchAssets("成片", { limit: 10, config })).hiddenMissing, 0);
});

test("legacy rows follow the requested kind and never fill more than half a page", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-legacy-"));
  const library = path.join(directory, "library");
  const migrated = path.join(directory, "migrated");
  fs.mkdirSync(library, { recursive: true });
  fs.mkdirSync(migrated, { recursive: true });
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const baseUrl = await startEmbeddingStub(context);
  fs.writeFileSync(path.join(library, "a.md"), "a");
  fs.writeFileSync(path.join(library, "b.md"), "b");
  fs.writeFileSync(path.join(migrated, "legacy-note.txt"), "legacy");
  fs.writeFileSync(path.join(migrated, "legacy-clip.mp4"), "legacy");
  const embedding = { baseUrl, model: "test-model", dimension: 2, inputStyle: "auto" };
  const config = assetConfig(directory, library, { embedding });

  await putLocalVectors("library-assets", [
    { key: "live-a", data: { float32: NEAR(0.9) }, metadata: {
      record_type: "asset", embedding_space: SPACE, embedding_model: "test-model", library_root: library,
      indexed_at_ms: 1_700_000_000_000, asset_path: path.join(library, "a.md"), asset_name: "a.md",
      file_kind: "document", extension: ".md",
    } },
    { key: "live-b", data: { float32: NEAR(0.8) }, metadata: {
      record_type: "asset", embedding_space: SPACE, embedding_model: "test-model", library_root: library,
      indexed_at_ms: 1_700_000_000_000, asset_path: path.join(library, "b.md"), asset_name: "b.md",
      file_kind: "document", extension: ".md",
    } },
  ], { localStorePath: directory, embeddingSpace: SPACE });
  // The shape a qdrant import leaves behind: its own table, its own record type and
  // source paths that are not part of any library root.
  const legacy = (key: string, score: number, metadata: Record<string, unknown>) => ({
    key,
    data: { float32: NEAR(score) },
    metadata: {
      record_type: "document_chunk",
      embedding_space: SPACE,
      embedding_model: "test-model",
      migration_source: "qdrant-embedded",
      ...metadata,
    },
  });
  await putLocalVectors("local-documents", [
    legacy("legacy-1", 0.99, { modality: "document", source_path: path.join(migrated, "legacy-note.txt") }),
    legacy("legacy-2", 0.97, { modality: "document", source_path: path.join(migrated, "legacy-note.txt") }),
    legacy("legacy-3", 0.95, { modality: "document", source_path: path.join(migrated, "legacy-note.txt") }),
    legacy("legacy-clip", 0.93, { modality: "video", source_path: path.join(migrated, "legacy-clip.mp4") }),
  ], { localStorePath: directory, embeddingSpace: SPACE });

  // Every legacy row outscores every live row, so only the cap keeps the library visible.
  const documents = await searchAssets("笔记", { kind: "document", limit: 4, config });
  assert.deepEqual(documents.hits.map((hit) => hit.id), ["legacy-1", "legacy-2", "live-a", "live-b"]);
  assert.equal(documents.hits.filter((hit) => hit.legacy).length, 2);
  assert.equal(documents.hits.some((hit) => hit.id === "legacy-3"), false);

  const everything = await searchAssets("笔记", { limit: 8, config });
  assert.deepEqual(everything.hits.map((hit) => hit.id), [
    "legacy-1", "legacy-2", "legacy-3", "legacy-clip", "live-a", "live-b",
  ]);

  const live = await searchAssets("笔记", { kind: "document", limit: 4, includeLegacy: false, config });
  assert.deepEqual(live.hits.map((hit) => hit.id), ["live-a", "live-b"]);
  assert.equal(live.hits.some((hit) => hit.legacy), false);
});

test("an embedding service that answers nothing fails instead of reporting no matches", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-empty-"));
  const library = path.join(directory, "library");
  fs.mkdirSync(library, { recursive: true });
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const baseUrl = await startEmbeddingStub(context, true);
  const config = assetConfig(directory, library, {
    embedding: { baseUrl, model: "test-model", dimension: 2, inputStyle: "auto" },
  });
  await assert.rejects(
    () => searchAssets("成片", { kind: "video", limit: 10, config }),
    /Embedding.*(没有返回向量|响应中没有向量)/,
  );
});

test("a kind filter searches past the first page of candidates", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-kind-"));
  const library = path.join(directory, "library");
  fs.mkdirSync(library, { recursive: true });
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const baseUrl = await startEmbeddingStub(context);
  fs.writeFileSync(path.join(library, "poster.png"), "poster");
  fs.writeFileSync(path.join(library, "clip.mp4"), "clip");
  const config = assetConfig(directory, library, {
    embedding: { baseUrl, model: "test-model", dimension: 2, inputStyle: "auto" },
  });

  // Six hundred image rows sit between the query and the only video row. The old
  // hard 500-candidate ceiling returned nothing even though the video existed.
  const rows = [];
  for (let index = 0; index < 600; index += 1) {
    rows.push({
      key: `image-${index}`,
      data: { float32: NEAR(0.9995 - index * 0.001) },
      metadata: {
        record_type: "asset", embedding_space: SPACE, embedding_model: "test-model",
        library_root: library, indexed_at_ms: 1_700_000_000_000,
        asset_path: path.join(library, "poster.png"), asset_name: "poster.png",
        file_kind: "image", extension: ".png",
      },
    });
  }
  rows.push({
    key: "video-1",
    data: { float32: NEAR(0.9) },
    metadata: {
      record_type: "asset", embedding_space: SPACE, embedding_model: "test-model",
      library_root: library, indexed_at_ms: 1_700_000_000_000,
      asset_path: path.join(library, "clip.mp4"), asset_name: "clip.mp4",
      file_kind: "video", extension: ".mp4", segment_index: 0,
      embedding_basis: "native_video_chunk_v1",
    },
  });
  await putLocalVectors("library-assets", rows, { localStorePath: directory, embeddingSpace: SPACE });

  const videos = await searchAssets("画面", { kind: "video", limit: 2, config });
  assert.deepEqual(videos.hits.map((hit) => hit.id), ["video-1"]);
  assert.equal(videos.hits.every((hit) => hit.kind === "video"), true);
});

test("an asset name match can repair a weak content-vector result", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-name-"));
  const library = path.join(directory, "library");
  fs.mkdirSync(library, { recursive: true });
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const baseUrl = await startEmbeddingStub(context);
  const openshell = path.join(library, "openshell.mp4");
  const snake = path.join(library, "snakegame.mp4");
  fs.writeFileSync(openshell, "video");
  fs.writeFileSync(snake, "video");
  const config = assetConfig(directory, library, {
    embedding: { baseUrl, model: "test-model", dimension: 2, inputStyle: "auto" },
  });

  const metadata = (assetPath: string) => ({
    record_type: "asset", embedding_space: SPACE, embedding_model: "test-model",
    library_root: library, indexed_at_ms: 1_700_000_000_000,
    asset_path: assetPath, asset_name: path.basename(assetPath),
    file_kind: "video", extension: ".mp4", segment_index: 0,
    embedding_basis: "native_video_chunk_v1",
  });
  await putLocalVectors("library-assets", [
    { key: "snake", data: { float32: NEAR(0.48) }, metadata: metadata(snake) },
    { key: "openshell", data: { float32: NEAR(0.41) }, metadata: metadata(openshell) },
  ], { localStorePath: directory, embeddingSpace: SPACE });

  const result = await searchAssets("OpenShell 外观设置演示", { kind: "video", limit: 2, config });
  assert.deepEqual(result.hits.map((hit) => hit.name), ["openshell.mp4", "snakegame.mp4"]);
  assert.ok((result.hits[0]?.score || 0) > (result.hits[1]?.score || 0));
});

test("an explicit file-format request participates in hybrid ranking", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-format-"));
  const library = path.join(directory, "library");
  fs.mkdirSync(library, { recursive: true });
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const baseUrl = await startEmbeddingStub(context);
  const csv = path.join(library, "manual-fixes.csv");
  const json = path.join(library, "subtitles.json");
  const pdf = path.join(library, "production-notes.pdf");
  fs.writeFileSync(csv, "before,after");
  fs.writeFileSync(json, "{}");
  fs.writeFileSync(pdf, "%PDF-test");
  const config = assetConfig(directory, library, {
    embedding: { baseUrl, model: "test-model", dimension: 2, inputStyle: "auto" },
  });
  const metadata = (assetPath: string) => ({
    record_type: "asset", embedding_space: SPACE, embedding_model: "test-model",
    library_root: library, indexed_at_ms: 1_700_000_000_000,
    asset_path: assetPath, asset_name: path.basename(assetPath),
    file_kind: "document", extension: path.extname(assetPath), segment_index: 0,
    embedding_basis: "document_text_v1",
  });
  const distractors = Array.from({ length: 30 }, (_, index) => ({
    key: `json-${index}`,
    data: { float32: NEAR(0.48 - index * 0.002) },
    metadata: metadata(json),
  }));
  await putLocalVectors("library-assets", [
    ...distractors,
    // Outside the old six-row recall window for a two-result page.
    { key: "csv", data: { float32: NEAR(0.37) }, metadata: metadata(csv) },
    { key: "pdf", data: { float32: NEAR(0.36) }, metadata: metadata(pdf) },
  ], { localStorePath: directory, embeddingSpace: SPACE });

  const result = await searchAssets("人工修正字幕错误的 CSV 表格", { kind: "document", limit: 2, config });
  assert.deepEqual(result.hits.map((hit) => hit.name), ["manual-fixes.csv", "subtitles.json"]);
  const pdfResult = await searchAssets("PDF 文档里的制作笔记", { kind: "document", limit: 2, config });
  assert.equal(pdfResult.hits[0]?.name, "production-notes.pdf");
});
