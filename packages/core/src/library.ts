// @ts-nocheck -- behavior-preserving port; Core ports now provide the typed migration boundary.
import { withOperation, throwIfAborted } from "@indexed/clients/operation";
import {
  deleteVectors as deleteCloudVectors,
  embedImage,
  embedTranscript,
  embedTranscriptQuery,
  embedVisualQuery,
  getVectors as getCloudVectors,
  listVectors as listCloudVectors,
  metadataFilter,
  putVectors as putCloudVectors,
  queryVectors as queryCloudVectors,
  scoreFromDistance,
  testDirectConfig as testCloudConfig,
  testEmbeddingConfig,
} from "@indexed/clients/direct-cloud";
import {
  deleteLocalVectors,
  getLocalVectors,
  listLocalVectors,
  localVectorStatus,
  putLocalVectors,
  queryLocalVectors,
} from "@indexed/clients/local-vectors";
import { rerank, rerankerHealth } from "@indexed/clients/rerank";
import { activeProfile, directConfig, loadConfig } from "@indexed/config";
import { deleteVideoWithRepository, prepareVideoDeletionWithRepository } from "./video-deletion.js";

function context(config = loadConfig()) {
  const profile = activeProfile(config);
  return { config, profile, direct: directConfig(profile) };
}

function usesLocalStorage(direct) {
  return direct.storageProvider === "local";
}

function deleteVectors(indexName, keys, direct) {
  return usesLocalStorage(direct)
    ? deleteLocalVectors(indexName, keys, direct)
    : deleteCloudVectors(indexName, keys, direct);
}

function getVectors(indexName, keys, direct, options = {}) {
  return usesLocalStorage(direct)
    ? getLocalVectors(indexName, keys, direct, options)
    : getCloudVectors(indexName, keys, direct, options);
}

function listVectors(indexName, direct, options = {}) {
  return usesLocalStorage(direct)
    ? listLocalVectors(indexName, direct, options)
    : listCloudVectors(indexName, direct, options);
}

function putVectors(indexName, vectors, direct) {
  return usesLocalStorage(direct)
    ? putLocalVectors(indexName, vectors, direct)
    : putCloudVectors(indexName, vectors, direct);
}

function queryVectors(indexName, vector, direct, options = {}) {
  return usesLocalStorage(direct)
    ? queryLocalVectors(indexName, vector, direct, options)
    : queryCloudVectors(indexName, vector, direct, options);
}

async function testStorage(direct) {
  if (!usesLocalStorage(direct)) return testCloudConfig(direct);
  const [embedding, storage] = await Promise.all([
    testEmbeddingConfig(direct),
    localVectorStatus(direct),
  ]);
  return { ok: true, embedding, storage, sampleCount: 0 };
}

function profileFilter(direct, extra = {}) {
  // Model names are not sufficient provenance: dimensions can match while
  // semantic spaces differ, so every query stays inside one explicit space.
  return metadataFilter({
    embedding_model: direct.embeddingModel,
    embedding_space: direct.embeddingSpace,
    ...extra,
  });
}

function result(row, modality) {
  const metadata = row.metadata || {};
  return {
    id: String(row.key || row.id || ""),
    modality,
    score: scoreFromDistance(row.distance),
    distance: Number(row.distance || 0),
    videoId: String(metadata.video_id || ""),
    sourceSite: String(metadata.source_site || "youtube"),
    title: String(metadata.title || "Untitled"),
    channel: String(metadata.channel_name || ""),
    sourceUrl: String(metadata.source_url || ""),
    startTime: Number(metadata.start_time ?? metadata.timestamp ?? 0),
    endTime: Number(metadata.end_time ?? metadata.timestamp ?? 0),
    transcript: String(metadata.transcript_text || ""),
    preview: String(metadata.preview_data_uri || metadata.thumbnail_url || ""),
    embeddingModel: String(metadata.embedding_model || ""),
    embeddingSpace: String(metadata.embedding_space || ""),
  };
}

export async function status(config = loadConfig()) {
  const { profile, direct } = context(config);
  const tested = await testStorage(direct);
  const filter = profileFilter(direct);
  const [visual, transcript] = await Promise.all([
    listVectors(direct.ossVisualIndex, direct, { filter, maxItems: 1 }),
    listVectors(direct.ossTranscriptIndex, direct, { filter, maxItems: 1 }),
  ]);
  return {
    ok: true,
    activeProfile: profile.id,
    label: profile.label,
    embeddingProvider: direct.embeddingProvider,
    embeddingModel: direct.embeddingModel,
    embeddingDimension: direct.embeddingDimension,
    embeddingSpace: direct.embeddingSpace,
    storageProvider: direct.storageProvider,
    embedding: tested.embedding,
    storage: tested.storage,
    visualReachable: Array.isArray(visual),
    transcriptReachable: Array.isArray(transcript),
    reranker: await rerankerHealth(profile),
  };
}

export async function search(query, options = {}, config = loadConfig()) {
  return withOperation(options, scope => searchWithin(query, { ...options, signal: scope.signal }, config));
}
async function searchWithin(query, { limit = 30, videoId = "", signal } = {}, config = loadConfig()) {
  const text = String(query || "").trim();
  if (!text) throw new Error("搜索文字不能为空");
  const { profile, direct } = context(config);
  const filter = profileFilter(direct, videoId ? { video_id: videoId } : {});
  const [visualVector, transcriptVector] = await Promise.all([
    embedVisualQuery(text, direct, { signal }),
    embedTranscriptQuery(text, direct, { signal }),
  ]);
  const [visual, transcript] = await Promise.all([
    queryVectors(direct.ossVisualIndex, visualVector, direct, { limit, filter, signal }),
    queryVectors(direct.ossTranscriptIndex, transcriptVector, direct, { limit, filter }),
  ]);
  const visualItems = (visual.vectors || []).map((row) => result(row, "visual"));
  const transcriptItems = (transcript.vectors || []).map((row) => result(row, "transcript"));
  const [visualRanked, transcriptRanked] = await Promise.all([
    rerank(text, visualItems, profile, { signal }),
    rerank(text, transcriptItems, profile, { signal }),
  ]);
  return {
    query: text,
    profile: profile.id,
    embeddingModel: direct.embeddingModel,
    embeddingSpace: direct.embeddingSpace,
    reranked: visualRanked.applied || transcriptRanked.applied,
    warnings: [visualRanked.warning, transcriptRanked.warning].filter(Boolean),
    visual: visualRanked.items,
    transcript: transcriptRanked.items,
  };
}

export async function searchImage(imageBase64, options = {}, config = loadConfig()) {
  return withOperation(options, scope => searchImageWithin(imageBase64, { ...options, signal: scope.signal }, config));
}
async function searchImageWithin(imageBase64, { mimeType = "image/png", query = "", limit = 30, videoId = "", signal } = {}, config = loadConfig()) {
  const { profile, direct } = context(config);
  const filter = profileFilter(direct, videoId ? { video_id: videoId } : {});
  const vector = await embedImage(String(imageBase64 || ""), mimeType, query, direct, { signal });
  const page = await queryVectors(direct.ossVisualIndex, vector, direct, { limit, filter });
  const visualItems = (page.vectors || []).map((row) => result(row, "visual"));
  const ranked = await rerank(query || "视觉相似画面", visualItems, profile, { signal });
  return {
    query,
    profile: profile.id,
    embeddingModel: direct.embeddingModel,
    embeddingSpace: direct.embeddingSpace,
    reranked: ranked.applied,
    warnings: [ranked.warning].filter(Boolean),
    visual: ranked.items,
    transcript: [],
  };
}

function aggregateRows(visualRows, transcriptRows) {
  const videos = new Map();
  const add = (row, modality) => {
    const metadata = row.metadata || {};
    const site = String(metadata.source_site || "youtube");
    const videoId = String(metadata.video_id || "");
    if (!videoId) return;
    const key = `${site}:${videoId}`;
    const current = videos.get(key) || {
      key,
      sourceSite: site,
      videoId,
      title: String(metadata.title || "Untitled"),
      channel: String(metadata.channel_name || ""),
      sourceUrl: String(metadata.source_url || ""),
      thumbnail: String(metadata.preview_data_uri || metadata.thumbnail_url || ""),
      visualCount: 0,
      transcriptCount: 0,
      visualSegments: new Set(),
      transcriptSegments: new Set(),
      duration: 0,
      segmentInterval: 0,
      indexedAt: 0,
      embeddingModel: String(metadata.embedding_model || ""),
      embeddingSpace: String(metadata.embedding_space || ""),
    };
    const segmentIndex = Number(metadata.segment_index);
    if (modality === "visual") {
      current.visualCount += 1;
      if (Number.isInteger(segmentIndex) && segmentIndex >= 0) current.visualSegments.add(segmentIndex);
    } else {
      current.transcriptCount += 1;
      if (Number.isInteger(segmentIndex) && segmentIndex >= 0) current.transcriptSegments.add(segmentIndex);
    }
    current.duration = Math.max(current.duration, Number(metadata.duration || metadata.end_time || 0));
    current.segmentInterval = Math.max(current.segmentInterval, Number(metadata.segment_interval || 0));
    current.indexedAt = Math.max(current.indexedAt, Number(metadata.indexed_at_ms || 0));
    if (!current.thumbnail) current.thumbnail = String(metadata.preview_data_uri || metadata.thumbnail_url || "");
    videos.set(key, current);
  };
  visualRows.forEach((row) => add(row, "visual"));
  transcriptRows.forEach((row) => add(row, "transcript"));
  return [...videos.values()].map((video) => ({
    ...video,
    visualSegments: [...video.visualSegments].sort((left, right) => left - right),
    transcriptSegments: [...video.transcriptSegments].sort((left, right) => left - right),
  })).sort((a, b) => b.indexedAt - a.indexedAt);
}

export async function listVideos({ limit = 100 } = {}, config = loadConfig()) {
  const { profile, direct } = context(config);
  const filter = profileFilter(direct);
  const [visual, transcript] = await Promise.all([
    listVectors(direct.ossVisualIndex, direct, { filter }),
    listVectors(direct.ossTranscriptIndex, direct, { filter }),
  ]);
  const videos = aggregateRows(visual, transcript);
  return {
    profile: profile.id,
    embeddingModel: direct.embeddingModel,
    count: videos.length,
    videos: videos.slice(0, Number(limit || 100)),
  };
}

async function rowsForVideo(profile, direct, sourceSite, videoId, returnData = false) {
  const filter = profileFilter(direct, { video_id: videoId, source_site: sourceSite });
  const [visualRows, transcriptRows] = await Promise.all([
    listVectors(direct.ossVisualIndex, direct, { filter }),
    listVectors(direct.ossTranscriptIndex, direct, { filter }),
  ]);
  if (!returnData) return { visualRows, transcriptRows };
  const hydrate = async (indexName, rows) => {
    const output = [];
    for (let offset = 0; offset < rows.length; offset += 100) {
      const keys = rows.slice(offset, offset + 100).map((row) => row.key);
      const page = await getVectors(indexName, keys, direct, { returnData: true, returnMetadata: true });
      output.push(...(page.vectors || []));
    }
    return output;
  };
  return {
    visualRows: await hydrate(direct.ossVisualIndex, visualRows),
    transcriptRows: await hydrate(direct.ossTranscriptIndex, transcriptRows),
  };
}

async function inChunks(rows, callback) {
  for (let offset = 0; offset < rows.length; offset += 500) await callback(rows.slice(offset, offset + 500));
}

function videoDeletionContext(sourceSite, videoId, config) {
  const { profile, direct } = context(config);
  const target = { profileId: profile.id, storageProvider: direct.storageProvider,
    embeddingModel: direct.embeddingModel, embeddingSpace: direct.embeddingSpace, sourceSite, videoId,
    storage: { region: direct.ossRegion, accountId: direct.ossAccountId, bucket: direct.ossBucket,
      localPath: usesLocalStorage(direct) ? direct.localStorePath : "",
      visualIndex: direct.ossVisualIndex, transcriptIndex: direct.ossTranscriptIndex } };
  const repository = {
    listKeys: async () => {
      const { visualRows, transcriptRows } = await rowsForVideo(profile, direct, sourceSite, videoId);
      const scopedKeys = (rows) => rows.filter((row) => row.metadata?.source_site === sourceSite
        && row.metadata?.video_id === videoId && row.metadata?.embedding_space === direct.embeddingSpace
        && row.metadata?.embedding_model === direct.embeddingModel && typeof row.key === "string" && row.key.length).map((row) => row.key);
      return { visualKeys: scopedKeys(visualRows), transcriptKeys: scopedKeys(transcriptRows) };
    },
    deleteKeys: (indexName, keys) => deleteVectors(indexName, keys, direct),
  };
  return { target, repository };
}

export async function prepareVideoDeletion(sourceSite, videoId, config = loadConfig()) {
  const { target, repository } = videoDeletionContext(sourceSite, videoId, config);
  return prepareVideoDeletionWithRepository(target, repository);
}

export async function deleteVideo(sourceSite, videoId, config = loadConfig(), options = {}) {
  const { target, repository } = videoDeletionContext(sourceSite, videoId, config);
  return deleteVideoWithRepository(target, repository, options.confirmation);
}

export async function updateVideo(sourceSite, videoId, patch, config = loadConfig()) {
  const allowed = Object.fromEntries(
    Object.entries(patch || {}).filter(([key]) => ["title", "channel_name", "note", "tags"].includes(key)),
  );
  if (!Object.keys(allowed).length) throw new Error("没有可更新的字段");
  const { profile, direct } = context(config);
  const { visualRows, transcriptRows } = await rowsForVideo(profile, direct, sourceSite, videoId, true);
  const update = async (indexName, rows) => {
    await inChunks(rows, async (chunk) => {
      const vectors = chunk.map((row) => ({
        key: row.key,
        data: row.data,
        metadata: { ...(row.metadata || {}), ...allowed, updated_at_ms: Date.now() },
      }));
      await putVectors(indexName, vectors, direct);
    });
  };
  await update(direct.ossVisualIndex, visualRows);
  await update(direct.ossTranscriptIndex, transcriptRows);
  return { ok: true, sourceSite, videoId, visualUpdated: visualRows.length, transcriptUpdated: transcriptRows.length, patch: allowed };
}
