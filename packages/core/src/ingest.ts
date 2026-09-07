// @ts-nocheck -- request payload compatibility is preserved while capture contracts stabilize.
import { withOperation, throwIfAborted } from "@indexed/clients/operation";
import { runOwnedCommand } from "./owned-command.js";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  embedImage,
  embedTranscript,
  embedVideo,
  embedVideoFrames,
  embeddingCorpusInputVersion,
  getVectors as getCloudVectors,
  putVectors as putCloudVectors,
  uuid5Url,
} from "@indexed/clients/direct-cloud";
import { getLocalVectors, putLocalVectors } from "@indexed/clients/local-vectors";
import { repairWebmBase64Duration } from "@indexed/clients/webm-duration";
import { activeProfile, directConfig, indexedHomePath, loadConfig } from "@indexed/config";
import { PRODUCT_DEFAULTS, SEGMENT_SCHEMAS } from "@indexed/contracts";

function context(config = loadConfig()) {
  const profile = activeProfile(config);
  return { profile, direct: directConfig(profile) };
}

function getVectors(indexName, keys, direct, options = {}) {
  return direct.storageProvider === "local"
    ? getLocalVectors(indexName, keys, direct, options)
    : getCloudVectors(indexName, keys, direct, options);
}

function putVectors(indexName, vectors, direct, options = {}) {
  throwIfAborted(options.signal);
  return direct.storageProvider === "local"
    ? putLocalVectors(indexName, vectors, direct)
    : putCloudVectors(indexName, vectors, direct, options);
}

function identity(kind, body, direct) {
  const site = String(body.source_site || "youtube");
  const schema = String(body.segment_schema || kind);
  const segment = Number(body.segment_index ?? 0);
  return `${site}:${kind}:${body.video_id}:${schema}:${segment}:${direct.embeddingSpace}`;
}

function commonMetadata(body, direct, embeddingInputVersion) {
  return {
    source_site: String(body.source_site || "youtube"),
    video_id: String(body.video_id || ""),
    title: String(body.title || ""),
    channel_id: String(body.channel_id || ""),
    channel_name: String(body.channel_name || ""),
    source_url: String(body.source_url || ""),
    thumbnail_url: String(body.thumbnail_url || ""),
    segment_index: Number(body.segment_index ?? 0),
    segment_interval: Number(body.segment_interval || PRODUCT_DEFAULTS.capture.segmentSeconds),
    embedding_model: direct.embeddingModel,
    embedding_space: direct.embeddingSpace,
    embedding_input_version: embeddingInputVersion,
    indexed_at_ms: Date.now(),
  };
}

async function alreadyStored(indexName, key, direct, options) {
  throwIfAborted(options.signal);
  const page = await getVectors(indexName, [key], direct, { ...options, returnData: false, returnMetadata: true });
  return Boolean(page.vectors?.length);
}

function ffmpegBinary() {
  const configured = String(process.env.INDEXED_FFMPEG_BINARY || "").trim();
  if (configured) return configured;
  const home = indexedHomePath();
  const bundled = home ? path.join(home, "runtime", "ffmpeg", "ffmpeg") : "";
  return bundled && fs.existsSync(bundled) ? bundled : "ffmpeg";
}

async function transcodeWebmForApple(videoBase64, options) {
  throwIfAborted(options.signal);
  const home = indexedHomePath();
  const root = String(process.env.INDEXED_TEMP_DIR || "").trim()
    || (home ? path.join(home, "tmp") : os.tmpdir());
  fs.mkdirSync(root, { recursive: true, mode: 0o700 });
  const directory = fs.mkdtempSync(path.join(root, "webm-"));
  const source = path.join(directory, "capture.webm");
  const output = path.join(directory, "capture.mp4");
  fs.writeFileSync(source, Buffer.from(videoBase64, "base64"), { mode: 0o600 });
  const common = [
    "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", source,
    "-map", "0:v:0", "-an", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
  ];
  try {
    const binary = ffmpegBinary();
    try {
      await runOwnedCommand(binary, [...common, "-c:v", "h264_videotoolbox", "-b:v", "1500k", output], options);
    } catch (hardwareError) {
      throwIfAborted(options.signal);
      fs.rmSync(output, { force: true });
      try {
        await runOwnedCommand(binary, [...common, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", output], options);
      } catch (softwareError) {
        throwIfAborted(options.signal);
        throw new Error(`浏览器 WebM 转换失败：${String(softwareError.message || hardwareError.message || softwareError)}`);
      }
    }
    return fs.readFileSync(output).toString("base64");
  } catch (error) {
    if (error?.code === "ENOENT") {
      throw new Error("Apple 原生后端需要 ffmpeg 转换浏览器 WebM；请设置 INDEXED_FFMPEG_BINARY");
    }
    throw error;
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

export async function prepareVideoEmbeddingInput(body, options = {}) {
  throwIfAborted(options.signal);
  if (Array.isArray(body.video_frames) && body.video_frames.length) {
    return { mimeType: "video/frames", videoBase64: "", videoFrames: body.video_frames };
  }
  const filename = String(body.video_filename || "").toLowerCase();
  const mimeType = filename.endsWith(".mp4") || filename.endsWith(".m4v")
    ? "video/mp4"
    : "video/webm";
  let videoBase64 = String(body.video_base64 || "");
  if (mimeType !== "video/webm") return { mimeType, videoBase64 };

  const capturedSeconds = Number(body.captured_seconds || 0);
  const segmentSeconds = Number(body.end_time || 0) - Number(body.start_time || 0);
  const expectedMilliseconds = Math.max(0, capturedSeconds > 0 ? capturedSeconds : segmentSeconds) * 1000;
  const repaired = await repairWebmBase64Duration(videoBase64, expectedMilliseconds);
  const tolerance = Math.max(1, expectedMilliseconds * 0.001);
  if (repaired.durationMilliseconds === null
    || (expectedMilliseconds > 0
      && Math.abs(repaired.durationMilliseconds - expectedMilliseconds) > tolerance)) {
    throw new Error("WebM 分片缺少有效 duration，无法安全提交给视频向量模型");
  }
  videoBase64 = repaired.videoBase64;
  if (options.transcodeWebm) {
    videoBase64 = await transcodeWebmForApple(videoBase64, options);
    return { mimeType: "video/mp4", videoBase64 };
  }
  return { mimeType, videoBase64 };
}

export async function ingestVideo(body, config = loadConfig(), options = {}) {
  return withOperation(options, scope => ingestVideoWithin(body, config, { ...options, signal: scope.signal }));
}
async function ingestVideoWithin(body, config, options) {
  const { direct } = context(config);
  const key = await uuid5Url(identity("video", body, direct));
  if (await alreadyStored(direct.ossVisualIndex, key, direct, options)) return { ok: true, cached: true, record_id: key };
  const { mimeType, videoBase64, videoFrames } = await prepareVideoEmbeddingInput(body, {
    signal: options.signal,
    transcodeWebm: direct.embeddingProvider === "apple-native",
  });
  const vector = videoFrames
    ? await embedVideoFrames(videoFrames, direct, options)
    : await embedVideo(videoBase64, mimeType, direct, options);
  const preview = body.preview_base64 ? `data:image/jpeg;base64,${String(body.preview_base64)}` : String(body.thumbnail_url || "");
  const metadata = {
    ...commonMetadata(body, direct, embeddingCorpusInputVersion(direct, "video")),
    record_type: `${String(body.source_site || "youtube")}_video`,
    start_time: Number(body.start_time || 0),
    end_time: Number(body.end_time || 0),
    timestamp: Number(body.start_time || 0),
    duration: Number(body.duration || 0),
    captured_seconds: Number(Number(body.captured_seconds || 0).toFixed(3)),
    frame_count: Math.max(0, Number(body.frame_count || videoFrames?.length || 0)),
    capture_fps: Math.max(0, Number(body.capture_fps || 0)),
    capture_width: Math.max(0, Number(body.width || 0)),
    capture_height: Math.max(0, Number(body.height || 0)),
    segment_schema: String(body.segment_schema || (videoFrames ? SEGMENT_SCHEMAS.videoFrames : SEGMENT_SCHEMAS.video)),
    embedding_basis: videoFrames ? "native_ordered_frames_2fps" : "native_video_10s",
    video_transport: videoFrames ? "ordered_frames" : mimeType,
    preview_data_uri: preview,
  };
  await putVectors(direct.ossVisualIndex, [{ key, data: { float32: vector }, metadata }], direct, options);
  return { ok: true, cached: false, record_id: key };
}

export async function ingestVisual(body, config = loadConfig(), options = {}) {
  return withOperation(options, scope => ingestVisualWithin(body, config, { ...options, signal: scope.signal }));
}
async function ingestVisualWithin(body, config, options) {
  const { direct } = context(config);
  const key = await uuid5Url(identity("visual", body, direct));
  if (await alreadyStored(direct.ossVisualIndex, key, direct, options)) return { ok: true, cached: true, record_id: key };
  const mimeType = String(body.image_filename || "").endsWith(".png") ? "image/png" : "image/jpeg";
  const vector = await embedImage(String(body.image_base64 || ""), mimeType, "", direct, options);
  const metadata = {
    ...commonMetadata(body, direct, embeddingCorpusInputVersion(direct, "image")),
    record_type: `${String(body.source_site || "youtube")}_visual`,
    timestamp: Number(body.timestamp || 0),
    duration: Number(body.duration || 0),
    segment_schema: String(body.segment_schema || SEGMENT_SCHEMAS.visual),
    preview_data_uri: `data:${mimeType};base64,${String(body.image_base64 || "")}`,
  };
  await putVectors(direct.ossVisualIndex, [{ key, data: { float32: vector }, metadata }], direct, options);
  return { ok: true, cached: false, record_id: key };
}

export async function ingestTranscript(body, config = loadConfig(), options = {}) {
  return withOperation(options, scope => ingestTranscriptWithin(body, config, { ...options, signal: scope.signal }));
}
async function ingestTranscriptWithin(body, config, options) {
  const text = String(body.text || "").replace(/\s+/g, " ").trim();
  if (!text) return { ok: true, skipped: true };
  const { direct } = context(config);
  const key = await uuid5Url(identity("transcript", body, direct));
  if (await alreadyStored(direct.ossTranscriptIndex, key, direct, options)) return { ok: true, cached: true, record_id: key };
  const vector = await embedTranscript(text, direct, options);
  const metadata = {
    ...commonMetadata(body, direct, embeddingCorpusInputVersion(direct, "text")),
    record_type: `${String(body.source_site || "youtube")}_transcript`,
    start_time: Number(body.start_time || 0),
    end_time: Number(body.end_time || 0),
    segment_schema: String(body.segment_schema || SEGMENT_SCHEMAS.transcript),
    transcript_text: text,
  };
  await putVectors(direct.ossTranscriptIndex, [{ key, data: { float32: vector }, metadata }], direct, options);
  return { ok: true, cached: false, record_id: key };
}

export async function ingest(kind, body, config = loadConfig(), options = {}) {
  if (kind === "video") return ingestVideo(body, config, options);
  if (kind === "visual") return ingestVisual(body, config, options);
  if (kind === "transcript") return ingestTranscript(body, config, options);
  throw new Error(`不支持的分片类型：${kind}`);
}
