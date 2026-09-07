import { execFile, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);

/**
 * Share of the model context a single video chunk may occupy. Measured against
 * the WEMM embedding service: a 30~180s clip at 1~4fps always lands at ~11.9k~13.6k
 * prompt tokens, so half of a 32k context is a safe ceiling for one chunk.
 */
export const MAX_CONTEXT_SHARE = 0.5;

/** Measured on the same service: the server resamples frames to ~110 tokens each. */
export const TOKENS_PER_FRAME = 110;

/** Fallback context length when the embedding service does not advertise one. */
export const DEFAULT_MAX_MODEL_LEN = 32768;

export interface VideoChunkSettings {
  chunkSeconds: number;
  maxChunkSeconds: number;
  fps: number;
  width: number;
  maxSegments: number;
}

export interface ChunkSegment {
  index: number;
  startSeconds: number;
  endSeconds: number;
}

export interface ChunkPlan {
  segments: ChunkSegment[];
  effectiveChunkSeconds: number;
  frameBudget: number;
  maxChunkSeconds: number;
  warning: string;
}

export interface MediaInfo {
  duration: number;
}

export interface VideoFramePacket {
  image_base64: string;
  mime_type: "image/jpeg";
  timestamp: number;
}

/**
 * Frames one chunk may spend before it would overflow the context budget.
 * 32768 -> floor(32768 * 0.5 / 110) = 148 frames on the measured service.
 */
export function frameBudget(maxModelLen: number): number {
  const context = Number.isFinite(maxModelLen) && maxModelLen > 0 ? maxModelLen : DEFAULT_MAX_MODEL_LEN;
  return Math.max(1, Math.floor(context * MAX_CONTEXT_SHARE / TOKENS_PER_FRAME));
}

/**
 * Split a video into consecutive chunks that cover `[0, duration]` without gaps.
 *
 * Chunk length is the smaller of the configured chunk length and what the frame
 * budget allows; when the resulting segment count would exceed `maxSegments` the
 * chunk length grows instead of dropping content, and the plan carries a warning.
 */
export function planChunks(durationSeconds: number, video: VideoChunkSettings, maxModelLen: number): ChunkPlan {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    throw new Error(`视频时长无效：${String(durationSeconds)}`);
  }
  const fps = Number.isFinite(video.fps) && video.fps > 0 ? video.fps : 2;
  const configuredChunk = Number.isFinite(video.chunkSeconds) && video.chunkSeconds > 0 ? Math.floor(video.chunkSeconds) : 30;
  const configuredMax = Number.isFinite(video.maxChunkSeconds) && video.maxChunkSeconds > 0
    ? Math.floor(video.maxChunkSeconds)
    : Math.max(configuredChunk, 60);
  const maxSegments = Number.isFinite(video.maxSegments) && video.maxSegments > 0 ? Math.floor(video.maxSegments) : 240;
  const budget = frameBudget(maxModelLen);
  const maxChunkSeconds = Math.max(1, Math.min(configuredMax, Math.floor(budget / fps)));
  let effectiveChunkSeconds = Math.max(1, Math.min(configuredChunk, maxChunkSeconds));

  const chunkCount = (): number => Math.ceil(durationSeconds / effectiveChunkSeconds);
  let warning = "";
  if (chunkCount() > maxSegments) {
    // Grow the chunk instead of truncating the tail: losing the last minutes of
    // raw footage is worse than the service dropping frames to fit the budget.
    effectiveChunkSeconds = Math.ceil(durationSeconds / maxSegments);
    warning = effectiveChunkSeconds > maxChunkSeconds
      ? `片段数超过上限 ${maxSegments}，单段时长放大到 ${effectiveChunkSeconds}s，超出 ${maxChunkSeconds}s 帧预算后服务端会丢帧`
      : `片段数超过上限 ${maxSegments}，单段时长放大到 ${effectiveChunkSeconds}s`;
  }

  const segments: ChunkSegment[] = [];
  let index = 0;
  for (let start = 0; start < durationSeconds; start += effectiveChunkSeconds) {
    segments.push({
      index,
      startSeconds: Number(start.toFixed(3)),
      endSeconds: Number(Math.min(durationSeconds, start + effectiveChunkSeconds).toFixed(3)),
    });
    index += 1;
  }
  return { segments, effectiveChunkSeconds, frameBudget: budget, maxChunkSeconds, warning };
}

/** Absolute paths that must never receive generated files (raw material is read-only). */
function isInside(parent: string, child: string): boolean {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

export function assertOutsideLibraryRoots(target: string, libraryRoots: string[] = []): void {
  const resolved = path.resolve(target);
  const conflict = (libraryRoots || [])
    .map((root) => String(root || "").trim())
    .filter(Boolean)
    .map((root) => path.resolve(root))
    .find((root) => isInside(root, resolved));
  if (conflict) throw new Error(`中间文件不能写入素材库目录：${resolved} 位于 ${conflict} 之内`);
}

function makeTemporaryDirectory(tmpDir: string, libraryRoots: string[]): string {
  assertOutsideLibraryRoots(tmpDir, libraryRoots);
  const target = path.resolve(tmpDir);
  fs.mkdirSync(target, { recursive: true });
  return target;
}

function safeFileBase(value: string): string {
  return path.basename(value, path.extname(value)).replace(/[^A-Za-z0-9._-]+/g, "_").slice(-64) || "chunk";
}

export function ffmpegAvailable(): boolean {
  try {
    const probe = spawnSync("ffmpeg", ["-version"], { encoding: "utf8", timeout: 15_000 });
    return !probe.error && probe.status === 0;
  } catch {
    return false;
  }
}

/** Read the container duration with ffprobe. */
export async function probeMedia(assetPath: string, options: { signal?: AbortSignal } = {}): Promise<MediaInfo> {
  const target = path.resolve(assetPath);
  const { stdout } = await run(
    "ffprobe",
    ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", target],
    { encoding: "utf8", timeout: 120_000, maxBuffer: 4 * 1024 * 1024, signal: options.signal },
  );
  const duration = Number.parseFloat(String(stdout || "").trim());
  if (!Number.isFinite(duration) || duration <= 0) throw new Error(`无法读取视频时长：${target}`);
  return { duration };
}

/**
 * Re-encode one chunk into `tmpDir`. Raw material is only ever read; the output
 * is a throwaway file the caller must remove via {@link withChunk}.
 */
export async function extractChunk(
  assetPath: string,
  startSeconds: number,
  endSeconds: number,
  video: VideoChunkSettings,
  tmpDir: string,
  libraryRoots: string[] = [],
  signal?: AbortSignal,
): Promise<string> {
  const source = path.resolve(assetPath);
  if (!fs.existsSync(source)) throw new Error(`素材不存在：${source}`);
  const directory = makeTemporaryDirectory(tmpDir, libraryRoots);
  const duration = Math.max(0.2, Number(endSeconds) - Number(startSeconds));
  const target = path.join(
    directory,
    `${safeFileBase(source)}-${Math.round(Number(startSeconds))}-${Math.round(Number(endSeconds))}.mp4`,
  );
  await run(
    "ffmpeg",
    [
      "-hide_banner", "-loglevel", "error", "-y",
      "-ss", String(Number(startSeconds)),
      "-i", source,
      "-t", String(Number(duration.toFixed(3))),
      "-vf", `scale=${Math.max(2, Math.floor(video.width))}:-2,fps=${video.fps}`,
      "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
      "-an", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
      target,
    ],
    { encoding: "utf8", timeout: 900_000, maxBuffer: 16 * 1024 * 1024, signal },
  );
  if (!fs.existsSync(target) || fs.statSync(target).size === 0) {
    throw new Error(`切片失败：${path.basename(source)} ${startSeconds}-${endSeconds}s`);
  }
  return target;
}

/** Extract one chunk, hand the file to `use`, and always delete it afterwards. */
export async function withChunk<T>(
  assetPath: string,
  startSeconds: number,
  endSeconds: number,
  video: VideoChunkSettings,
  tmpDir: string,
  use: (chunkPath: string) => Promise<T>,
  libraryRoots: string[] = [],
  signal?: AbortSignal,
): Promise<T> {
  const chunk = await extractChunk(assetPath, startSeconds, endSeconds, video, tmpDir, libraryRoots, signal);
  try {
    return await use(chunk);
  } finally {
    fs.rmSync(chunk, { force: true });
  }
}

/**
 * Decode one video segment into an ordered JPEG packet. Apple-native backends
 * consume the same 2 fps shape as the browser extension, avoiding a second
 * video-container decode inside the model service.
 */
export async function withFramePacket<T>(
  assetPath: string,
  startSeconds: number,
  endSeconds: number,
  video: VideoChunkSettings,
  tmpDir: string,
  use: (frames: VideoFramePacket[]) => Promise<T>,
  libraryRoots: string[] = [],
  signal?: AbortSignal,
): Promise<T> {
  const source = path.resolve(assetPath);
  if (!fs.existsSync(source)) throw new Error(`素材不存在：${source}`);
  const scratchRoot = makeTemporaryDirectory(tmpDir, libraryRoots);
  const directory = fs.mkdtempSync(path.join(scratchRoot, `${safeFileBase(source)}-frames-`));
  const duration = Math.max(0.2, Number(endSeconds) - Number(startSeconds));
  const pattern = path.join(directory, "frame-%06d.jpg");
  try {
    await run(
      "ffmpeg",
      [
        "-hide_banner", "-loglevel", "error", "-y",
        "-ss", String(Number(startSeconds)),
        "-i", source,
        "-t", String(Number(duration.toFixed(3))),
        "-vf", `fps=${video.fps},scale=${Math.max(2, Math.floor(video.width))}:-2`,
        "-q:v", "5", "-f", "image2", "-start_number", "0", pattern,
      ],
      { encoding: "utf8", timeout: 900_000, maxBuffer: 16 * 1024 * 1024, signal },
    );
    const names = fs.readdirSync(directory).filter((name) => /^frame-\d+\.jpg$/.test(name)).sort();
    if (!names.length) throw new Error(`抽帧失败：${path.basename(source)} ${startSeconds}-${endSeconds}s`);
    const fps = Number.isFinite(video.fps) && video.fps > 0 ? video.fps : 2;
    const frames = names.map((name, index): VideoFramePacket => ({
      image_base64: fs.readFileSync(path.join(directory, name)).toString("base64"),
      mime_type: "image/jpeg",
      timestamp: Number((index / fps).toFixed(3)),
    }));
    return await use(frames);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

function previewCacheKey(assetPath: string, atSeconds: number): string {
  const stat = fs.statSync(assetPath);
  return crypto
    .createHash("sha1")
    .update(`${assetPath}|${stat.size}|${Math.round(stat.mtimeMs)}|${Math.floor(Number(atSeconds))}`)
    .digest("hex");
}

/**
 * Extract a single JPEG frame into a persistent preview cache, keyed by
 * path + size + mtime + whole second so repeat requests cost no ffmpeg run.
 */
export async function extractFrame(
  assetPath: string,
  atSeconds: number,
  options: { previewDir: string; libraryRoots?: string[] },
): Promise<string> {
  const source = path.resolve(assetPath);
  if (!fs.existsSync(source)) throw new Error(`素材不存在：${source}`);
  const directory = makeTemporaryDirectory(options.previewDir, options.libraryRoots || []);
  const target = path.join(directory, `${previewCacheKey(source, atSeconds)}.jpg`);
  if (fs.existsSync(target) && fs.statSync(target).size > 0) return target;

  const temporary = `${target}.${process.pid}.tmp`;
  const at = Math.max(0, Math.floor(Number(atSeconds) || 0));
  const grab = async (seconds: number): Promise<boolean> => {
    fs.rmSync(temporary, { force: true });
    await run(
      "ffmpeg",
      [
        "-hide_banner", "-loglevel", "error", "-y",
        "-ss", String(seconds),
        "-i", source,
        "-frames:v", "1",
        "-vf", "scale=320:-2",
        "-q:v", "4",
        // The scratch name ends in .tmp, so the output muxer has to be named:
        // ffmpeg otherwise refuses an output format it cannot guess from the suffix.
        "-f", "image2", "-update", "1",
        temporary,
      ],
      { encoding: "utf8", timeout: 300_000, maxBuffer: 8 * 1024 * 1024 },
    ).catch(() => undefined);
    return fs.existsSync(temporary) && fs.statSync(temporary).size > 0;
  };
  try {
    // A requested second past the last decodable frame yields no output; fall
    // back to the head of the clip rather than failing the preview request.
    let produced = await grab(at);
    if (!produced && at > 0) produced = await grab(0);
    if (!produced) throw new Error(`抽帧失败：${path.basename(source)} ${at}s`);
    fs.renameSync(temporary, target);
  } finally {
    fs.rmSync(temporary, { force: true });
  }
  return target;
}
