import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import {
  assertOutsideLibraryRoots,
  DEFAULT_MAX_MODEL_LEN,
  extractChunk,
  extractFrame,
  ffmpegAvailable,
  frameBudget,
  MAX_CONTEXT_SHARE,
  planChunks,
  TOKENS_PER_FRAME,
  withFramePacket,
} from "../packages/core/src/asset-chunks.js";

const run = promisify(execFile);
const HAS_FFMPEG = ffmpegAvailable();

const VIDEO = { chunkSeconds: 30, maxChunkSeconds: 60, fps: 2, width: 1280, maxSegments: 240 };

test("sampling before scaling preserves the ordered JPEG packet and cleans scratch files", { skip: !HAS_FFMPEG }, async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-frame-order-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const source = path.join(directory, "clip.mp4");
  await run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=2", "-pix_fmt", "yuv420p", source]);
  const old = path.join(directory, "old");
  fs.mkdirSync(old);
  await run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-ss", "0", "-i", source, "-t", "2", "-vf", "scale=640:-2,fps=2", "-q:v", "5", "-f", "image2", "-start_number", "0", path.join(old, "frame-%06d.jpg")]);
  const scratch = path.join(directory, "scratch");
  await withFramePacket(source, 0, 2, { ...VIDEO, width: 640 }, scratch, async (frames) => {
    assert.equal(frames.length, 4);
    for (const [index, frame] of frames.entries()) {
      assert.equal(frame.timestamp, index / 2);
      assert.deepEqual(Buffer.from(frame.image_base64, "base64"), fs.readFileSync(path.join(old, `frame-${String(index).padStart(6, "0")}.jpg`)));
    }
  });
  assert.deepEqual(fs.readdirSync(scratch), []);
});

/** Video settings after the config normalizer has had its way with them. */
function videoSettings(overrides: Record<string, unknown> = {}) {
  const merged = { ...VIDEO, ...overrides };
  return {
    chunkSeconds: Number(merged.chunkSeconds),
    maxChunkSeconds: Number(merged.maxChunkSeconds),
    fps: Number(merged.fps),
    width: Number(merged.width),
    maxSegments: Number(merged.maxSegments),
  };
}

/** The invariant every emitted plan must hold, whatever narrowed the chunk. */
function assertCoversTimeline(plan: ReturnType<typeof planChunks>, durationSeconds: number, maxSegments = Number.POSITIVE_INFINITY) {
  assert.ok(plan.segments.length, "a plan must never emit zero segments");
  assert.ok(plan.segments.length <= maxSegments, "segment count must stay inside the cap");
  assert.equal(plan.segments[0]?.startSeconds, 0, "the first segment must start at zero");
  assert.equal(plan.segments.at(-1)?.endSeconds, durationSeconds, "the last segment must reach the end");
  plan.segments.forEach((segment, index) => {
    assert.equal(segment.index, index, "segment indexes must be sequential");
    assert.ok(segment.endSeconds > segment.startSeconds, "segments must not be empty");
    assert.ok(segment.startSeconds >= 0 && segment.endSeconds <= durationSeconds, "segments must stay in range");
    assert.ok(segment.endSeconds - segment.startSeconds <= plan.effectiveChunkSeconds + 1e-6, "segments must not outgrow the chunk");
    if (index > 0) assert.equal(segment.startSeconds, plan.segments[index - 1]?.endSeconds, "segments must not overlap or gap");
  });
}

test("frame budget keeps one chunk inside the advertised model context", () => {
  assert.equal(MAX_CONTEXT_SHARE, 0.5);
  assert.equal(TOKENS_PER_FRAME, 110);
  assert.equal(frameBudget(32768), 148);
  for (const maxModelLen of [220, 440, 8192, 32560, 32768, 131072]) {
    const frames = frameBudget(maxModelLen);
    assert.ok(frames >= 1);
    assert.ok(
      frames * TOKENS_PER_FRAME <= maxModelLen * MAX_CONTEXT_SHARE,
      `${maxModelLen} must not spend more than ${MAX_CONTEXT_SHARE} of its context`,
    );
    assert.ok(
      (frames + 1) * TOKENS_PER_FRAME > maxModelLen * MAX_CONTEXT_SHARE,
      `${maxModelLen} must use the widest frame count that still fits`,
    );
  }
});

test("frame budget rounds at the context boundary and falls back when the model is unknown", () => {
  // 32560 * 0.5 / 110 is exactly 148 frames; one token less drops to 147.
  assert.equal(frameBudget(32560), 148);
  assert.equal(frameBudget(32559), 147);
  // 440 * 0.5 / 110 is exactly 2 frames; one token less rounds down to 1.
  assert.equal(frameBudget(440), 2);
  assert.equal(frameBudget(439), 1);
  // A context too small for a single frame still plans one frame.
  assert.equal(frameBudget(220), 1);
  assert.equal(frameBudget(219), 1);
  assert.equal(frameBudget(1), 1);
  // An unusable advertised context length must never widen the budget to Infinity.
  for (const unusable of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(frameBudget(unusable), frameBudget(DEFAULT_MAX_MODEL_LEN));
    assert.ok(Number.isFinite(frameBudget(unusable)));
  }
  // Bigger contexts never plan fewer frames.
  const sizes = [8192, 16384, 32768, 65536, 131072];
  sizes.slice(1).forEach((size, index) => {
    assert.ok(frameBudget(size) >= frameBudget(sizes[index] as number));
  });
});

test("short clips are planned as a single segment that reaches the end", () => {
  const plan = planChunks(10, videoSettings(), 32768);
  assert.equal(plan.segments.length, 1);
  assert.deepEqual(plan.segments, [{ index: 0, startSeconds: 0, endSeconds: 10 }]);
  assert.equal(plan.effectiveChunkSeconds, VIDEO.chunkSeconds);
  assert.equal(plan.maxChunkSeconds, VIDEO.maxChunkSeconds);
  assert.equal(plan.frameBudget, frameBudget(32768));
  assert.equal(plan.warning, "");
  // A clip exactly one chunk long stays one segment; one second more splits.
  assert.equal(planChunks(30, videoSettings(), 32768).segments.length, 1);
  assert.equal(planChunks(31, videoSettings(), 32768).segments.length, 2);
  // Fractional durations survive the rounding to milliseconds.
  const fraction = planChunks(90.5, videoSettings(), 32768);
  assert.equal(fraction.segments.at(-1)?.endSeconds, 90.5);
  assertCoversTimeline(fraction, 90.5);
});

test("chunk length is narrowed by the frame budget before the configured ceiling", () => {
  // 148 frames at 10 fps leaves 14 seconds of context, below the configured 60.
  const tight = planChunks(90, videoSettings({ fps: 10 }), 32768);
  assert.equal(tight.frameBudget, frameBudget(32768));
  assert.equal(tight.maxChunkSeconds, 14);
  assert.equal(tight.effectiveChunkSeconds, 14);
  assert.ok(tight.effectiveChunkSeconds * 10 <= tight.frameBudget);
  assert.ok((tight.effectiveChunkSeconds + 1) * 10 > tight.frameBudget);
  assertCoversTimeline(tight, 90);

  // A smaller configured ceiling wins while the frame budget still fits.
  const narrow = planChunks(100, videoSettings({ chunkSeconds: 120, maxChunkSeconds: 45 }), 32768);
  assert.equal(narrow.maxChunkSeconds, 45);
  assert.equal(narrow.effectiveChunkSeconds, 45);
  assert.equal(narrow.warning, "");
  assertCoversTimeline(narrow, 100);

  // The configured chunk length is used whenever the frame budget allows it.
  const roomy = planChunks(100, videoSettings({ chunkSeconds: 20, maxChunkSeconds: 60 }), 32768);
  assert.equal(roomy.effectiveChunkSeconds, 20);
  assert.equal(roomy.maxChunkSeconds, 60);

  // A tiny context length must narrow the chunk even with generous settings.
  const tiny = planChunks(30, videoSettings({ chunkSeconds: 30, maxChunkSeconds: 60, fps: 1 }), 220);
  assert.equal(tiny.frameBudget, 1);
  assert.equal(tiny.maxChunkSeconds, 1);
  assert.equal(tiny.effectiveChunkSeconds, 1);
  assertCoversTimeline(tiny, 30);
});

test("segment cap grows the chunk instead of dropping the tail of the video", () => {
  // 1000s at 30s per chunk is 34 segments; the cap of 10 must not lose footage.
  const capped = planChunks(1000, videoSettings({ maxSegments: 10 }), 32768);
  assert.equal(capped.segments.length, 10);
  assert.equal(capped.effectiveChunkSeconds, 100);
  assert.match(capped.warning, /片段数超过上限 10/);
  assert.match(capped.warning, /100s/);
  // Growing past the frame budget is worth a louder warning than staying inside it.
  assert.match(capped.warning, /丢帧/);
  assertCoversTimeline(capped, 1000, 10);

  // When the grown chunk still fits the budget the warning stays calm.
  const calm = planChunks(1000, videoSettings({ maxChunkSeconds: 300, fps: 0.5, maxSegments: 10 }), 32768);
  assert.equal(calm.segments.length, 10);
  assert.equal(calm.effectiveChunkSeconds, 100);
  assert.ok(calm.effectiveChunkSeconds <= calm.maxChunkSeconds);
  assert.match(calm.warning, /片段数超过上限 10/);
  assert.doesNotMatch(calm.warning, /丢帧/);
  assertCoversTimeline(calm, 1000, 10);

  // A chunk that does not divide the duration evenly still lands exactly on it.
  const ragged = planChunks(1000, videoSettings({ maxSegments: 9 }), 32768);
  assert.equal(ragged.segments.length, 9);
  assertCoversTimeline(ragged, 1000, 9);

  // The cap only bites when the segment count would actually exceed it.
  const exact = planChunks(300, videoSettings({ maxSegments: 10, chunkSeconds: 30 }), 32768);
  assert.equal(exact.segments.length, 10);
  assert.equal(exact.warning, "");
  assertCoversTimeline(exact, 300, 10);
});

test("chunk planning rejects unusable durations and repairs unusable settings", () => {
  for (const duration of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(() => planChunks(duration, videoSettings(), 32768), /视频时长无效/);
  }
  // Zeroed or non-finite settings fall back rather than producing a runaway plan.
  const repaired = planChunks(100, videoSettings({ chunkSeconds: 0, maxChunkSeconds: Number.NaN, fps: 0, width: 0, maxSegments: 0 }), 32768);
  assert.equal(repaired.effectiveChunkSeconds, VIDEO.chunkSeconds);
  assert.equal(repaired.maxChunkSeconds, VIDEO.maxChunkSeconds);
  assert.equal(repaired.frameBudget, frameBudget(32768));
  assertCoversTimeline(repaired, 100, VIDEO.maxSegments);
  // A ceiling below the requested chunk length is honored, not swapped away.
  const clamped = planChunks(100, videoSettings({ chunkSeconds: 40, maxChunkSeconds: 25 }), 32768);
  assert.equal(clamped.effectiveChunkSeconds, 25);
  assertCoversTimeline(clamped, 100);
});

test("intermediate files are refused inside every registered library root", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-chunks-guard-"));
  const library = path.join(directory, "library");
  const other = path.join(directory, "other-library");
  const outside = path.join(directory, "work");
  fs.mkdirSync(library);
  fs.mkdirSync(other);

  assert.throws(() => assertOutsideLibraryRoots(library, [library]), /中间文件不能写入素材库目录/);
  assert.throws(() => assertOutsideLibraryRoots(path.join(library, "chunks"), [library]), /中间文件不能写入素材库目录/);
  assert.throws(() => assertOutsideLibraryRoots(path.join(library, "deep", "nested", "tmp"), [library]), /中间文件不能写入素材库目录/);
  // The message has to name both the offending path and the root it sits in.
  assert.throws(
    () => assertOutsideLibraryRoots(path.join(library, "chunks"), [other, library]),
    (error: Error) => error.message.includes(path.join(library, "chunks")) && error.message.includes(library),
  );
  // Dot segments are resolved before the containment test, so they cannot smuggle a path in.
  assert.throws(() => assertOutsideLibraryRoots(path.join(library, "sub", "..", "chunks"), [library]), /中间文件不能写入素材库目录/);
  assert.throws(() => assertOutsideLibraryRoots(`${library}/`, [`${library}/`]), /中间文件不能写入素材库目录/);
  // The parent directory of a library root is not inside it.
  assert.equal(assertOutsideLibraryRoots(outside, [library]), undefined);
  // A sibling that merely shares the root's name prefix is outside the root.
  assert.equal(assertOutsideLibraryRoots(`${library}-backup`, [library]), undefined);
  assert.equal(assertOutsideLibraryRoots(path.join(outside, "chunks"), [library, other]), undefined);
  // A missing root entry is skipped instead of matching everything.
  assert.equal(assertOutsideLibraryRoots(outside, ["", "   "]), undefined);
  assert.equal(assertOutsideLibraryRoots(library, []), undefined);
  fs.rmSync(directory, { recursive: true, force: true });
});

test("chunk extraction never writes a scratch directory into raw material", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-chunks-write-"));
  const library = path.join(directory, "library");
  const scratch = path.join(directory, "lancedb");
  fs.mkdirSync(library);
  const source = path.join(library, "clip.mp4");
  fs.writeFileSync(source, "not really a video");
  const inside = path.join(library, "chunks");
  await assert.rejects(
    () => extractChunk(source, 0, 10, videoSettings(), inside, [library]),
    /中间文件不能写入素材库目录/,
  );
  assert.equal(fs.existsSync(inside), false, "the guard must run before the directory is created");
  await assert.rejects(
    () => extractChunk(path.join(library, "missing.mp4"), 0, 10, videoSettings(), path.join(scratch, "chunks"), [library]),
    /素材不存在/,
  );
  assert.equal(fs.existsSync(scratch), false);
  fs.rmSync(directory, { recursive: true, force: true });
});

test("one preview frame is a cached JPEG written outside the library", { skip: HAS_FFMPEG ? false : "需要 ffmpeg" }, async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-frame-"));
  const library = path.join(directory, "library");
  const previews = path.join(directory, "lancedb", "previews");
  fs.mkdirSync(library, { recursive: true });
  const source = path.join(library, "clip.mp4");
  await run(
    "ffmpeg",
    [
      "-hide_banner", "-loglevel", "error", "-y",
      "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=5:duration=4",
      "-pix_fmt", "yuv420p", "-an",
      source,
    ],
    { encoding: "utf8", timeout: 120_000 },
  );

  // The scratch file this function writes ends in .tmp; ffmpeg has to be told
  // the output format, or every preview request dies on an unguessable suffix.
  const frame = await extractFrame(source, 1, { previewDir: previews, libraryRoots: [library] });
  assert.equal(path.dirname(frame), path.resolve(previews));
  const bytes = fs.readFileSync(frame);
  assert.deepEqual([...bytes.subarray(0, 3)], [0xff, 0xd8, 0xff], "the frame must be a real JPEG");
  assert.ok(bytes.length > 512);
  assert.deepEqual(fs.readdirSync(path.dirname(frame)).filter((name) => name.endsWith(".tmp")), [], "no scratch file may survive");

  // The same second of the same file must be served from the cache.
  const before = fs.statSync(frame).mtimeMs;
  assert.equal(await extractFrame(source, 1, { previewDir: previews, libraryRoots: [library] }), frame);
  assert.equal(fs.statSync(frame).mtimeMs, before, "a cache hit must not run ffmpeg again");

  // A requested second past the last decodable frame falls back to the head.
  const fallback = await extractFrame(source, 45, { previewDir: previews, libraryRoots: [library] });
  assert.notEqual(fallback, frame);
  assert.ok(fs.statSync(fallback).size > 512);

  // And the preview cache is never allowed inside someone's footage.
  await assert.rejects(
    () => extractFrame(source, 1, { previewDir: path.join(library, "previews"), libraryRoots: [library] }),
    /中间文件不能写入素材库目录/,
  );
  assert.equal(fs.existsSync(path.join(library, "previews")), false);
  fs.rmSync(directory, { recursive: true, force: true });
});
