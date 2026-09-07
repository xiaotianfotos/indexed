import assert from "node:assert/strict";
import test from "node:test";
import {
  readWebmDurationMilliseconds,
  repairWebmBase64Duration,
} from "@indexed/clients/webm-duration";
import { prepareVideoEmbeddingInput } from "@indexed/core";
import { prepareCapturedVideoForIngest } from "../apps/extension/src/video-ingest.js";

function vint(value: number): number[] {
  assert.ok(value >= 0 && value < 127);
  return [0x80 | value];
}

function recorderWebmWithoutDuration(): Uint8Array {
  const timecodeScale = [0x2a, 0xd7, 0xb1, ...vint(3), 0x0f, 0x42, 0x40];
  const info = [0x15, 0x49, 0xa9, 0x66, ...vint(timecodeScale.length), ...timecodeScale];
  return Uint8Array.from([0x18, 0x53, 0x80, 0x67, ...vint(info.length), ...info]);
}

function base64(value: Uint8Array): string {
  return Buffer.from(value).toString("base64");
}

test("WebM duration repair is shared and idempotent", async () => {
  const source = recorderWebmWithoutDuration();
  assert.equal(readWebmDurationMilliseconds(source), null);

  const first = await repairWebmBase64Duration(base64(source), 10_000);
  assert.equal(first.changed, true);
  assert.ok(Math.abs((first.durationMilliseconds || 0) - 10_000) < 0.001);

  const second = await repairWebmBase64Duration(first.videoBase64, 10_000);
  assert.equal(second.changed, false);
  assert.equal(second.videoBase64, first.videoBase64);
});

test("extension prepares the same repaired WebM before choosing a backend", async () => {
  const original = base64(recorderWebmWithoutDuration());
  const prepared = await prepareCapturedVideoForIngest({
    video_filename: "youtube-segment.webm",
    video_base64: original,
    captured_seconds: 10,
  });
  assert.notEqual(prepared.video_base64, original);
  const bytes = Buffer.from(String(prepared.video_base64), "base64");
  assert.ok(Math.abs((readWebmDurationMilliseconds(bytes) || 0) - 10_000) < 0.001);
});

test("server repairs legacy local payloads and rejects an invalid WebM before WEMM", async () => {
  const prepared = await prepareVideoEmbeddingInput({
    video_filename: "legacy.webm",
    video_base64: base64(recorderWebmWithoutDuration()),
    captured_seconds: 10,
  });
  assert.equal(prepared.mimeType, "video/webm");
  assert.ok(Math.abs((readWebmDurationMilliseconds(Buffer.from(prepared.videoBase64, "base64")) || 0) - 10_000) < 0.001);

  await assert.rejects(
    prepareVideoEmbeddingInput({
      video_filename: "broken.webm",
      video_base64: Buffer.from("not a webm").toString("base64"),
      captured_seconds: 10,
    }),
    /缺少有效 duration/,
  );
});

test("MP4 payloads remain byte-for-byte unchanged", async () => {
  const prepared = await prepareVideoEmbeddingInput({
    video_filename: "segment.mp4",
    video_base64: "original-mp4-base64",
    captured_seconds: 10,
  });
  assert.deepEqual(prepared, { mimeType: "video/mp4", videoBase64: "original-mp4-base64" });

  const extensionMp4 = { video_filename: "segment.MP4", video_base64: "cloud-mp4", captured_seconds: 10 };
  const extensionM4v = { video_filename: "segment.m4v", video_base64: "cloud-m4v", captured_seconds: 10 };
  assert.equal(await prepareCapturedVideoForIngest(extensionMp4), extensionMp4);
  assert.equal(await prepareCapturedVideoForIngest(extensionM4v), extensionM4v);
  assert.deepEqual(await prepareVideoEmbeddingInput(extensionM4v), {
    mimeType: "video/mp4",
    videoBase64: "cloud-m4v",
  });
});

test("ordered frame packets bypass recording repair and preserve timestamps", async () => {
  const frames = [
    { image_base64: "frame-a", mime_type: "image/jpeg", timestamp: 0 },
    { image_base64: "frame-b", mime_type: "image/jpeg", timestamp: 0.5 },
  ];
  const body = {
    video_frames: frames,
    video_filename: "",
    video_base64: "",
    captured_seconds: 1,
  };

  assert.equal(await prepareCapturedVideoForIngest(body), body);
  assert.deepEqual(await prepareVideoEmbeddingInput(body), {
    mimeType: "video/frames",
    videoBase64: "",
    videoFrames: frames,
  });
});
