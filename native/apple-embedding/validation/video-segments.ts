import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";
import { AppleEmbeddingBackend } from "@indexed/apple-embedding-backend";
import { planChunks, probeMedia, withFramePacket, type VideoChunkSettings, type VideoFramePacket } from "../../../packages/core/src/asset-chunks.js";

// Numerical/work-count regression for full-sized head/middle/tail segments,
// including two concurrent requests. Backend timing acceptance remains separate.
const { values } = parseArgs({ options: Object.fromEntries(["video", "package", "binary", "source-config", "mode", "output"].map(key => [key, { type: "string" as const }])) });
for (const key of ["video", "package", "binary", "source-config", "mode", "output"]) assert(values[key], `Missing --${key}`);
assert(values.mode === "c" || values.mode === "d");
const output = path.resolve(values.output!);
assert(!fs.existsSync(output), "Refusing to overwrite captured vectors");
const source = JSON.parse(fs.readFileSync(values["source-config"]!, "utf8"));
const video = source.profiles[source.activeProfile].video as VideoChunkSettings;
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-segment-quality-"));
process.env.INDEXED_HOME = path.join(scratch, "home");
const binary = path.resolve(values.binary!);
const backend = new AppleEmbeddingBackend({ binary, modelPackage: path.resolve(values.package!), executionMode: values.mode,
  coreMLCache: path.join(scratch, "coreml"), autoRestart: false });
try {
  const media = await probeMedia(values.video!);
  const plan = planChunks(media.duration, video, 8192);
  const selected = [...new Set([0, Math.floor(plan.segments.length / 2), plan.segments.length - 1])];
  const packets: VideoFramePacket[][] = [];
  for (const index of selected) {
    const segment = plan.segments[index]!;
    packets.push(await withFramePacket(values.video!, segment.startSeconds, segment.endSeconds, video, scratch, async frames => frames));
  }
  await backend.validate(true);
  const runtime = await backend.start();
  const base = runtime.baseUrl.replace(/\/v1\/?$/, "").replace(/\/$/, "");
  const headers = { "content-type": "application/json", Authorization: `Bearer ${runtime.apiKey}` };
  const embed = async (frames: VideoFramePacket[]) => {
    const body = JSON.stringify({ model: runtime.model, messages: [{ role: "user", content: [
      { type: "video_frames", frames: frames.map(frame => ({ image_url: { url: `data:${frame.mime_type};base64,${frame.image_base64}` }, timestamp: frame.timestamp })) },
      { type: "text", text: "Represent this video." },
    ] }] });
    const response = await fetch(`${base}/v1/embeddings`, { method: "POST", headers, body, signal: AbortSignal.timeout(60_000) });
    assert(response.ok, await response.clone().text());
    const result = await response.json() as { data: Array<{ embedding: number[] }>; usage: { prompt_tokens: number }; indexed: { embedding_space: string } };
    const vector = result.data[0]!.embedding;
    assert(vector.length === 2048 && vector.every(Number.isFinite));
    return { inputSHA256: createHash("sha256").update(body).digest("hex"), frames: frames.length, tokens: result.usage.prompt_tokens,
      embeddingSpace: result.indexed.embedding_space, vector };
  };
  const cases = [];
  for (const [index, packet] of packets.entries()) {
    const before = await backend.health();
    const result = await embed(packet);
    cases.push({ segment: plan.segments[selected[index]!]!, ...result, before, after: await backend.health() });
    console.log(`${values.mode} segment ${selected[index]}: ${result.frames} frames, ${result.tokens} tokens`);
  }
  const before = await backend.health();
  const concurrent = { before, samples: await Promise.all([embed(packets[0]!), embed(packets[2]!)]), after: await backend.health() };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify({ schema: 1, mode: values.mode, binarySHA256: createHash("sha256").update(fs.readFileSync(binary)).digest("hex"),
    video, duration: media.duration, cases, concurrent }, null, 2), { mode: 0o600 });
} catch (error) {
  console.error(backend.status().stderrTail.join("\n"));
  throw error;
} finally {
  await backend.stop();
  fs.rmSync(scratch, { recursive: true, force: true });
}
