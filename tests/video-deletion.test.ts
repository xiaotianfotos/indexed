import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { deleteVideo, prepareVideoDeletion } from "@indexed/core";
import { writeConfig } from "@indexed/config";
import { startServer } from "@indexed/server";

const cloudConfig = () => ({ activeProfile: "default", library: { roots: [], autoScan: false }, profiles: { default: {
  label: "Test", spaceId: "deletion-space", embedding: { model: "test-model", dimension: 3, baseUrl: "", inputStyle: "wemm" },
  storage: { provider: "aliyun", region: "test-region", accountId: "test-account", bucket: "test-bucket",
    visualIndex: "visual", transcriptIndex: "transcript", accessKeyId: "test-id", accessKeySecret: "test-secret" },
} } });

function mockCloud(context: { after: (fn: () => unknown) => void }) {
  const original = globalThis.fetch;
  const metadata = { embedding_model: "test-model", embedding_space: "deletion-space", source_site: "youtube", video_id: "video-1" };
  const visual = Array.from({ length: 501 }, (_, i) => ({ key: `visual-${i}`, metadata }));
  const transcript = [{ key: "transcript-0", metadata }];
  const deleted: Array<{ indexName: string; keys: string[] }> = [];
  let reads = 0;
  globalThis.fetch = async (input, init) => {
    const url = new URL(String(input));
    if (url.search === "?listVectors") {
      reads += 1;
      const request = JSON.parse(String(init?.body));
      return Response.json({ vectors: [...(request.indexName === "visual" ? visual : transcript),
        // The service must recheck provenance even if a storage implementation ignores its filter.
        { key: "unrelated", metadata: { ...metadata, embedding_space: "foreign-space" } },
        { key: "other-site", metadata: { ...metadata, source_site: "bilibili" } }] });
    }
    if (url.search === "?deleteVectors") {
      deleted.push(JSON.parse(String(init?.body)));
      return Response.json({});
    }
    if (!['127.0.0.1', 'localhost'].includes(url.hostname)) throw new Error(`Unexpected network access in deletion test: ${url.origin}`);
    return original(input, init);
  };
  context.after(() => { globalThis.fetch = original; });
  return { visual, deleted, reads: () => reads };
}

test("the actual Core cloud delete requires exact confirmation and rejects changed targets before deleting", async (context) => {
  const cloud = mockCloud(context);
  const config = cloudConfig();
  await assert.rejects(deleteVideo("youtube", "video-1", config), /显式确认/);
  assert.equal(cloud.reads(), 0);
  assert.equal(cloud.deleted.length, 0);
  const plan = await prepareVideoDeletion("youtube", "video-1", config);
  assert.equal(plan.visualCount, 501);
  assert.equal(plan.transcriptCount, 1);
  assert.doesNotMatch(JSON.stringify(plan), /test-secret|test-id|authorization/i);
  const confirmation = { schema: plan.schema, token: plan.token, confirmed: true };
  await assert.rejects(deleteVideo("youtube", "video-1", config, { confirmation: { ...confirmation, confirmed: false } }), /显式确认/);
  for (const field of ["region", "accountId", "bucket", "visualIndex", "transcriptIndex"] as const) {
    const changed = structuredClone(config);
    changed.profiles.default.storage[field] = "changed";
    await assert.rejects(deleteVideo("youtube", "video-1", changed, { confirmation }), /重新预览/);
  }
  const changedSpace = structuredClone(config);
  changedSpace.profiles.default.spaceId = "another-space";
  await assert.rejects(deleteVideo("youtube", "video-1", changedSpace, { confirmation }), /重新预览/);
  const changedProfile = { ...config, activeProfile: "other", profiles: { other: config.profiles.default } };
  await assert.rejects(deleteVideo("youtube", "video-1", changedProfile, { confirmation }), /重新预览/);
  await assert.rejects(deleteVideo("bilibili", "video-1", config, { confirmation }), /重新预览/);
  await assert.rejects(deleteVideo("youtube", "other-video", config, { confirmation }), /重新预览/);
  cloud.visual.push({ ...cloud.visual[0]!, key: "newly-indexed" });
  await assert.rejects(deleteVideo("youtube", "video-1", config, { confirmation }), /重新预览/);
  cloud.visual.pop();
  assert.equal(cloud.deleted.length, 0);
  const result = await deleteVideo("youtube", "video-1", config, { confirmation });
  assert.equal(result.visualDeleted, 501);
  assert.equal(result.transcriptDeleted, 1);
  assert.deepEqual(cloud.deleted.map((item) => item.keys.length), [500, 1, 1]);
  assert.equal(cloud.deleted.flatMap((item) => item.keys).some((key) => key === "unrelated" || key === "other-site"), false);
});

test("HTTP cloud deletion goes through the same Core confirmation boundary", async (context) => {
  const cloud = mockCloud(context);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-delete-api-"));
  const previous = { config: process.env.INDEXED_CONFIG, state: process.env.INDEXED_STATE_DIR };
  process.env.INDEXED_CONFIG = path.join(directory, "config.json");
  process.env.INDEXED_STATE_DIR = path.join(directory, "state");
  writeConfig(cloudConfig());
  const server = await startServer({ host: "127.0.0.1", port: 0 });
  context.after(async () => {
    await server.close();
    if (previous.config === undefined) delete process.env.INDEXED_CONFIG;
    else process.env.INDEXED_CONFIG = previous.config;
    if (previous.state === undefined) delete process.env.INDEXED_STATE_DIR;
    else process.env.INDEXED_STATE_DIR = previous.state;
    fs.rmSync(directory, { recursive: true, force: true });
  });
  const endpoint = `${server.url}api/videos/youtube/video-1`;
  assert.equal((await fetch(endpoint, { method: "DELETE" })).status, 428);
  assert.equal(cloud.deleted.length, 0);
  const preview = await fetch(`${endpoint}/deletion-preview`, { method: "POST" });
  assert.equal(preview.status, 200);
  const plan = await preview.json();
  assert.equal((await fetch(`${endpoint}/unexpected`, { method: "DELETE" })).status, 404);
  assert.equal(cloud.deleted.length, 0);
  const deleted = await fetch(endpoint, { method: "DELETE", headers: { "content-type": "application/json" },
    body: JSON.stringify({ confirmation: { confirmed: true, schema: plan.schema, token: plan.token } }) });
  assert.equal(deleted.status, 200);
  assert.equal((await deleted.json()).visualDeleted, 501);
});
