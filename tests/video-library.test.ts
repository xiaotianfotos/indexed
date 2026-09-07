import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { putLocalVectors } from "@indexed/clients/local-vectors";
import { listVideos } from "@indexed/core";

const SPACE = "test-model-2-wemm-indexed-v1";

test("local video listing exposes activity and saved segment details", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-video-library-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const direct = { localStorePath: directory, embeddingSpace: SPACE };
  const metadata = {
    source_site: "bilibili",
    video_id: "BV-test",
    title: "Local video",
    source_url: "https://www.bilibili.com/video/BV-test",
    embedding_model: "test-model",
    embedding_space: SPACE,
    duration: 25,
    segment_interval: 10,
  };
  await putLocalVectors("video-visual", [
    { key: "visual-0", data: { float32: [1, 0] }, metadata: { ...metadata, segment_index: 0, indexed_at_ms: 100 } },
    { key: "visual-2", data: { float32: [0, 1] }, metadata: { ...metadata, segment_index: 2, indexed_at_ms: 300 } },
  ], direct);
  await putLocalVectors("video-transcript", [
    { key: "transcript-1", data: { float32: [1, 0] }, metadata: { ...metadata, segment_index: 1, indexed_at_ms: 200 } },
  ], direct);
  const config = {
    activeProfile: "default",
    library: { roots: [] },
    profiles: {
      default: {
        label: "Default",
        spaceId: SPACE,
        embedding: { baseUrl: "http://embedding.test", model: "test-model", dimension: 2, inputStyle: "wemm" },
        reranker: { enabled: false, baseUrl: "", model: "", candidates: 20 },
        storage: {
          provider: "local",
          path: directory,
          visualIndex: "video-visual",
          transcriptIndex: "video-transcript",
        },
      },
    },
  };

  const result = await listVideos({ limit: 10 }, config);
  assert.equal(result.count, 1);
  assert.deepEqual(result.videos[0], {
    key: "bilibili:BV-test",
    sourceSite: "bilibili",
    videoId: "BV-test",
    title: "Local video",
    channel: "",
    sourceUrl: "https://www.bilibili.com/video/BV-test",
    thumbnail: "",
    visualCount: 2,
    transcriptCount: 1,
    visualSegments: [0, 2],
    transcriptSegments: [1],
    duration: 25,
    segmentInterval: 10,
    indexedAt: 300,
    embeddingModel: "test-model",
    embeddingSpace: SPACE,
  });
});
