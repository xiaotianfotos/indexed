import assert from "node:assert/strict";
import test from "node:test";
import { embeddingInputStyle, embeddingSpace, metadataFilter, signVectorRequest } from "@indexed/clients/direct-cloud";

const config = {
  embeddingModel: "wemm-embedding-9b",
  embeddingDimension: 4096,
  embeddingInputStyle: "auto",
  ossRegion: "cn-hangzhou",
  ossAccountId: "1234567890",
  ossBucket: "demo-bucket",
  ossAccessKeyId: "test-id",
  ossAccessKeySecret: "test-secret",
};

test("embedding model resolves a stable semantic space", () => {
  assert.equal(embeddingInputStyle(config), "wemm");
  assert.equal(embeddingSpace(config), "wemm-embedding-9b-4096-wemm-indexed-v1");
  assert.deepEqual(metadataFilter({ embedding_model: config.embeddingModel, video_id: "abc" }), {
    $and: [
      { embedding_model: { $eq: "wemm-embedding-9b" } },
      { video_id: { $eq: "abc" } },
    ],
  });
});

test("OSS V4 request signing remains deterministic", async () => {
  const signed = await signVectorRequest("listVectors", config, new Date("2026-08-27T00:00:00Z"));
  assert.match(signed.url, /^https:\/\/demo-bucket-1234567890\.cn-hangzhou\.oss-vectors\.aliyuncs\.com\/\?listVectors$/);
  assert.match(signed.headers.authorization!, /^OSS4-HMAC-SHA256 Credential=test-id\//);
});
