import assert from "node:assert/strict";
import test from "node:test";
import {
  discoverEmbeddingConfig,
  embedImage,
  embeddingInputStyle,
  embeddingCorpusInputVersion,
  embeddingSpace,
  metadataFilter,
  signVectorRequest,
} from "@indexed/clients/direct-cloud";

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
  assert.equal(embeddingCorpusInputVersion(config, "text"), "wemm-text-user-v1");
  assert.equal(embeddingCorpusInputVersion({ ...config, embeddingProvider: "apple-native" }, "text"), "wemm-text-user-v1:apple-prompt-v2");
  assert.deepEqual(metadataFilter({ embedding_model: config.embeddingModel, video_id: "abc" }), {
    $and: [
      { embedding_model: { $eq: "wemm-embedding-9b" } },
      { video_id: { $eq: "abc" } },
    ],
  });
});

test("embedding service discovery resolves model, dimension and semantic space", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.endsWith("/v1/models")) {
      return new Response(JSON.stringify({ data: [{
        id: "wemm-embedding-9b",
        embedding_space: "wemm-q8-4-fingerprint",
      }] }), { status: 200 });
    }
    if (url.endsWith("/v1/embeddings")) {
      return new Response(JSON.stringify({ data: [{ embedding: [0.1, 0.2, 0.3, 0.4] }] }), { status: 200 });
    }
    return new Response("not found", { status: 404 });
  };
  try {
    assert.deepEqual(await discoverEmbeddingConfig({
      embeddingBaseUrl: "http://127.0.0.1:5007/",
      embeddingModel: "old-model",
    }), {
      baseUrl: "http://127.0.0.1:5007",
      model: "wemm-embedding-9b",
      dimension: 4,
      inputStyle: "wemm",
      space: "wemm-q8-4-fingerprint",
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("embedding discovery falls back for servers without an advertised space", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.endsWith("/v1/models")) {
      return new Response(JSON.stringify({ data: [{ id: "legacy-wemm" }] }), { status: 200 });
    }
    return new Response(JSON.stringify({ data: [{ embedding: [0.1, 0.2] }] }), { status: 200 });
  };
  try {
    const discovered = await discoverEmbeddingConfig({ embeddingBaseUrl: "http://legacy.test" });
    assert.equal(discovered.space, "legacy-wemm-2-wemm-indexed-v1");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("embedding requests retry transient helper overloads", async () => {
  const originalFetch = globalThis.fetch;
  let attempts = 0;
  globalThis.fetch = async () => {
    attempts += 1;
    if (attempts < 3) {
      return new Response("busy", { status: 429, headers: { "retry-after": "0" } });
    }
    return new Response(JSON.stringify({ data: [{ embedding: [0.6, 0.8] }] }), { status: 200 });
  };
  try {
    const vector = await embedImage("aW1hZ2U=", "image/png", "", {
      embeddingBaseUrl: "http://127.0.0.1:18768",
      embeddingModel: "wemm-embedding-2b-apple-2",
      embeddingDimension: 2,
      embeddingInputStyle: "wemm",
    });
    assert.deepEqual(vector, [0.6, 0.8]);
    assert.equal(attempts, 3);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("OSS V4 request signing remains deterministic", async () => {
  const signed = await signVectorRequest("listVectors", config, new Date("2026-08-27T00:00:00Z"));
  assert.match(signed.url, /^https:\/\/demo-bucket-1234567890\.cn-hangzhou\.oss-vectors\.aliyuncs\.com\/\?listVectors$/);
  assert.match(signed.headers.authorization!, /^OSS4-HMAC-SHA256 Credential=test-id\//);
});
