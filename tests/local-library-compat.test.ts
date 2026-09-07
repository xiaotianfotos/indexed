import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { putLocalVectors } from "@indexed/clients/local-vectors";
import { listLocalFiles, searchLocalFiles } from "@indexed/core";

test("local library reads migrated legacy asset records and deduplicates chunks", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-assets-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const space = "wemm-embedding-9b-2-wemm-indexed-v1";
  const vectorConfig = { localStorePath: directory, embeddingSpace: space };
  await putLocalVectors("assets", [
    { key: "asset-a-0", data: { float32: [1, 0] }, metadata: { asset_id: "asset-a", title: "A", source_path: "/library/a.md", project_name: "Alpha", modality: "document", content_excerpt: "first", embedding_model: "wemm-embedding-9b", embedding_space: space } },
    { key: "asset-a-1", data: { float32: [0.9, 0.1] }, metadata: { asset_id: "asset-a", title: "A", source_path: "/library/a.md", project_name: "Alpha", modality: "document", content_excerpt: "second", embedding_model: "wemm-embedding-9b", embedding_space: space } },
    { key: "asset-b", data: { float32: [0, 1] }, metadata: { asset_id: "asset-b", title: "B", source_path: "/library/b.png", project_name: "Beta", modality: "image", embedding_model: "wemm-embedding-9b", embedding_space: space } },
  ], vectorConfig);
  const config = {
    activeProfile: "default",
    library: { roots: [] },
    profiles: {
      default: {
        label: "Default",
        spaceId: space,
        embedding: { baseUrl: "http://embedding.test", model: "wemm-embedding-9b", dimension: 2, inputStyle: "wemm" },
        reranker: { enabled: false, baseUrl: "", model: "", candidates: 20 },
        storage: { provider: "local", path: directory, region: "", accountId: "", bucket: "", visualIndex: "visual", transcriptIndex: "transcript", documentIndex: "assets" },
      },
    },
  };
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({ data: [{ embedding: [1, 0] }] }), { status: 200 });
  try {
    const listed = await listLocalFiles({ limit: 10 }, config);
    assert.equal(listed.files.length, 2);
    const searched = await searchLocalFiles("Alpha", { limit: 10 }, config);
    assert.equal(searched.files.length, 2);
    assert.equal(searched.files[0]?.assetId, "asset-a");
    assert.equal(searched.files[0]?.projectName, "Alpha");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
