import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  deleteLocalVectors,
  getLocalVectors,
  listLocalVectors,
  localVectorDatabasePath,
  localVectorStatus,
  putLocalVectors,
  queryLocalVectors,
  withLocalVectorSession,
  type LocalVectorSessionTimings,
} from "@indexed/clients/local-vectors";

test("scan session reuses handles, sees writes, isolates spaces and closes on failure", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-zvec-session-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const config = { localStorePath: directory, embeddingSpace: "session" };
  let timings: LocalVectorSessionTimings | undefined;
  await withLocalVectorSession(config, async () => {
    for (let i = 0; i < 4; i++) {
      await putLocalVectors("images", [{ key: String(i), data: { float32: [1, 0] }, metadata: { embedding_space: config.embeddingSpace } }], config);
      assert.equal((await listLocalVectors("images", config)).length, i + 1);
      assert.equal((await queryLocalVectors("images", [1, 0], config)).vectors.length, i + 1);
    }
    await withLocalVectorSession(config, async () => {
      await deleteLocalVectors("images", ["0"], config);
    });
    assert.equal((await getLocalVectors("images", ["1"], config)).vectors.length, 1);
    assert.equal((await localVectorStatus(config)).collections[0]?.count, 3);
    assert.equal((await listLocalVectors("images", { ...config, embeddingSpace: "other" })).length, 0);
  }, (value) => { timings = value; });
  assert.equal(timings?.opens, 1);
  assert.equal((await listLocalVectors("images", config)).length, 3);
  await assert.rejects(withLocalVectorSession(config, async () => {
    await listLocalVectors("images", config);
    throw new Error("cancelled test scan");
  }), /cancelled test scan/);
  assert.equal((await listLocalVectors("images", config)).length, 3);
});

test("local vector paths expand the current user's home directory", () => {
  assert.equal(localVectorDatabasePath({ localStorePath: "~/.indexed" }), path.join(os.homedir(), ".indexed"));
});

test("local vectors default to INDEXED_HOME", () => {
  const previousHome = process.env.INDEXED_HOME;
  const previousData = process.env.INDEXED_DATA_DIR;
  try {
    process.env.INDEXED_HOME = "/Volumes/example/.indexed";
    delete process.env.INDEXED_DATA_DIR;
    assert.equal(
      localVectorDatabasePath({}),
      "/Volumes/example/.indexed/data/zvec",
    );
  } finally {
    if (previousHome === undefined) delete process.env.INDEXED_HOME;
    else process.env.INDEXED_HOME = previousHome;
    if (previousData === undefined) delete process.env.INDEXED_DATA_DIR;
    else process.env.INDEXED_DATA_DIR = previousData;
  }
});

test("a legacy LanceDB directory cannot be opened as zvec", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-legacy-guard-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  fs.mkdirSync(path.join(directory, "legacy.lance"));
  await assert.rejects(
    putLocalVectors("visual", [{ key: "one", data: { float32: [1, 0] }, metadata: {} }], { localStorePath: directory }),
    /仍是 LanceDB/,
  );
  assert.deepEqual(fs.readdirSync(directory), ["legacy.lance"]);
});

test("zvec persists, pre-filters, searches and deletes local vectors", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-zvec-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const config = { localStorePath: directory, embeddingSpace: "demo-space" };
  const indexName = "visual_timeline_with_a_long_existing_business_collection_name";
  const rows = [
    { key: "one", data: { float32: [1, 0, 0] }, metadata: { embedding_model: "demo", embedding_space: "demo-space", video_id: "a", source_site: "youtube", file_kind: "image" } },
    { key: "two", data: { float32: [0, 1, 0] }, metadata: { embedding_model: "demo", embedding_space: "demo-space", video_id: "b", source_site: "bilibili", file_kind: "video" } },
  ];
  await putLocalVectors(indexName, rows, config);
  assert.equal((await localVectorStatus(config)).provider, "zvec");
  assert.equal((await listLocalVectors(indexName, config)).length, 2);
  const result = await queryLocalVectors(indexName, [0.9, 0.1, 0], config, {
    limit: 1,
    filter: { $and: [{ file_kind: { $eq: "image" } }, { embedding_space: { $eq: "demo-space" } }] },
  });
  assert.ok(result.vectors[0]);
  assert.equal(result.vectors[0].key, "one");
  const exact = await getLocalVectors(indexName, ["two"], config, { returnData: true });
  assert.ok(exact.vectors[0]);
  assert.deepEqual(exact.vectors[0].data?.float32, [0, 1, 0]);
  await deleteLocalVectors(indexName, ["one"], config);
  assert.deepEqual((await listLocalVectors(indexName, config)).map((row) => row.key), ["two"]);
});
