import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import zvec from "@zvec/zvec";
import {
  deleteLocalVectors, getLocalVectors, listLocalVectors, localVectorCollectionName, localVectorStatus,
  putLocalVectors, queryLocalVectors, type VectorRecord,
} from "@indexed/clients/local-vectors";

const row = (key: string, space: string, vector = [1, 0, 0]): VectorRecord => ({
  key, data: { float32: vector }, metadata: { embedding_space: space, title: key, nested: { tags: ["a", "中文"] } },
});

function temporary(context: { after: (fn: () => unknown) => void }) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-zvec-identity-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
}

function legacyStore(directory: string, rows: VectorRecord[], name = "images__review_space") {
  fs.mkdirSync(directory, { recursive: true });
  const collection = zvec.ZVecCreateAndOpen(path.join(directory, name), new zvec.ZVecCollectionSchema({
    name: "legacy_test", vectors: { name: "vector", dataType: zvec.ZVecDataType.VECTOR_FP32, dimension: 3,
      indexParams: { indexType: zvec.ZVecIndexType.FLAT, metricType: zvec.ZVecMetricType.COSINE } },
    fields: [{ name: "embedding_space", dataType: zvec.ZVecDataType.STRING },
      { name: "metadata_json", dataType: zvec.ZVecDataType.STRING }],
  }));
  try {
    if (rows.length) collection.upsertSync(rows.map((item) => ({ id: item.key, vectors: { vector: item.data!.float32! },
      fields: { embedding_space: String(item.metadata!.embedding_space), metadata_json: JSON.stringify(item.metadata) } })));
  } finally { collection.closeSync(); }
  fs.writeFileSync(path.join(directory, ".indexed-zvec.json"), JSON.stringify({ format: "indexed-zvec", version: 1,
    collections: { [name]: { dimension: 3, updatedAt: "2026-01-01T00:00:00.000Z" } } }));
}

function checksums(directory: string): Record<string, string> {
  const output: Record<string, string> = {};
  const visit = (current: string) => {
    for (const name of fs.readdirSync(current).sort()) {
      const file = path.join(current, name);
      if (fs.statSync(file).isDirectory()) visit(file);
      else output[path.relative(directory, file)] = crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
    }
  };
  visit(directory);
  return output;
}

test("raw index and space identities remain isolated across query, get, overwrite and delete", async (context) => {
  const directory = temporary(context);
  const cases = [
    { index: "images-a", space: "review-space" }, { index: "images_a", space: "review-space" },
    { index: "images-a", space: "review_space" }, { index: "images-a", space: " review-space " },
    { index: "图片", space: "中文/模型" }, { index: "图片!", space: "中文_模型" },
  ];
  assert.equal(new Set(cases.map(({ index, space }) => localVectorCollectionName(index, { embeddingSpace: space }))).size, cases.length);
  for (const [i, item] of cases.entries()) await putLocalVectors(item.index, [row("same-key", item.space, [1, i / 8, 0])], {
    localStorePath: directory, embeddingSpace: item.space,
  });
  for (const [i, item] of cases.entries()) {
    const config = { localStorePath: directory, embeddingSpace: item.space };
    assert.equal((await queryLocalVectors(item.index, [1, 0, 0], config)).vectors[0]?.metadata?.embedding_space, item.space);
    assert.deepEqual((await getLocalVectors(item.index, ["same-key"], config)).vectors[0]?.data?.float32, [1, i / 8, 0]);
    assert.equal((await listLocalVectors(item.index, config)).length, 1);
  }
  const first = { localStorePath: directory, embeddingSpace: cases[0]!.space };
  assert.deepEqual(await deleteLocalVectors(cases[0]!.index, ["same-key"], first), { deleted: 1 });
  assert.equal((await getLocalVectors(cases[1]!.index, ["same-key"], first)).vectors.length, 1);
  const manifest = JSON.parse(fs.readFileSync(path.join(directory, ".indexed-zvec.json"), "utf8"));
  assert.equal(manifest.version, 2);
  assert.equal(Object.keys(manifest.collections).length, cases.length);
});

test("writes reject foreign metadata and a caller cannot override the query space filter", async (context) => {
  const config = { localStorePath: temporary(context), embeddingSpace: "space-a" };
  await assert.rejects(putLocalVectors("images", [row("foreign", "space-b")], config), /embedding_space/);
  await putLocalVectors("images", [row("a", "space-a")], config);
  assert.equal((await queryLocalVectors("images", [1, 0, 0], config, { filter: { embedding_space: { $eq: "space-b" } } })).vectors.length, 0);
  const filename = path.join(config.localStorePath, ".indexed-zvec.json");
  const manifest = JSON.parse(fs.readFileSync(filename, "utf8"));
  manifest.collections[localVectorCollectionName("images", config)].identity.indexName = "images-tampered";
  fs.writeFileSync(filename, JSON.stringify(manifest));
  await assert.rejects(getLocalVectors("images", ["a"], config), /身份校验失败/);
});

test("v1 never silently becomes an empty or mixed v2 index", async (context) => {
  const directory = temporary(context);
  legacyStore(directory, [row("a", "review-space"), row("b", "review_space")]);
  const before = checksums(directory);
  const config = { localStorePath: directory, embeddingSpace: "review-space" };
  for (const operation of [
    () => listLocalVectors("images", config), () => getLocalVectors("images", ["a"], config),
    () => queryLocalVectors("images", [1, 0, 0], config), () => deleteLocalVectors("images", ["a"], config),
    () => putLocalVectors("images", [row("new", "review-space")], config),
  ]) await assert.rejects(operation(), /重新扫描建立索引/);
  assert.equal((await localVectorStatus(config)).rebuild_required, true);
  assert.deepEqual(checksums(directory), before);
});

test("v2 preserves FP32 originals across cosine engine reopen and repeated reads", async (context) => {
  const config = { localStorePath: temporary(context), embeddingSpace: "exact-fp32" };
  const input = row("fractional", config.embeddingSpace, [0.125, -0.5, 0.75]);
  await putLocalVectors("images", [input], config);
  for (let pass = 0; pass < 3; pass += 1) {
    const retrieved = await getLocalVectors("images", [input.key], config);
    assert.deepEqual(retrieved.vectors[0]?.data, input.data);
    assert.equal((await queryLocalVectors("images", [1, 0, 0], config)).vectors[0]?.key, input.key);
  }
});

