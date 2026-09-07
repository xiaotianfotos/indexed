import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { activeProfile, directConfig, loadConfig, setPath, writeConfig } from "@indexed/config";
import { listLocalVectors, putLocalVectors } from "@indexed/clients/local-vectors";
import { startServer } from "@indexed/server";

/**
 * Deterministic stand-in for the embedding service: the first component is the
 * length of the text it was handed, so a test can prove which version of a note a
 * vector was built from without talking to a real model.
 */
async function startEmbeddingStub(space: string, context: { after: (fn: () => unknown) => void }) {
  const seen: string[] = [];
  const stub = http.createServer((request, response) => {
    let raw = "";
    request.on("data", (chunk) => { raw += chunk; });
    request.on("end", () => {
      let text = "";
      try {
        const payload = JSON.parse(raw || "{}") as { messages?: Array<{ role?: string; content?: Array<{ text?: string }> }> };
        const user = [...(payload.messages || [])].reverse().find((message) => message.role === "user");
        text = String(user?.content?.find((item) => typeof item.text === "string")?.text || "");
      } catch { text = ""; }
      seen.push(text);
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ data: [{ embedding: [text.length % 997, space.length % 13] }] }));
    });
  });
  await new Promise<void>((resolve) => stub.listen(0, "127.0.0.1", resolve));
  context.after(() => stub.close());
  const address = stub.address();
  return { url: `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`, seen };
}

function useAssetEnvironment(spaceId: string, embeddingBaseUrl: string) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-document-"));
  const store = path.join(directory, "lancedb");
  process.env.INDEXED_CONFIG = path.join(directory, "config.json");
  process.env.INDEXED_DATA_DIR = store;
  const config = loadConfig();
  setPath(config, "activeProfile", "default");
  setPath(config, "profiles.default.storage.provider", "local");
  setPath(config, "profiles.default.storage.path", store);
  setPath(config, "profiles.default.storage.assetIndex", "library-assets");
  setPath(config, "profiles.default.embedding.baseUrl", embeddingBaseUrl);
  setPath(config, "profiles.default.embedding.model", "asset-test-model");
  setPath(config, "profiles.default.embedding.dimension", 2);
  setPath(config, "profiles.default.spaceId", spaceId);
  setPath(config, "library.autoScan", false);
  writeConfig(config);
  const library = path.join(directory, "library");
  fs.mkdirSync(library);
  const next = loadConfig();
  next.library = { ...next.library, roots: [library] };
  writeConfig(next);
  return { directory, store, library };
}

async function startAssetServer(directory: string, context: { after: (fn: () => unknown) => void }) {
  const started = await startServer({ host: "127.0.0.1", port: 0 });
  context.after(() => started.server.close());
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return started;
}

async function json(url: string, init?: RequestInit) {
  const headers = new Headers(init?.headers);
  if (init?.body && !headers.has("content-type")) headers.set("content-type", "application/json");
  const response = await fetch(url, { ...init, headers });
  const text = await response.text();
  return { status: response.status, text, body: JSON.parse(text) as any };
}

function rows(assetPath: string) {
  const direct = directConfig(activeProfile(loadConfig()));
  return listLocalVectors("library-assets", direct, {
    filter: { asset_path: { $eq: assetPath } },
    returnData: true,
  });
}

test("saving a note replaces its own vector and keeps the metadata contract", async (context) => {
  const embedding = await startEmbeddingStub("asset-edit-space", context);
  const environment = useAssetEnvironment("asset-edit-space", embedding.url);
  const started = await startAssetServer(environment.directory, context);
  const note = path.join(environment.library, "clip.md");
  // The shot-library convention: a note beside the clip it describes.
  const clip = path.join(environment.library, "clip.mp4");
  fs.writeFileSync(clip, "fake video bytes");
  fs.writeFileSync(note, "旧版脚本\r\n第二行\n");
  fs.chmodSync(note, 0o640);

  // The clip's own row carries a copy of that note for its card. It has to follow
  // the note, without its pixel-derived vector moving at all.
  const direct = directConfig(activeProfile(loadConfig()));
  await putLocalVectors("library-assets", [{
    key: "seeded-clip-media-row",
    data: { float32: [0.25, -0.5] },
    metadata: {
      record_type: "asset",
      embedding_space: "asset-edit-space",
      asset_path: clip,
      file_kind: "video",
      embedding_basis: "native_video_chunk_v1",
      text_preview: "旧版脚本",
    },
  }], direct);

  const first = await json(`${started.url}api/assets/document`, {
    method: "PUT",
    body: JSON.stringify({ path: note, text: "新版脚本\n海岸线航拍\n更长的一段文字\n" }),
  });
  assert.equal(first.status, 200, first.text);
  assert.equal(Object.keys(first.body).sort().join(","), "bytes,embeddingSpace,modifiedAt,name,notice,ok,path,previews,removed,vectors");
  assert.equal(first.body.ok, true);
  // macOS exposes the same temporary directory through /var and /private/var;
  // the API intentionally returns the canonical real path.
  assert.equal(first.body.path, fs.realpathSync(note));
  assert.equal(first.body.notice, "已按段落更新 1 个片段");
  assert.equal(first.body.vectors, 1);
  assert.equal(first.body.removed, 0);
  assert.equal(first.body.embeddingSpace, "asset-edit-space");
  assert.equal(first.body.previews, 1);

  // LF normalisation, requested permissions, and no temporary left behind.
  assert.equal(fs.readFileSync(note, "utf8"), "新版脚本\n海岸线航拍\n更长的一段文字\n");
  assert.equal(fs.statSync(note).mode & 0o777, 0o640);
  assert.deepEqual(fs.readdirSync(environment.library).sort(), ["clip.md", "clip.mp4"]);

  // The clip row keeps its key and its pixels-derived vector; only its quoted
  // excerpt travels with the note.
  const mediaRow = (await rows(clip))[0];
  assert.equal(mediaRow?.key, "seeded-clip-media-row");
  assert.deepEqual(mediaRow?.data?.float32, [0.25, -0.5]);
  assert.equal(mediaRow?.metadata?.text_preview, "新版脚本\n海岸线航拍\n更长的一段文字");
  assert.equal(mediaRow?.metadata?.embedding_basis, "native_video_chunk_v1");

  let stored = await rows(note);
  assert.equal(stored.length, 1);
  const metadata: Record<string, unknown> = stored[0]?.metadata || {};
  const stat = fs.statSync(note);
  assert.equal(metadata.record_type, "asset");
  assert.equal(metadata.embedding_space, "asset-edit-space");
  assert.equal(metadata.embedding_model, "asset-test-model");
  assert.equal(metadata.embedding_input_version, "qwen-text-system-user-v1");
  assert.equal(metadata.document_chunk_version, "paragraph-window-v1");
  assert.equal(metadata.file_kind, "document");
  assert.equal(metadata.extension, ".md");
  assert.equal(metadata.asset_name, "clip.md");
  assert.equal(metadata.library_root, environment.library);
  assert.equal(metadata.segment_index, 0);
  // The beside-the-clip text is recorded as a sidecar of that clip.
  assert.equal(metadata.embedding_basis, "sidecar_text_v1");
  assert.equal(metadata.sidecar_of, path.join(environment.library, "clip.mp4"));
  assert.equal(metadata.text_preview, "新版脚本\n海岸线航拍\n更长的一段文字");
  assert.equal(metadata.fingerprint, `${stat.size}:${Math.round(stat.mtimeMs)}`);
  assert.equal(stored[0]?.data?.float32?.[0], "新版脚本\n海岸线航拍\n更长的一段文字".length % 997);

  // A long note becomes multiple recall rows, including a passage near its tail.
  const longText = `${Array.from({ length: 80 }, (_, index) => `第 ${index + 1} 段\n${`海岸线素材说明 ${index + 1}。`.repeat(12)}`).join("\n\n")}\n\n尾部唯一关键词：琥珀色潮汐。`;
  const expanded = await json(`${started.url}api/assets/document`, {
    method: "PUT",
    body: JSON.stringify({ path: note, text: longText }),
  });
  assert.equal(expanded.status, 200);
  assert.ok(expanded.body.vectors > 1);
  stored = await rows(note);
  assert.equal(stored.length, expanded.body.vectors);
  assert.ok(embedding.seen.some((text) => text.includes("琥珀色潮汐")));

  // A shorter save must remove every previous tail chunk.
  const second = await json(`${started.url}api/assets/document`, {
    method: "PUT",
    body: JSON.stringify({ path: note, text: "缩写" }),
  });
  assert.equal(second.status, 200);
  assert.equal(second.body.vectors, 1);
  assert.ok(second.body.removed > 0);
  stored = await rows(note);
  assert.equal(stored.length, 1);
  assert.equal(stored[0]?.data?.float32?.[0], "缩写".length % 997);
  assert.equal(stored[0]?.metadata?.text_preview, "缩写");

  const read = await json(`${started.url}api/assets/document?asset=${encodeURIComponent(note)}`);
  assert.equal(read.status, 200);
  assert.equal(Object.keys(read.body).sort().join(","), "bytes,extension,libraryPath,modifiedAt,name,path,text,vectors");
  assert.equal(read.body.text, "缩写");
  assert.equal(read.body.vectors, 1);
  assert.equal(read.body.libraryPath, environment.library);

  // Emptying a note takes it out of recall instead of embedding an empty string.
  const cleared = await json(`${started.url}api/assets/document`, {
    method: "PUT",
    body: JSON.stringify({ path: note, text: "   \n" }),
  });
  assert.equal(cleared.status, 200);
  assert.equal(cleared.body.vectors, 0);
  assert.equal(cleared.body.removed, 1);
  assert.match(cleared.body.notice, /内容为空/);
  assert.equal((await rows(note)).length, 0);
  // An emptied note stops being quoted by its clip as well.
  assert.equal(cleared.body.previews, 1);
  assert.equal((await rows(clip))[0]?.metadata?.text_preview, "");
});

test("text editing stays inside registered libraries and note-like files", async (context) => {
  // No embedding endpoint: every case here has to be refused before any model call.
  const environment = useAssetEnvironment("asset-guard-space", "");
  const started = await startAssetServer(environment.directory, context);
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-outside-"));
  context.after(() => fs.rmSync(outside, { recursive: true, force: true }));

  const note = path.join(environment.library, "note.md");
  fs.writeFileSync(note, "可以编辑\n");
  const source = path.join(environment.library, "kernel.ts");
  fs.writeFileSync(source, "export const indexed = 1;\n");
  const clip = path.join(environment.library, "clip.mp4");
  fs.writeFileSync(clip, "video bytes");
  const escape = path.join(environment.library, "escape.md");
  fs.symlinkSync(path.join(outside, "private.md"), escape);
  fs.writeFileSync(path.join(outside, "private.md"), "库外的私密笔记\n");
  const gbk = path.join(environment.library, "gbk.txt");
  fs.writeFileSync(gbk, Buffer.from([0xb1, 0xb8, 0xb7, 0xdd]));

  const put = (body: unknown) => json(`${started.url}api/assets/document`, { method: "PUT", body: JSON.stringify(body) });

  // Source code is indexed as a document but is not editable material.
  const code = await put({ path: source, text: "export const indexed = 2;\n" });
  assert.equal(code.status, 400);
  assert.match(code.body.error, /不支持文本编辑/);
  assert.equal(fs.readFileSync(source, "utf8"), "export const indexed = 1;\n");

  // Media is never writable, and the reason names the extension.
  const media = await put({ path: clip, text: "overwrite" });
  assert.equal(media.status, 400);
  assert.match(media.body.error, /\.mp4 不支持文本编辑/);

  // The containment guard is the one the stream route uses, answer included.
  const symlink = await put({ path: escape, text: "escape" });
  assert.equal(symlink.status, 400);
  assert.match(symlink.body.error, /不在已注册/);
  assert.equal(fs.readFileSync(path.join(outside, "private.md"), "utf8"), "库外的私密笔记\n");

  const missing = await put({ path: path.join(environment.library, "gone.md"), text: "x" });
  assert.equal(missing.status, 404);
  assert.match(missing.body.error, /素材文件不存在/);

  const relative = await put({ path: "note.md", text: "x" });
  assert.equal(relative.status, 400);
  assert.match(relative.body.error, /绝对路径/);

  const noPath = await put({ text: "x" });
  assert.equal(noPath.status, 400);
  assert.match(noPath.body.error, /需要素材路径/);

  // A non-UTF-8 note is refused rather than rewritten as replacement characters.
  const unreadable = await json(`${started.url}api/assets/document?asset=${encodeURIComponent(gbk)}`);
  assert.equal(unreadable.status, 400);
  assert.match(unreadable.body.error, /不是 UTF-8/);
  assert.deepEqual([...fs.readFileSync(gbk)], [0xb1, 0xb8, 0xb7, 0xdd]);

  const tooLong = await put({ path: note, text: "a".repeat(1024 * 1024 + 1) });
  assert.equal(tooLong.status, 400);
  assert.match(tooLong.body.error, /文本过长/);
  assert.equal(fs.readFileSync(note, "utf8"), "可以编辑\n");

  const read = await json(`${started.url}api/assets/document?asset=${encodeURIComponent(note)}`);
  assert.equal(read.status, 200);
  assert.equal(read.body.text, "可以编辑\n");
  // Nothing reached the vector side while baseUrl was empty.
  assert.equal(read.body.vectors, 0);
});
