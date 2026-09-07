import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { loadConfig, setPath, writeConfig } from "@indexed/config";
import { putLocalVectors } from "@indexed/clients/local-vectors";
import { addLibrary, scanJob, queueScan } from "@indexed/core";
import { startServer } from "@indexed/server";

/**
 * Every case gets its own config file and its own zvec directory, so a route
 * can never read vectors or library roots registered by another case.
 */
function useAssetEnvironment(spaceId: string) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-asset-api-"));
  const store = path.join(directory, "lancedb");
  process.env.INDEXED_CONFIG = path.join(directory, "config.json");
  process.env.INDEXED_DATA_DIR = store;
  process.env.INDEXED_STATE_DIR = path.join(directory, "state");
  const config = loadConfig();
  setPath(config, "activeProfile", "default");
  setPath(config, "profiles.default.storage.provider", "local");
  setPath(config, "profiles.default.storage.path", store);
  setPath(config, "profiles.default.storage.assetIndex", "library-assets");
  // An empty embedding base url makes every model call fail locally instead of
  // reaching the network; only guard routes are exercised here.
  setPath(config, "profiles.default.embedding.baseUrl", "");
  setPath(config, "profiles.default.embedding.model", "asset-test-model");
  setPath(config, "profiles.default.embedding.dimension", 2);
  setPath(config, "profiles.default.spaceId", spaceId);
  // The scanner would otherwise walk real directories on server start.
  setPath(config, "library.autoScan", false);
  setPath(config, "library.scanIntervalSeconds", 901);
  setPath(config, "library.maxAssetsPerScan", 401);
  writeConfig(config);
  return directory;
}

function registerRoots(roots: string[]) {
  const config = loadConfig();
  config.library = { ...config.library, roots };
  writeConfig(config);
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
  return { status: response.status, type: response.headers.get("content-type") || "", text, body: JSON.parse(text) as any };
}

test("asset library status is served from the local profile", async (context) => {
  const directory = useAssetEnvironment("asset-status-space");
  const library = path.join(directory, "library");
  fs.mkdirSync(library);
  registerRoots([library]);
  const started = await startAssetServer(directory, context);

  const status = await json(`${started.url}api/assets/libraries`);
  assert.equal(status.status, 200);
  assert.equal(Object.keys(status.body).sort().join(","), "autoScan,libraries,maxAssetsPerScan,maxFilesPerLibrary,scanIntervalSeconds");
  assert.equal(status.body.autoScan, false);
  assert.equal(status.body.scanIntervalSeconds, 901);
  assert.equal(status.body.maxAssetsPerScan, 401);
  assert.equal(status.body.maxFilesPerLibrary, 100_000);
  assert.equal(status.body.libraries.length, 1);
  const entry = status.body.libraries[0];
  assert.equal(Object.keys(entry).sort().join(","), "assetCount,id,job,kinds,lastScanAt,name,path,state");
  assert.equal(entry.id, library);
  assert.equal(entry.path, library);
  assert.equal(entry.name, path.basename(library));
  assert.deepEqual(entry.kinds, ["video", "image", "document"]);
  assert.equal(entry.assetCount, 0);
  assert.equal(entry.lastScanAt, 0);
  assert.equal(entry.state, "idle");
  assert.equal(entry.job, null);

  // A configured root that vanishes is still reported, never silently dropped.
  const missing = await json(`${started.url}api/assets/libraries`);
  assert.deepEqual(missing.body.libraries.map((item: { id: string }) => item.id), [library]);
});

test("ingest test report is independent of registered directories", async (context) => {
  const directory = useAssetEnvironment("asset-benchmark-space");
  const library = path.join(directory, "demo");
  fs.mkdirSync(library);
  const started = await startAssetServer(directory, context);
  const before = loadConfig().library.roots;
  const report = await json(`${started.url}api/ingest-test`);
  assert.equal(report.status, 200);
  assert.equal(report.body.schema, 1);
  assert.deepEqual(loadConfig().library.roots, before);
});

test("asset library routes reject unusable targets", async (context) => {
  const directory = useAssetEnvironment("asset-library-routes-space");
  const library = path.join(directory, "library");
  const added = path.join(directory, "added");
  fs.mkdirSync(library);
  fs.mkdirSync(added);
  registerRoots([library]);
  const started = await startAssetServer(directory, context);

  const unknownRemoval = await json(`${started.url}api/assets/libraries?id=${encodeURIComponent(path.join(directory, "nope"))}`, { method: "DELETE" });
  assert.equal(unknownRemoval.status, 400);
  assert.match(unknownRemoval.body.error, /素材库未注册/);
  assert.equal(unknownRemoval.body.configPath, process.env.INDEXED_CONFIG);

  // Registering the directory queues its scan and persists the new root.
  const register = await json(`${started.url}api/assets/libraries`, { method: "POST", body: JSON.stringify({ path: added, kinds: ["video", "image"] }) });
  assert.equal(register.status, 200);
  assert.equal(register.body.ok, true);
  assert.equal(register.body.library.path, added);
  assert.deepEqual(loadConfig().library.roots, [library, added]);
  assert.deepEqual(loadConfig().library.rootKinds[added], ["video", "image"]);
  const listed = await json(`${started.url}api/assets/libraries`);
  assert.deepEqual(listed.body.libraries.map((item: { name: string }) => item.name), [path.basename(library), path.basename(added)]);
  assert.deepEqual(listed.body.libraries.find((item: { path: string }) => item.path === added).kinds, ["video", "image"]);

  // The picker can update an existing root's tracked kinds without duplicating it.
  const updated = await json(`${started.url}api/assets/libraries`, { method: "POST", body: JSON.stringify({ path: added, kinds: ["document"] }) });
  assert.equal(updated.status, 200);
  assert.deepEqual(updated.body.library.kinds, ["document"]);
  assert.deepEqual(loadConfig().library.rootKinds[added], ["document"]);
  const noKinds = await json(`${started.url}api/assets/libraries`, { method: "POST", body: JSON.stringify({ path: added, kinds: [] }) });
  assert.equal(noKinds.status, 400);
  assert.match(noKinds.body.error, /至少选择一种/);

  // A queued scan of an empty directory finishes without ever calling out.
  const queued = await json(`${started.url}api/assets/libraries/scan`, { method: "POST", body: JSON.stringify({ path: added }) });
  assert.equal(queued.status, 200);
  assert.ok(Array.isArray(queued.body.queued));
  const settled = await json(`${started.url}api/assets/libraries`);
  assert.ok(settled.body.libraries.some((item: { state: string }) => item.state === "done" || item.state === "idle"));

  // Repeat requests for a vanished directory are dropped instead of failing.
  const missing = await json(`${started.url}api/assets/libraries/scan`, { method: "POST", body: JSON.stringify({ path: path.join(directory, "ghost") }) });
  assert.deepEqual(missing.body.queued, []);
  assert.equal(scanJob(path.join(directory, "ghost")), null);

  // Unregistering keeps the remaining root and leaves the vectors alone.
  const removed = await json(`${started.url}api/assets/libraries?id=${encodeURIComponent(added)}`, { method: "DELETE" });
  assert.equal(removed.status, 200);
  assert.deepEqual(removed.body, { ok: true, removed: added, purged: false, scanCancelled: false });
  assert.deepEqual(loadConfig().library.roots, [library]);
  assert.equal(added in loadConfig().library.rootKinds, false);
});

test("asset list routes validate the query and report a count", async (context) => {
  const directory = useAssetEnvironment("asset-list-space");
  const library = path.join(directory, "library");
  fs.mkdirSync(library);
  registerRoots([library]);
  const started = await startAssetServer(directory, context);

  const listed = await json(`${started.url}api/assets`);
  assert.equal(listed.status, 200);
  assert.deepEqual(listed.body, { count: 0, assets: [] });

  for (const kind of ["video", "image", "document"]) {
    const filtered = await json(`${started.url}api/assets?kind=${kind}`);
    assert.equal(filtered.status, 200);
    assert.deepEqual(filtered.body, { count: 0, assets: [] });
  }

  // Unknown kinds are rejected before any vector table is opened.
  const bogus = await json(`${started.url}api/assets?kind=bogus`);
  assert.equal(bogus.status, 400);
  assert.match(bogus.body.error, /不支持的素材类型/);
  assert.equal(bogus.body.configPath, process.env.INDEXED_CONFIG);

  // Garbage limits degrade to the default instead of throwing or listing everything.
  const garbage = await json(`${started.url}api/assets?limit=abc`);
  assert.equal(garbage.status, 200);
  assert.deepEqual(garbage.body, { count: 0, assets: [] });

  // Search guards run before the embedding service is asked for anything.
  const emptySearch = await json(`${started.url}api/assets/search`, { method: "POST", body: JSON.stringify({ query: "  " }) });
  assert.equal(emptySearch.status, 400);
  assert.match(emptySearch.body.error, /搜索文字不能为空/);
  const badKind = await json(`${started.url}api/assets/search`, { method: "POST", body: JSON.stringify({ query: "clip", kind: "bogus" }) });
  assert.equal(badKind.status, 400);
  assert.match(badKind.body.error, /不支持的素材类型/);
});

test("asset file routes refuse any path outside the registered libraries", async (context) => {
  const directory = useAssetEnvironment("asset-stream-space");
  const library = path.join(directory, "library");
  const outside = path.join(directory, "outside");
  fs.mkdirSync(library);
  fs.mkdirSync(outside);
  const sentinel = "raw-bytes-must-not-leak";
  const secret = path.join(outside, "secret.mp4");
  fs.writeFileSync(secret, sentinel);
  const clip = path.join(library, "clip.mp4");
  fs.writeFileSync(clip, "mp4-bytes");
  const notes = path.join(library, "notes.bin");
  fs.writeFileSync(notes, "binary-notes");
  // A symlink inside the library still resolves to the file outside it.
  const escape = path.join(library, "escape.mp4");
  fs.symlinkSync(secret, escape);
  registerRoots([library]);
  const started = await startAssetServer(directory, context);
  const stream = (asset: string) => json(`${started.url}api/assets/stream?asset=${encodeURIComponent(asset)}`);

  // Serving a library file is the one case that returns bytes.
  const served = await fetch(`${started.url}api/assets/stream?asset=${encodeURIComponent(clip)}`);
  assert.equal(served.status, 200);
  assert.equal(served.headers.get("content-type"), "video/mp4");
  assert.equal(served.headers.get("accept-ranges"), "bytes");
  assert.equal(await served.text(), "mp4-bytes");

  for (const [asset, pattern] of [
    ["", /需要素材路径/],
    ["relative/clip.mp4", /绝对路径/],
    [secret, /不在已注册的素材库/],
    [path.join(library, "..", "outside", "secret.mp4"), /不在已注册的素材库/],
    [escape, /不在已注册的素材库/],
    [path.join(library, "missing.mp4"), /素材文件不存在/],
    [notes, /不支持的素材类型/],
  ] as Array<[string, RegExp]>) {
    const refused = await stream(asset);
    assert.match(String(refused.status), /^4\d\d$/);
    assert.match(refused.type, /application\/json/);
    assert.match(refused.body.error, pattern);
    assert.equal("raw-bytes-must-not-leak" in { [refused.text]: true }, false);
    assert.doesNotMatch(refused.text, /raw-bytes-must-not-leak|mp4-bytes|binary-notes/);
  }

  // Missing files inside the library answer 404, other refusals answer 400.
  assert.equal((await stream(path.join(library, "missing.mp4"))).status, 404);
  assert.equal((await stream("")).status, 400);
  assert.equal((await stream(secret)).status, 400);

  // Preview extraction is guarded the same way, before ffmpeg is ever spawned.
  const outsidePreview = await json(`${started.url}api/assets/preview?asset=${encodeURIComponent(secret)}`);
  assert.equal(outsidePreview.status, 400);
  assert.match(outsidePreview.body.error, /不在已注册的素材库/);
  const documentPreview = await json(`${started.url}api/assets/preview?asset=${encodeURIComponent(notes)}`);
  assert.equal(documentPreview.status, 400);
  assert.match(documentPreview.body.error, /不支持/);
});

test("asset routes without a handler answer the generic API 404", async (context) => {
  const directory = useAssetEnvironment("asset-404-space");
  const library = path.join(directory, "library");
  fs.mkdirSync(library);
  registerRoots([library]);
  const started = await startAssetServer(directory, context);

  // GET /api/assets is a real route and is covered by the listing case above.
  for (const route of ["api/assets/nothing", "api/nothing"]) {
    const response = await fetch(`${started.url}${route}`);
    assert.equal(response.status, 404);
    assert.deepEqual(await response.json(), { error: "Unknown API route" });
  }

  // Asset routes only answer the methods they actually implement.
  const wrongMethod = await fetch(`${started.url}api/assets/libraries`, { method: "DELETE" });
  assert.equal(wrongMethod.status, 400);
  assert.match((await wrongMethod.json()).error, /素材库未注册/);
});

test("asset prune removes rows whose file is gone", async (context) => {
  const spaceId = "asset-prune-space";
  const directory = useAssetEnvironment(spaceId);
  const library = path.join(directory, "library");
  fs.mkdirSync(library);
  registerRoots([library]);
  fs.writeFileSync(path.join(library, "keep.md"), "keep");
  const gone = path.join(library, "gone.mp4");
  const row = (key: string, metadata: Record<string, unknown>) => ({
    key,
    data: { float32: [1, 0] },
    metadata: {
      record_type: "asset",
      embedding_space: spaceId,
      embedding_model: "asset-test-model",
      library_root: library,
      indexed_at_ms: 1_700_000_000_000,
      ...metadata,
    },
  });
  await putLocalVectors("library-assets", [
    row("keep-1", { asset_path: path.join(library, "keep.md"), asset_name: "keep.md", file_kind: "document", extension: ".md" }),
    row("gone-1", { asset_path: gone, asset_name: "gone.mp4", file_kind: "video", extension: ".mp4", segment_index: 0 }),
    row("gone-2", { asset_path: gone, asset_name: "gone.mp4", file_kind: "video", extension: ".mp4", segment_index: 1 }),
  ], { localStorePath: path.join(directory, "lancedb"), embeddingSpace: spaceId });
  const started = await startAssetServer(directory, context);

  // Listing is honest about the disk first: dead rows need an explicit opt-in.
  assert.equal((await json(`${started.url}api/assets?limit=50`)).body.count, 1);
  assert.equal((await json(`${started.url}api/assets?limit=50&includeMissing=true`)).body.count, 3);

  const preview = await json(`${started.url}api/assets/prune`, { method: "POST", body: JSON.stringify({ dryRun: true }) });
  assert.equal(preview.status, 200);
  assert.equal(preview.body.dryRun, true);
  assert.equal(preview.body.removedRows, 0);
  assert.equal(preview.body.scanned, 3);
  assert.deepEqual([preview.body.missingAssets[0].path, preview.body.missingAssets[0].rows], [gone, 2]);
  assert.equal((await json(`${started.url}api/assets?limit=50&includeMissing=true`)).body.count, 3);

  const pruned = await json(`${started.url}api/assets/prune`, { method: "POST", body: JSON.stringify({}) });
  assert.equal(pruned.body.removedRows, 2);
  assert.deepEqual(pruned.body.missingAssets.map((item: { path: string }) => item.path), [gone]);
  assert.equal((await json(`${started.url}api/assets?limit=50&includeMissing=true`)).body.count, 1);
});

test("directory browse lists folders only, flagged with what is already registered", async (context) => {
  const directory = useAssetEnvironment("asset-browse-space");
  const root = path.join(directory, "footage");
  fs.mkdirSync(path.join(root, "shot-a", "nested"), { recursive: true });
  fs.mkdirSync(path.join(root, ".hidden"));
  fs.mkdirSync(path.join(root, "node_modules"));
  fs.writeFileSync(path.join(root, "clip.mp4"), "x");
  const note = path.join(root, "shot-a", "note.md");
  fs.writeFileSync(note, "secret frame description");
  registerRoots([root]);
  const started = await startAssetServer(directory, context);

  const listed = await (await fetch(`${started.url}api/assets/directories/browse?path=${encodeURIComponent(root)}`)).json();
  assert.equal(listed.path, root);
  assert.deepEqual(listed.kinds, ["video", "image", "document"]);
  assert.deepEqual(listed.documentFormats.map((item: { id: string }) => item.id), ["plain", "word", "pdf"]);
  // Files, dot directories and vendored directories stay out of the picker.
  assert.deepEqual(listed.entries.map((entry: { name: string }) => entry.name), ["shot-a"]);
  assert.equal(listed.entries[0].registered, false);
  assert.equal(listed.entries[0].containsLibrary, false);
  assert.equal(listed.parent, directory);

  const entered = await (await fetch(`${started.url}api/assets/directories/browse?path=${encodeURIComponent(path.join(root, "shot-a"))}`)).json();
  assert.deepEqual(entered.entries.map((entry: { name: string }) => entry.name), ["nested"]);

  const above = await (await fetch(`${started.url}api/assets/directories/browse?path=${encodeURIComponent(directory)}`)).json();
  const marked = above.entries.find((entry: { path: string }) => entry.path === root);
  assert.equal(marked.registered, true);

  // A file path is refused, and nothing about its contents leaks into the error.
  const refused = await fetch(`${started.url}api/assets/directories/browse?path=${encodeURIComponent(note)}`);
  assert.equal(refused.status, 400);
  assert.match((await refused.json()).error, /目录不存在/);
  assert.doesNotMatch(JSON.stringify(await fetch(`${started.url}api/assets/directories/browse?path=${encodeURIComponent(note)}`).then((response) => response.json())), /secret/);
});

test("coverage reports what a directory holds against what the index has", async (context) => {
  const spaceId = "asset-coverage-space";
  const directory = useAssetEnvironment(spaceId);
  const library = path.join(directory, "library");
  fs.mkdirSync(library);
  registerRoots([library]);
  const keep = path.join(library, "keep.md");
  const edit = path.join(library, "edit.md");
  const fresh = path.join(library, "fresh.png");
  fs.writeFileSync(keep, "keep");
  fs.writeFileSync(edit, "edit");
  fs.writeFileSync(fresh, "png");
  const keepStat = fs.statSync(keep);
  const row = (key: string, metadata: Record<string, unknown>) => ({
    key,
    data: { float32: [1, 0] },
    metadata: {
      record_type: "asset",
      embedding_space: spaceId,
      embedding_model: "asset-test-model",
      library_root: library,
      indexed_at_ms: 1_700_000_000_000,
      ...metadata,
    },
  });
  await putLocalVectors("library-assets", [
    row("keep-1", { asset_path: keep, asset_name: "keep.md", file_kind: "document", file_size: keepStat.size, modified_at_ms: Math.round(keepStat.mtimeMs), embedding_input_version: "qwen-text-system-user-v1", document_chunk_version: "paragraph-window-v1" }),
    row("edit-1", { asset_path: edit, asset_name: "edit.md", file_kind: "document", file_size: 4, modified_at_ms: fs.statSync(edit).mtimeMs - 5_000, embedding_input_version: "qwen-text-system-user-v1", document_chunk_version: "paragraph-window-v1" }),
    row("dead-1", { asset_path: path.join(library, "dead.mp4"), asset_name: "dead.mp4", file_kind: "video", file_size: 10, modified_at_ms: 1 }),
  ], { localStorePath: path.join(directory, "lancedb"), embeddingSpace: spaceId });
  const started = await startAssetServer(directory, context);

  const coverage = await (await fetch(`${started.url}api/assets/coverage`)).json();
  assert.equal(coverage.embeddingSpace, spaceId);
  assert.deepEqual(coverage.kinds, ["video", "image", "document"]);
  const item = coverage.libraries[0];
  assert.equal(item.path, library);
  assert.equal(item.name, "library");
  assert.deepEqual(item.kinds, ["video", "image", "document"]);
  assert.equal(item.files, 3);
  assert.equal(item.cached, 1);
  assert.equal(item.pending, 1);
  assert.equal(item.stale, 1);
  assert.equal(item.missing, 1);
  assert.equal(item.truncated, false);
  assert.deepEqual(item.byKind, {
    video: { files: 0, cached: 0, pending: 0, stale: 0 },
    image: { files: 1, cached: 0, pending: 1, stale: 0 },
    document: { files: 2, cached: 1, pending: 0, stale: 1 },
  });
  assert.deepEqual(coverage.totals, { files: 3, cached: 1, pending: 1, stale: 1, missing: 1 });
  assert.deepEqual(coverage.totalsByKind, item.byKind);
  assert.equal(coverage.ingestEstimate.pendingAssets, 2);
  assert.equal(coverage.ingestEstimate.source, "none");
  assert.equal(coverage.ingestEstimate.estimatedMs, null);
  assert.deepEqual(coverage.ingestEstimate.unavailableKinds, ["image", "document"]);
  const estimate = await json(`${started.url}api/assets/ingest-estimate`);
  assert.equal(estimate.status, 200);
  assert.equal(estimate.body.pendingAssets, 2);

  // A single directory can be re-walked on its own.
  const scoped = await (await fetch(`${started.url}api/assets/coverage?library=${encodeURIComponent(library)}&refresh=true`)).json();
  assert.equal(scoped.libraries.length, 1);
  assert.equal(scoped.totals.pending, 1);

  // Saving the background scan settings lands in the same config the CLI reads.
  const saved = await json(`${started.url}api/assets/scan-settings`, {
    method: "PUT",
    body: JSON.stringify({ autoScan: true, scanIntervalSeconds: 1_200, maxAssetsPerScan: 25, maxFilesPerLibrary: 75_000, nope: 1 }),
  });
  assert.equal(saved.status, 200);
  assert.equal(saved.body.autoScan, true);
  assert.equal(saved.body.scanIntervalSeconds, 1_200);
  assert.equal(saved.body.maxAssetsPerScan, 25);
  assert.equal(saved.body.maxFilesPerLibrary, 75_000);
  const config = loadConfig();
  assert.equal(config.library.autoScan, true);
  assert.equal(config.library.scanIntervalSeconds, 1_200);
  assert.equal(config.library.maxAssetsPerScan, 25);
  assert.equal(config.library.maxFilesPerLibrary, 75_000);
});

test("removing a library aborts its active embedding request", async (context) => {
  const directory = useAssetEnvironment("asset-cancel-space");
  const library = path.join(directory, "library");
  fs.mkdirSync(library);
  fs.writeFileSync(path.join(library, "cancel-me.md"), "cancel this background embedding request");
  registerRoots([library]);

  let markStarted: () => void = () => undefined;
  const embeddingStarted = new Promise<void>((resolve) => { markStarted = resolve; });
  let requestClosed = false;
  const embedding = http.createServer((request, response) => {
    if (request.url === "/v1/models") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ data: [{ id: "asset-test-model", max_model_len: 8192 }] }));
      return;
    }
    markStarted();
    request.once("close", () => { requestClosed = true; });
    // A real backend may still finish its current kernel, but cancelling fetch
    // must close the client request and prevent every following asset.
  });
  await new Promise<void>((resolve) => embedding.listen(0, "127.0.0.1", resolve));
  context.after(() => embedding.close());
  const address = embedding.address();
  const config = loadConfig();
  config.profiles.default.embedding.baseUrl = `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
  writeConfig(config);
  const started = await startAssetServer(directory, context);

  const queued = await json(`${started.url}api/assets/libraries/scan`, {
    method: "POST",
    body: JSON.stringify({ path: library }),
  });
  assert.equal(queued.status, 200);
  await Promise.race([
    embeddingStarted,
    new Promise((_, reject) => setTimeout(() => reject(new Error("embedding request did not start")), 5_000)),
  ]);
  const before = Date.now();
  const removed = await json(`${started.url}api/assets/libraries?id=${encodeURIComponent(library)}`, { method: "DELETE" });
  assert.equal(removed.status, 200);
  assert.equal(removed.body.scanCancelled, true);
  assert.ok(Date.now() - before < 2_000);
  for (let attempt = 0; attempt < 20 && !requestClosed; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  assert.equal(requestClosed, true);
  assert.deepEqual(loadConfig().library.roots, []);
});

test("overlapping directories are refused before they can double-index the same file", async () => {
  const directory = useAssetEnvironment("asset-overlap-space");
  const nested = path.join(directory, "footage", "shots");
  fs.mkdirSync(nested, { recursive: true });
  fs.mkdirSync(path.join(nested, "extra"));
  registerRoots([nested]);

  await assert.rejects(() => addLibrary(path.join(directory, "footage"), { scan: false }), /已经包含登记的素材目录/);
  await assert.rejects(() => addLibrary(path.join(nested, "extra"), { scan: false }), /在已登记的素材目录之内/);
  assert.deepEqual(loadConfig().library.roots, [nested]);

  const sibling = path.join(directory, "other");
  fs.mkdirSync(sibling);
  const added = await addLibrary(sibling, { scan: false });
  assert.equal(added.library.id, sibling);
  assert.equal(loadConfig().library.roots.length, 2);
});
