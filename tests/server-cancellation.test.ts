import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { listLocalVectors } from "@indexed/clients/local-vectors";
import { activeProfile, directConfig, loadConfig, writeConfig } from "@indexed/config";
import { scanLibrary } from "@indexed/core";
import { startServer } from "@indexed/server";

async function until(predicate: () => boolean) {
  const end = Date.now() + 3000;
  while (!predicate()) { assert(Date.now() < end, "Request lifetime did not finish"); await new Promise(resolve => setTimeout(resolve, 5)); }
}

test("search, capture ingestion and the embedding proxy propagate disconnect/deadline without leaking upstream work", { timeout: 20_000 }, async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-cancel-api-"));
  const oldConfig = process.env.INDEXED_CONFIG, oldState = process.env.INDEXED_STATE_DIR;
  process.env.INDEXED_CONFIG = path.join(root, "config.json"); process.env.INDEXED_STATE_DIR = path.join(root, "state");
  t.after(() => {
    if (oldConfig === undefined) delete process.env.INDEXED_CONFIG; else process.env.INDEXED_CONFIG = oldConfig;
    if (oldState === undefined) delete process.env.INDEXED_STATE_DIR; else process.env.INDEXED_STATE_DIR = oldState;
    fs.rmSync(root, { recursive: true, force: true });
  });
  let calls = 0, pauseModel = false;
  const active = new Set<http.ServerResponse>();
  const embedding = http.createServer((req, res) => {
    if (req.url === "/v1/models" && pauseModel) { calls++; active.add(res); res.on("close", () => active.delete(res)); req.resume(); return; }
    if (req.url === "/v1/models") { res.end(JSON.stringify({ data: [{ id: "fixture", max_model_len: 8192 }] })); return; }
    assert.equal(req.url, "/v1/embeddings");
    req.resume(); req.on("end", () => { calls++; active.add(res); res.on("close", () => active.delete(res)); });
  });
  await new Promise<void>(resolve => embedding.listen(0, "127.0.0.1", resolve));
  t.after(() => { embedding.closeAllConnections(); embedding.close(); });
  const address = embedding.address(); assert(address && typeof address === "object");
  const config = loadConfig();
  config.library = { ...config.library, roots: [], autoScan: false };
  config.profiles.default.embedding = { ...config.profiles.default.embedding, provider: "http",
    baseUrl: `http://127.0.0.1:${address.port}`, model: "fixture", dimension: 64 };
  config.profiles.default.storage = { ...config.profiles.default.storage, provider: "local", path: path.join(root, "vectors") };
  writeConfig(config);
  const server = await startServer({ host: "127.0.0.1", port: 0 }); t.after(() => server.close());
  const routes = ["api/assets/search", "api/local-library/search", "api/search", "api/search/image", "api/embedding/v1/embeddings", "api/ingest/transcript", "api/ingest/visual", "api/ingest/video"];
  for (const route of routes) {
    const controller = new AbortController(); const before = calls;
    const result = fetch(`${server.url}${route}`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "fixed synthetic query", kind: "document", imageBase64: "AA==", input: "fixture", text: "transcript", video_id: route, image_base64: "AA==", video_frames: [{ image_base64: "AA==", timestamp: 0 }] }), signal: controller.signal });
    const rejected = assert.rejects(result, { name: "AbortError" });
    await until(() => calls > before); controller.abort(); await rejected; await until(() => active.size === 0);
  }
  pauseModel = true;
  const discoveryController = new AbortController(), discoveryBefore = calls;
  const discovery = fetch(`${server.url}api/assets/model-info`, { signal: discoveryController.signal });
  const rejectedDiscovery = assert.rejects(discovery, { name: "AbortError" });
  await until(() => calls > discoveryBefore); discoveryController.abort(); await rejectedDiscovery; await until(() => active.size === 0);
  const library = path.join(root, "library"); fs.mkdirSync(library); fs.writeFileSync(path.join(library, "fixture.md"), "Synthetic document");
  config.library.roots = [library];
  const scanController = new AbortController(), scanBefore = calls;
  const scan = scanLibrary(library, { config, signal: scanController.signal });
  const rejectedScan = assert.rejects(scan, { name: "AbortError" });
  await until(() => calls > scanBefore); scanController.abort(); await rejectedScan; await until(() => active.size === 0);
  pauseModel = false;
  const direct = directConfig(activeProfile(config));
  for (const index of [direct.ossVisualIndex, direct.ossTranscriptIndex]) {
    assert.deepEqual(await listLocalVectors(index, direct), [], "Cancelled ingest must not write vectors");
  }
  const timed = await fetch(`${server.url}api/assets/search`, { method: "POST",
    headers: { "content-type": "application/json", "x-indexed-timeout-ms": "80" },
    body: JSON.stringify({ query: "fixed query", kind: "document" }) });
  assert.equal(timed.status, 504); assert.match((await timed.json()).error, /超时/); await until(() => active.size === 0);
  const before = calls;
  const invalid = await fetch(`${server.url}api/search`, { method: "POST",
    headers: { "content-type": "application/json", "x-indexed-timeout-ms": "0" }, body: '{"query":"fixture"}' });
  assert.equal(invalid.status, 400); await invalid.text(); assert.equal(calls, before);
  // An upload that never completes must receive a structured deadline response.
  const slow = await new Promise<{ status: number; body: string }>((resolve, reject) => {
    const request = http.request(`${server.url}api/search`, { method: "POST", headers: {
      "content-type": "application/json", "content-length": "1000", "x-indexed-timeout-ms": "80",
    } }, response => {
      let body = ""; response.on("data", chunk => { body += String(chunk); });
      response.on("end", () => { resolve({ status: response.statusCode!, body }); request.destroy(); });
    });
    request.on("error", reject); request.write("{");
  });
  assert.equal(slow.status, 504); assert.match(slow.body, /超时/); assert.equal(calls, before);
});
