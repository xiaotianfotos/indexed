import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { loadConfig, writeConfig } from "@indexed/config";
import { startServer } from "@indexed/server";

test("dashboard and redacted configuration are served locally", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-server-"));
  const embedding = http.createServer((_request, response) => {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ data: [{
      id: "draft-model",
      max_model_len: 8192,
      embedding_space: "draft-q8-space-fingerprint",
    }] }));
  });
  await new Promise<void>((resolve) => embedding.listen(0, "127.0.0.1", resolve));
  context.after(() => embedding.close());
  const embeddingAddress = embedding.address();
  const embeddingBaseUrl = `http://127.0.0.1:${typeof embeddingAddress === "object" && embeddingAddress ? embeddingAddress.port : 0}`;
  process.env.INDEXED_CONFIG = path.join(directory, "config.json");
  process.env.INDEXED_STATE_DIR = path.join(directory, "state");
  const started = await startServer({ host: "127.0.0.1", port: 0 });
  context.after(() => started.close());
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const page = await fetch(started.url);
  assert.equal(page.status, 200);
  const html = await page.text();
  assert.match(html, /检索/);
  assert.match(html, /素材目录/);
  assert.match(html, /追踪文件类型/);
  assert.match(html, /素材与扫描/);
  assert.match(html, /模型与数据库/);
  assert.match(html, /id="embedding-provider"/);
  assert.match(html, /Apple 原生 WeMM/);
  for (const mode of ["a", "b", "c", "d"]) assert.match(html, new RegExp(`option value="${mode}"`));
  assert.doesNotMatch(html, /option value="e"|id="apple-decoder-bundles"/);
  assert.match(html, /入库预估/);
  assert.match(html, /id="scan-engine-settings"/);
  assert.match(html, /<details class="scan-options config-advanced">/);
  assert.match(html, /aria-label="自动扫描"/);
  assert.doesNotMatch(html, /适合性能演示|选大类就够了|素材在后台索引，你不用管列表/);
  assert.doesNotMatch(html, /阿里云 OSS Vectors/);
  const dashboardScript = await fetch(`${started.url}app.js`);
  assert.equal(dashboardScript.status, 200);
  assert.match(dashboardScript.headers.get("content-type") || "", /text\/javascript/);
  const dashboardJavaScript = await dashboardScript.text();
  assert.doesNotMatch(dashboardJavaScript, /from\s+["']@indexed\//);
  const config = await fetch(`${started.url}api/config`).then((response) => response.json());
  assert.equal(config.activeProfile, "default");
  assert.equal(config.profiles.default.storage.accessKeySecret, "");
  assert.equal(config.performanceHistoryPath, path.join(directory, "state", "ingest-performance.json"));
  const extensionConfig = await fetch(`${started.url}api/extension-config`).then((response) => response.json());
  assert.equal(extensionConfig.embeddingDimension, 4096);
  assert.equal(extensionConfig.storageProvider, "local");
  assert.equal("ossAccessKeySecret" in extensionConfig, false);

  const invalid = await fetch(`${started.url}api/profiles/Bad_ID`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ label: "bad" }),
  });
  assert.equal(invalid.status, 400);

  const saved = await fetch(`${started.url}api/profiles/remote`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      label: "Remote",
      spaceId: "remote-space",
      embedding: { baseUrl: "http://embedding.test", model: "remote-model", dimension: 1024, inputStyle: "wemm" },
      storage: { provider: "local", path: "~/.indexed-remote", accessKeyId: "ak-value", accessKeySecret: "secret-value" },
    }),
  }).then((response) => response.json());
  assert.equal(saved.profiles.remote.storage.accessKeySecret, "");
  assert.equal(saved.profiles.remote.storage.hasAccessKeySecret, true);
  assert.equal(saved.profiles.remote.resolvedStoragePath, path.join(os.homedir(), ".indexed-remote"));
  assert.doesNotMatch(JSON.stringify(saved), /secret-value/);

  const tested = await fetch(`${started.url}api/profiles/test`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      id: "remote",
      profile: {
        embedding: { baseUrl: embeddingBaseUrl, model: "draft-model", dimension: 1024, inputStyle: "wemm" },
        storage: { provider: "local", path: "~/.indexed-draft" },
      },
    }),
  }).then((response) => response.json());
  assert.equal(tested.ok, true);
  assert.equal(tested.persisted, false);
  assert.equal(tested.model, "draft-model");
  assert.equal(tested.maxModelLen, 8192);
  assert.equal(tested.embeddingSpace, "draft-q8-space-fingerprint");
  assert.equal(tested.resolvedStoragePath, path.join(os.homedir(), ".indexed-draft"));
  assert.equal(loadConfig().profiles.remote.embedding.model, "remote-model");

  const cloudTest = await fetch(`${started.url}api/profiles/test`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id: "remote", profile: { storage: { provider: "aliyun" } } }),
  });
  assert.equal(cloudTest.status, 400);
  assert.match((await cloudTest.json()).error, /本地版仅支持 zvec/);

  const cloudSave = await fetch(`${started.url}api/profiles/cloud`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ storage: { provider: "aliyun" } }),
  });
  assert.equal(cloudSave.status, 400);
  assert.match((await cloudSave.json()).error, /本地版仅支持 zvec/);

  const legacyConfig = loadConfig();
  legacyConfig.profiles.cloud = {
    ...legacyConfig.profiles.default,
    storage: { ...legacyConfig.profiles.default.storage, provider: "aliyun" },
  };
  writeConfig(legacyConfig);
  const cloudActivate = await fetch(`${started.url}api/profiles/cloud/activate`, { method: "POST" });
  assert.equal(cloudActivate.status, 400);
  assert.match((await cloudActivate.json()).error, /本地版仅支持 zvec/);

  const activated = await fetch(`${started.url}api/profiles/remote/activate`, { method: "POST" }).then((response) => response.json());
  assert.equal(activated.activeProfile, "remote");
  const removed = await fetch(`${started.url}api/profiles/remote`, { method: "DELETE" }).then((response) => response.json());
  assert.equal(removed.activeProfile, "default");

  const modelPackage = path.join(directory, "apple-model");
  fs.mkdirSync(modelPackage);
  const helper = path.resolve("tests/fixtures/fake-apple-embedding-helper.mjs");
  const appleSaved = await fetch(`${started.url}api/profiles/default`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      spaceId: "",
      embedding: {
        provider: "apple-native",
        baseUrl: "",
        model: "custom-name-must-be-ignored",
        dimension: 256,
        inputStyle: "wemm",
        native: { binary: helper, modelPackage, mode: "fast", visionCompute: "ane", autoRestart: false },
      },
    }),
  }).then((response) => response.json());
  assert.equal(appleSaved.profiles.default.embedding.provider, "apple-native");
  assert.equal(appleSaved.profiles.default.embedding.model, "wemm-embedding-2b-apple-256");
  assert.equal(appleSaved.profiles.default.embedding.apiKey, "");

  const backend = await fetch(`${started.url}api/embedding-backend`).then((response) => response.json());
  assert.equal(backend.managed, true);
  assert.equal(backend.state, "ready");
  assert.equal(backend.health.status, "ready");
  assert.equal("apiKey" in backend, false);
  const appleTest = await fetch(`${started.url}api/profiles/test`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id: "default", profile: appleSaved.profiles.default }),
  }).then((response) => response.json());
  assert.equal(appleTest.ok, true);
  assert.equal(appleTest.persisted, false);
  assert.equal(appleTest.model, "wemm-embedding-2b-apple-256");
  assert.equal(appleTest.embeddingSpace, "fake-wemm-256-same-space");
  const appleExtension = await fetch(`${started.url}api/extension-config`).then((response) => response.json());
  assert.equal(appleExtension.embeddingBaseUrl, `${started.url.replace(/\/$/, "")}/api/embedding`);
  assert.equal(appleExtension.embeddingSpace, "fake-wemm-256-same-space");
  const appleModels = await fetch(`${appleExtension.embeddingBaseUrl}/v1/models`).then((response) => response.json());
  assert.equal(appleModels.data[0].embedding_space, appleExtension.embeddingSpace);
  const appleEmbedding = await fetch(`${appleExtension.embeddingBaseUrl}/v1/embeddings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: appleExtension.embeddingModel, input: "proxy" }),
  }).then((response) => response.json());
  assert.equal(appleEmbedding.data[0].embedding.length, 256);
  const simpleRequest = await fetch(`${appleExtension.embeddingBaseUrl}/v1/embeddings`, {
    method: "POST",
    body: JSON.stringify({ model: appleExtension.embeddingModel, input: "blocked simple request" }),
  });
  assert.equal(simpleRequest.status, 415);
  const simpleRestart = await fetch(`${started.url}api/embedding-backend/restart`, { method: "POST" });
  assert.equal(simpleRestart.status, 415);
});
