// @ts-nocheck -- route behavior is preserved while API payload contracts are introduced.
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  AppleEmbeddingBackend,
  appleEmbeddingOptionsFromProfile,
  isAppleEmbeddingProfile,
} from "@indexed/apple-embedding-backend";
import {
  activeProfile,
  clearRuntimeEmbeddingOverride,
  configPath,
  directConfig,
  loadConfig,
  mergeProfile,
  publicConfig,
  runtimeEmbeddingOverride,
  setRuntimeEmbeddingOverride,
  writeConfig,
} from "@indexed/config";
import {
  addLibrary,
  assetModelInfo,
  browseAssetDirectories,
  cancelScan,
  hasActiveAssetScans,
  deleteVideo,
  prepareVideoDeletion,
  ingest,
  librariesStatus,
  libraryCoverage,
  listAssets,
  pruneMissingAssets,
  listLocalFiles,
  listVideos,
  queueScan,
  readAssetDocument,
  removeLibrary,
  resolveAssetForServing,
  resolveAssetPreview,
  scanLocalLibrary,
  search,
  searchAllAssets,
  searchImage,
  searchLocalFiles,
  startAutoScan,
  status,
  stopAutoScan,
  updateVideo,
  writeAssetDocument,
} from "@indexed/core";
import { PRODUCT_DEFAULTS } from "@indexed/contracts";
import { postEmbedding } from "@indexed/clients/embedding-transport";
import { withRequestOperation } from "./request-operation.js";
import { IngestTestController } from "./ingest-test.js";
import { assertRequestBodyHeaders, assertRequestSource, localListenHost, readJSONBody, RequestPolicyError } from "./request-policy.js";
import { bundledExtensionOrigins } from "./extension-origin.js";

const dashboardCandidates = [
  process.env.INDEXED_DASHBOARD_DIR,
  path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../dashboard"),
  path.resolve(process.cwd(), "dist/dashboard"),
  path.resolve(process.cwd(), "apps/dashboard"),
].filter(Boolean).map((value) => path.resolve(value));
const dashboardRoot = dashboardCandidates.find((candidate) => fs.existsSync(path.join(candidate, "index.html")))
  || dashboardCandidates.at(-1);

class ServerEmbeddingRuntime {
  backend = null;
  profileId = "";
  signature = "";

  async reconcile(config = loadConfig(), { force = false } = {}) {
    const profile = activeProfile(config);
    if (!isAppleEmbeddingProfile(profile)) {
      await this.stop();
      return null;
    }
    const signature = JSON.stringify({ id: profile.id, embedding: profile.embedding });
    if (!force && this.backend && this.signature === signature) {
      return this.backend.runtime || await this.backend.start();
    }
    await this.stop();
    const profileId = profile.id;
    let backend;
    backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile, {
      onStderr: (line) => process.stderr.write(`[indexed/apple-embedding] ${line}\n`),
      onStateChange: (state) => {
        if (this.backend !== backend || this.profileId !== profileId) return;
        if (state.state === "ready" && backend.runtime) {
          setRuntimeEmbeddingOverride(profileId, backend.runtime);
        } else if (state.state === "failed" || state.state === "stopped") {
          clearRuntimeEmbeddingOverride(profileId);
        }
      },
    }));
    this.backend = backend;
    this.profileId = profileId;
    this.signature = signature;
    try {
      const runtime = await backend.start();
      setRuntimeEmbeddingOverride(profileId, runtime);
      return runtime;
    } catch (error) {
      clearRuntimeEmbeddingOverride(profileId);
      throw error;
    }
  }

  async stop() {
    const backend = this.backend;
    const profileId = this.profileId;
    this.backend = null;
    this.profileId = "";
    this.signature = "";
    if (profileId) clearRuntimeEmbeddingOverride(profileId);
    await backend?.stop();
  }

  async restart(config = loadConfig()) {
    return this.reconcile(config, { force: true });
  }

  async prepare(config = loadConfig()) {
    const profile = activeProfile(config);
    if (!isAppleEmbeddingProfile(profile)) throw new Error("当前档案不是 apple-native embedding provider");
    await this.stop();
    const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile));
    let result;
    let preparationError;
    try {
      result = await backend.prepare();
    } catch (error) {
      preparationError = error;
    }
    await backend.stop();
    try {
      await this.reconcile(config, { force: true });
    } catch (restartError) {
      if (preparationError) {
        throw new AggregateError(
          [preparationError, restartError],
          "Core ML 准备失败，原 embedding 后端也未能恢复",
        );
      }
      throw restartError;
    }
    if (preparationError) throw preparationError;
    return result;
  }

  async validate(config = loadConfig(), full = false) {
    const profile = activeProfile(config);
    if (!isAppleEmbeddingProfile(profile)) throw new Error("当前档案不是 apple-native embedding provider");
    const backend = this.backend || new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile));
    return backend.validate(full);
  }

  async status({ health = true } = {}) {
    if (!this.backend) return { managed: false, state: "remote" };
    const state = this.backend.status();
    let helperHealth = null;
    if (health && state.state === "ready") {
      try {
        helperHealth = await this.backend.health();
      } catch (error) {
        helperHealth = { status: "unavailable", error: String(error?.message || error) };
      }
    }
    return { ...state, health: helperHealth };
  }
}

function json(response, statusCode, value) {
  response.writeHead(statusCode, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
  response.end(JSON.stringify(value));
}

const body = readJSONBody;

function assertLoopbackRequest(request) {
  const remoteAddress = String(request.socket.remoteAddress || "");
  if (remoteAddress === "::1" || remoteAddress.startsWith("127.") || remoteAddress.startsWith("::ffff:127.")) return;
  const error = new Error("Apple embedding 管理和代理接口只接受本机回环请求");
  error.status = 403;
  throw error;
}

function assertJSONRequest(request) {
  if (/^application\/json(?:\s*;|$)/i.test(String(request.headers["content-type"] || ""))) return;
  const error = new Error("Apple embedding POST 接口只接受 application/json");
  error.status = 415;
  throw error;
}

async function proxyEmbedding(request, response, endpoint) {
  assertLoopbackRequest(request);
  if (request.method === "POST") assertJSONRequest(request);
  return withRequestOperation(request, response, async operation => {
    const direct = directConfig(activeProfile());
    if (!direct.embeddingBaseUrl) throw new Error("当前 embedding 后端尚未就绪");
    const headers = { "content-type": "application/json" };
    if (direct.embeddingApiKey) headers.authorization = `Bearer ${direct.embeddingApiKey}`;
    const upstream = request.method === "POST"
      ? await postEmbedding(direct.embeddingBaseUrl, await body(request, operation.signal), headers,
          { signal: operation.signal, timeoutMs: operation.remainingMS, native: direct.embeddingProvider === "apple-native" })
      : await (async () => {
          const response = await fetch(`${direct.embeddingBaseUrl}${endpoint}`, { headers, signal: operation.signal });
          return { response, text: await response.text() };
        })();
    void operation.remainingMS;
    const data = Buffer.from(upstream.text);
    response.writeHead(upstream.response.status, {
      "content-type": upstream.response.headers.get("content-type") || "application/json; charset=utf-8",
      "cache-control": "no-store", "content-length": String(data.length),
    });
    response.end(data);
  });
}

function routeParts(url) {
  return decodeURIComponent(url.pathname).split("/").filter(Boolean);
}

function assertLocalProfile(profile) {
  if (profile?.storage?.provider && profile.storage.provider !== "local") {
    throw new Error("本地版仅支持 zvec");
  }
}

async function testProfileModel(config, id, embeddingRuntime) {
  const profile = activeProfile(config);
  if (!isAppleEmbeddingProfile(profile)) return assetModelInfo(config);
  const signature = JSON.stringify({ id: profile.id, embedding: profile.embedding });
  if (embeddingRuntime.profileId === id && embeddingRuntime.signature === signature && embeddingRuntime.backend?.runtime) {
    return assetModelInfo(config);
  }
  const previous = runtimeEmbeddingOverride(id);
  const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile));
  try {
    const runtime = await backend.start();
    setRuntimeEmbeddingOverride(id, runtime);
    return await assetModelInfo(config);
  } finally {
    if (previous) setRuntimeEmbeddingOverride(id, previous);
    else clearRuntimeEmbeddingOverride(id);
    await backend.stop();
  }
}

function mime(file) {
  return file.endsWith(".css") ? "text/css; charset=utf-8"
    : file.endsWith(".js") ? "text/javascript; charset=utf-8"
      : file.endsWith(".svg") ? "image/svg+xml; charset=utf-8"
      : "text/html; charset=utf-8";
}

const developmentDashboardRoot = dashboardCandidates.at(-1);

function serveAsset(url, response) {
  const name = url.pathname === "/" ? "index.html" : path.basename(url.pathname);
  const developmentAssets = { "app.js": "src/main.ts", "styles.css": "src/styles.css", "index.html": "index.html" };
  // Browser assets must come from the bundle when it exists. Serving main.ts as
  // app.js breaks its workspace imports in Chrome; the dev command builds first.
  const candidates = [path.join(dashboardRoot, name), path.join(developmentDashboardRoot, developmentAssets[name] || name)]
    .filter((file) => file.startsWith(dashboardRoot) || file.startsWith(developmentDashboardRoot))
    .filter((file) => fs.existsSync(file));
  const target = candidates[0];
  if (!target) return false;
  response.writeHead(200, { "content-type": mime(name), "cache-control": "no-cache" });
  response.end(fs.readFileSync(target));
  return true;
}

/** Stream a local media file with byte ranges so video seeking works in the browser. */
function sendFile(request, response, file, contentType, cacheControl = "no-store") {
  const total = fs.statSync(file).size;
  const range = /^bytes=(\d*)-(\d*)$/.exec(String(request.headers.range || ""));
  if (range && (range[1] || range[2])) {
    let start = range[1] ? Number(range[1]) : 0;
    let end = range[2] ? Number(range[2]) : total - 1;
    if (!range[1] && range[2]) {
      start = Math.max(0, total - Number(range[2]));
      end = total - 1;
    }
    end = Math.min(end, total - 1);
    if (start >= total || start > end) {
      response.writeHead(416, { "content-range": `bytes */${total}` });
      response.end();
      return;
    }
    response.writeHead(206, {
      "content-type": contentType,
      "content-range": `bytes ${start}-${end}/${total}`,
      "accept-ranges": "bytes",
      "content-length": String(end - start + 1),
      "cache-control": cacheControl,
    });
    fs.createReadStream(file, { start, end }).pipe(response);
    return;
  }
  response.writeHead(200, {
    "content-type": contentType,
    "content-length": String(total),
    "accept-ranges": "bytes",
    "cache-control": cacheControl,
  });
  fs.createReadStream(file).pipe(response);
}

async function handler(request, response, embeddingRuntime, ingestTest, sourcePolicy) {
  try {
    assertRequestSource(request, sourcePolicy);
    const url = new URL(request.url, "http://localhost");
    if (!url.pathname.startsWith("/api/")) {
      if (!serveAsset(url, response)) json(response, 404, { error: "Not found" });
      return;
    }
    assertRequestBodyHeaders(request);
    const parts = routeParts(url);
    if (url.pathname.startsWith("/api/ingest-test")) {
      assertLoopbackRequest(request);
      if (request.method === "GET" && url.pathname === "/api/ingest-test") {
        json(response, 200, ingestTest.report); return;
      }
      assertJSONRequest(request);
      if (request.method === "POST" && url.pathname === "/api/ingest-test/stop") {
        void ingestTest.stop().catch((error) => { ingestTest.report.error = String(error); });
        json(response, 202, ingestTest.report); return;
      }
      if (request.method === "POST" && url.pathname === "/api/ingest-test/start") {
        if (hasActiveAssetScans()) {
          json(response, 409, { error: "请先停止素材扫描，再开始入库测试" }); return;
        }
        const input = await body(request);
        json(response, 202, ingestTest.start(String(input.path || ""), Number(input.limit ?? 10), Number(input.repeats ?? 1))); return;
      }
    }
    if (ingestTest.busy && (request.method !== "GET" || /\/search|\/embedding\//.test(url.pathname))) {
      json(response, 409, { error: "入库测试正在独占模型，请等待测试结束或停止测试" }); return;
    }
    if (request.method === "GET" && url.pathname === "/api/embedding/v1/models") {
      await proxyEmbedding(request, response, "/v1/models");
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/embedding/v1/embeddings") {
      await proxyEmbedding(request, response, "/v1/embeddings");
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/embedding-backend") {
      assertLoopbackRequest(request);
      json(response, 200, await embeddingRuntime.status());
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/embedding-backend/restart") {
      assertLoopbackRequest(request);
      assertJSONRequest(request);
      await embeddingRuntime.restart();
      json(response, 200, await embeddingRuntime.status());
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/embedding-backend/prepare") {
      assertLoopbackRequest(request);
      assertJSONRequest(request);
      json(response, 200, { ok: true, ...(await embeddingRuntime.prepare()) });
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/embedding-backend/validate") {
      assertLoopbackRequest(request);
      assertJSONRequest(request);
      const input = await body(request);
      json(response, 200, { ok: true, ...(await embeddingRuntime.validate(loadConfig(), input.full === true)) });
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/config") {
      json(response, 200, publicConfig());
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/extension-config") {
      const profile = activeProfile();
      const direct = directConfig(profile);
      json(response, 200, {
        profileId: profile.id,
        label: profile.label,
        embeddingBaseUrl: isAppleEmbeddingProfile(profile)
          ? `http://127.0.0.1:${request.socket.localPort}/api/embedding`
          : direct.embeddingBaseUrl,
        embeddingModel: direct.embeddingModel,
        embeddingDimension: direct.embeddingDimension,
        embeddingInputStyle: direct.embeddingInputStyle,
        embeddingSpace: direct.embeddingSpace,
        embeddingProvider: direct.embeddingProvider,
        storageProvider: direct.storageProvider,
        ossRegion: direct.ossRegion,
        ossAccountId: direct.ossAccountId,
        ossBucket: direct.ossBucket,
        ossVisualIndex: direct.ossVisualIndex,
        ossTranscriptIndex: direct.ossTranscriptIndex,
      });
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/local-library") {
      json(response, 200, await listLocalFiles({ limit: Number(url.searchParams.get("limit") || 200) }));
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/local-library/search") {
      await withRequestOperation(request, response, async operation => {
        const input = await body(request, operation.signal);
        json(response, 200, await searchLocalFiles(input.query, { limit: Number(input.limit || 30), signal: operation.signal }));
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/local-library/scan") {
      const input = await body(request);
      json(response, 200, await scanLocalLibrary({ limit: Number(input.limit || 500) }));
      return;
    }
    if (request.method === "PUT" && url.pathname === "/api/local-library/roots") {
      const input = await body(request);
      const config = loadConfig();
      config.library = {
        ...config.library,
        roots: [...new Set((input.roots || []).map((item) => String(item).trim()).filter(Boolean))],
      };
      writeConfig(config);
      json(response, 200, { ok: true, roots: config.library.roots });
      return;
    }
    if (parts[1] === "assets") {
      if (request.method === "GET" && url.pathname === "/api/assets/libraries") {
        json(response, 200, await librariesStatus());
        return;
      }
      if (request.method === "POST" && url.pathname === "/api/assets/libraries") {
        const input = await body(request);
        json(response, 200, await addLibrary(String(input.path || ""), {
          kinds: Array.isArray(input.kinds) ? input.kinds : undefined,
        }));
        return;
      }
      if (request.method === "POST" && url.pathname === "/api/assets/libraries/scan") {
        const input = await body(request);
        json(response, 200, { queued: queueScan(String(input.path || ""), loadConfig(), { background: input.background === true }) });
        return;
      }
      if (request.method === "POST" && url.pathname === "/api/assets/libraries/stop") {
        const input = await body(request);
        json(response, 200, { ok: true, ...cancelScan(String(input.path || "")) });
        return;
      }
      if (request.method === "GET" && url.pathname === "/api/assets/directories/browse") {
        json(response, 200, browseAssetDirectories({ path: url.searchParams.get("path") || "" }));
        return;
      }
      if (request.method === "GET" && url.pathname === "/api/assets/coverage") {
        json(response, 200, await libraryCoverage({
          library: url.searchParams.get("library") || "",
          refresh: url.searchParams.get("refresh") === "true",
        }));
        return;
      }
      if (request.method === "GET" && url.pathname === "/api/assets/ingest-estimate") {
        json(response, 200, (await libraryCoverage()).ingestEstimate);
        return;
      }
      if (request.method === "PUT" && url.pathname === "/api/assets/scan-settings") {
        const input = await body(request);
        const config = loadConfig();
        const library = { ...(config.library || {}) };
        if (typeof input.autoScan === "boolean") library.autoScan = input.autoScan;
        if (Number.isFinite(Number(input.scanIntervalSeconds))) {
          library.scanIntervalSeconds = Math.min(86400, Math.max(30, Math.floor(Number(input.scanIntervalSeconds))));
        }
        if (Number.isFinite(Number(input.maxAssetsPerScan))) {
          library.maxAssetsPerScan = Math.min(5000, Math.max(1, Math.floor(Number(input.maxAssetsPerScan))));
        }
        if (Number.isFinite(Number(input.maxFilesPerLibrary))) {
          library.maxFilesPerLibrary = Math.min(1_000_000, Math.max(1_000, Math.floor(Number(input.maxFilesPerLibrary))));
        }
        config.library = library;
        writeConfig(config);
        // The timer reads its interval once, so a change only lands with a restart.
        stopAutoScan();
        const autoScan = startAutoScan(config);
        json(response, 200, { ok: true, autoScanStarted: autoScan.started, intervalSeconds: autoScan.intervalSeconds, ...(await librariesStatus(config)) });
        return;
      }
      if (request.method === "POST" && url.pathname === "/api/assets/prune") {
        const input = await body(request);
        json(response, 200, await pruneMissingAssets({
          library: String(input.path || input.library || ""),
          dryRun: input.dryRun === true,
        }));
        return;
      }
      if (request.method === "DELETE" && url.pathname === "/api/assets/libraries") {
        json(response, 200, await removeLibrary(String(url.searchParams.get("id") || ""), {
          purge: url.searchParams.get("purge") === "true",
        }));
        return;
      }
      if (request.method === "GET" && url.pathname === "/api/assets") {
        json(response, 200, await listAssets({
          limit: Number(url.searchParams.get("limit") || 200),
          kind: String(url.searchParams.get("kind") || ""),
          library: String(url.searchParams.get("library") || ""),
          includeMissing: url.searchParams.get("includeMissing") === "true",
        }));
        return;
      }
      if (request.method === "POST" && url.pathname === "/api/assets/search") {
        await withRequestOperation(request, response, async operation => {
          const input = await body(request, operation.signal);
          json(response, 200, await searchAllAssets(String(input.query || ""), {
            signal: operation.signal,
            kind: String(input.kind || ""),
            limit: Number(input.limit || 30),
            includeMissing: input.includeMissing === true,
            includeLegacy: input.includeLegacy !== false,
          }));
        });
        return;
      }
      if (request.method === "GET" && url.pathname === "/api/assets/model-info") {
        await withRequestOperation(request, response, async operation => {
          json(response, 200, await assetModelInfo(loadConfig(), { signal: operation.signal }));
        });
        return;
      }
      if (request.method === "GET" && url.pathname === "/api/assets/stream") {
        const asset = resolveAssetForServing(String(url.searchParams.get("asset") || ""));
        sendFile(request, response, asset.path, asset.contentType);
        return;
      }
      if (request.method === "GET" && url.pathname === "/api/assets/preview") {
        const frame = await resolveAssetPreview(
          String(url.searchParams.get("asset") || ""),
          Number(url.searchParams.get("at") || 0),
        );
        sendFile(request, response, frame.path, frame.contentType, "private, max-age=300");
        return;
      }
      if (request.method === "GET" && url.pathname === "/api/assets/document") {
        json(response, 200, await readAssetDocument(String(url.searchParams.get("asset") || "")));
        return;
      }
      if (request.method === "PUT" && url.pathname === "/api/assets/document") {
        const input = await body(request);
        json(response, 200, await writeAssetDocument(String(input.path || ""), String(input.text ?? "")));
        return;
      }
    }
    if (request.method === "POST" && url.pathname === "/api/profiles/test") {
      const input = await body(request);
      const id = String(input.id || "");
      if (!/^[a-z0-9][a-z0-9-]{1,48}$/.test(id)) throw new Error("配置档案 ID 只能使用小写英文、数字和短横线");
      assertLocalProfile(input.profile);
      const config = loadConfig();
      config.profiles[id] = mergeProfile(config.profiles[id] || {}, input.profile || {});
      if (!Object.prototype.hasOwnProperty.call(input.profile || {}, "spaceId")) {
        config.profiles[id].spaceId = "";
      }
      config.profiles[id].storage = { ...config.profiles[id].storage, provider: "local" };
      config.activeProfile = id;
      const info = await testProfileModel(config, id, embeddingRuntime);
      const visible = publicConfig(config);
      json(response, 200, {
        ok: true,
        ...info,
        resolvedStoragePath: visible.profiles[id]?.resolvedStoragePath || "",
        persisted: false,
      });
      return;
    }
    if (request.method === "PUT" && parts[1] === "profiles" && parts[2]) {
      const config = loadConfig();
      const id = parts[2];
      if (!/^[a-z0-9][a-z0-9-]{1,48}$/.test(id)) throw new Error("配置档案 ID 只能使用小写英文、数字和短横线");
      const profile = mergeProfile(config.profiles[id] || {}, await body(request));
      assertLocalProfile(profile);
      config.profiles[id] = profile;
      if (!config.activeProfile) config.activeProfile = id;
      writeConfig(config);
      if (config.activeProfile === id) {
        if (url.searchParams.get("defer") === "true") {
          void embeddingRuntime.reconcile(config).catch((error) => {
            process.stderr.write(`[indexed/apple-embedding] apply failed: ${String(error?.message || error)}\n`);
          });
        } else {
          await embeddingRuntime.reconcile(config);
        }
      }
      json(response, 200, publicConfig());
      return;
    }
    if (request.method === "DELETE" && parts[1] === "profiles" && parts[2]) {
      const config = loadConfig();
      const id = parts[2];
      if (!config.profiles[id]) throw new Error(`配置档案不存在：${id}`);
      if (Object.keys(config.profiles).length === 1) throw new Error("至少保留一个配置档案");
      delete config.profiles[id];
      if (config.activeProfile === id) config.activeProfile = Object.keys(config.profiles)[0];
      writeConfig(config);
      await embeddingRuntime.reconcile(config);
      json(response, 200, publicConfig());
      return;
    }
    if (request.method === "POST" && parts[1] === "profiles" && parts[2] && parts[3] === "activate") {
      const config = loadConfig();
      if (!config.profiles[parts[2]]) throw new Error(`配置档案不存在：${parts[2]}`);
      assertLocalProfile(config.profiles[parts[2]]);
      config.activeProfile = parts[2];
      writeConfig(config);
      await embeddingRuntime.reconcile(config);
      json(response, 200, publicConfig());
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/status") {
      json(response, 200, await status());
      return;
    }
    if (request.method === "POST" && parts[1] === "ingest" && parts[2]) {
      await withRequestOperation(request, response, async operation => {
        json(response, 200, await ingest(parts[2], await body(request, operation.signal), loadConfig(), { signal: operation.signal }));
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/search") {
      await withRequestOperation(request, response, async operation => {
        const input = await body(request, operation.signal);
        json(response, 200, await search(input.query, { limit: input.limit, videoId: input.videoId, signal: operation.signal }));
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/search/image") {
      await withRequestOperation(request, response, async operation => {
        const input = await body(request, operation.signal);
        json(response, 200, await searchImage(input.imageBase64, {
          signal: operation.signal,
          mimeType: input.mimeType,
          query: input.query,
          limit: input.limit,
          videoId: input.videoId,
        }));
      });
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/videos") {
      json(response, 200, await listVideos({ limit: Number(url.searchParams.get("limit") || 100) }));
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/extension-library") {
      const data = await listVideos({ limit: 1000 });
      const videos = data.videos.map((video) => ({
        video_id: video.videoId,
        source_site: video.sourceSite,
        title: video.title,
        channel_name: video.channel,
        source_url: video.sourceUrl,
        thumbnail_url: video.thumbnail,
        visual_count: video.visualCount,
        transcript_count: video.transcriptCount,
        saved_visual_segments: video.visualSegments,
        saved_transcript_segments: video.transcriptSegments,
        duration: video.duration,
        segment_interval: video.segmentInterval,
        last_indexed_at: video.indexedAt,
      }));
      json(response, 200, {
        videos,
        video_count: videos.length,
        visual_count: videos.reduce((sum, item) => sum + item.visual_count, 0),
        transcript_count: videos.reduce((sum, item) => sum + item.transcript_count, 0),
        storage_provider: activeProfile().storage.provider,
        storage_label: activeProfile().storage.provider === "local" ? "本地 zvec" : "阿里云 OSS Vectors",
      });
      return;
    }
    if (request.method === "POST" && parts[1] === "videos" && parts[2] && parts[3] && parts[4] === "deletion-preview" && parts.length === 5) {
      json(response, 200, await prepareVideoDeletion(parts[2], parts[3]));
      return;
    }
    if (["DELETE", "PATCH"].includes(request.method) && parts[1] === "videos" && parts[2] && parts[3] && parts.length === 4) {
      const site = parts[2];
      const id = parts[3];
      const result = request.method === "DELETE"
        ? await deleteVideo(site, id, loadConfig(), await body(request))
        : await updateVideo(site, id, await body(request));
      json(response, 200, result);
      return;
    }
    json(response, 404, { error: "Unknown API route" });
  } catch (error) {
    if (response.destroyed) return;
    if (response.headersSent) {
      response.destroy();
      return;
    }
    const errorStatus = Number(error?.status);
    if (errorStatus === 413 || !request.complete) response.setHeader("connection", "close");
    json(response, errorStatus >= 400 && errorStatus <= 599 ? errorStatus : 400, {
      error: String(error?.message || error),
      ...(error instanceof RequestPolicyError ? {} : { configPath: configPath() }),
    });
  }
}

export async function startServer(options = {}) {
  const config = loadConfig();
  const host = localListenHost(String(options.host || config.server.host || PRODUCT_DEFAULTS.server.host));
  const port = Number(options.port !== undefined && options.port !== "" ? options.port : (config.server.port || PRODUCT_DEFAULTS.server.port));
  const sourcePolicy = { host, extensionOrigins: bundledExtensionOrigins(dashboardRoot) };
  const embeddingRuntime = new ServerEmbeddingRuntime();
  let activeModelRequests = 0;
  const ingestTest = new IngestTestController({ runtime: embeddingRuntime,
    executionSnapshot: async () => (await embeddingRuntime.status()).health?.private_ane ?? null,
    canStart: () => { if (activeModelRequests) throw new Error("还有入库、检索或配置请求正在处理，请等待完成后再测试"); },
    onStart: () => stopAutoScan(),
    onFinish: () => { try { startAutoScan(loadConfig()); } catch { /* Nonlocal profiles have no auto scan. */ } },
  });
  const server = http.createServer((request, response) => {
    const counted = request.method !== "GET" && !String(request.url || "").startsWith("/api/ingest-test");
    if (counted) activeModelRequests += 1;
    void handler(request, response, embeddingRuntime, ingestTest, sourcePolicy).finally(() => { if (counted) activeModelRequests -= 1; });
  });
  // Asset writes and local-vector compaction can leave a renderer connection idle
  // for longer than Node's 5-second default. Keep it alive across those pauses so
  // Chromium/undici cannot race a just-expired socket on the next API call.
  server.keepAliveTimeout = 65_000;
  server.headersTimeout = 70_000;
  server.once("close", () => {
    stopAutoScan();
    cancelScan();
    void ingestTest.close().finally(() => embeddingRuntime.stop());
  });
  try {
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(port, host, resolve);
    });
  } catch (error) {
    await embeddingRuntime.stop();
    throw error;
  }
  const address = server.address();
  const boundPort = typeof address === "object" && address ? address.port : port;
  // The HTTP control plane is available immediately, even when a pure-ANE model
  // needs a long cold start. Callers can poll /api/embedding-backend or await this
  // promise when their next operation requires vectors.
  const embeddingReady = embeddingRuntime.reconcile(config);
  void embeddingReady.then(() => {
    try {
      if (!ingestTest.busy) startAutoScan(config);
    } catch {
      // The asset library only runs on a local profile; a cloud profile skips auto scan.
    }
  }, (error) => {
    process.stderr.write(`[indexed/apple-embedding] startup failed: ${String(error?.message || error)}\n`);
  });
  return {
    server,
    host,
    port: boundPort,
    url: `http://${host.includes(":") ? `[${host}]` : host}:${boundPort}/`,
    profile: activeProfile(config).id,
    embeddingBackend: embeddingRuntime,
    embeddingReady,
    close: async () => {
      stopAutoScan();
      cancelScan();
      await ingestTest.close();
      await embeddingRuntime.stop();
      if (!server.listening) return;
      await new Promise<void>((resolve, reject) => {
        server.close((error) => error ? reject(error) : resolve());
      });
    },
  };
}
