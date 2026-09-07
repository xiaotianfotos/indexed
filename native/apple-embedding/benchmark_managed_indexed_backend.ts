#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { activeProfile, directConfig, loadConfig, writeConfig } from "@indexed/config";
import { listAssets, scanLibrary } from "@indexed/core";
import { startServer } from "@indexed/server";

interface QueryCase { id: string; text: string; expected: string }

const QUERIES: QueryCase[] = [
  { id: "blue-marble-en", text: "a matte blue mug on a white marble counter beside a closed dark green book", expected: "mug_blue_marble.png" },
  { id: "blue-marble-zh", text: "白色大理石台面上的哑光蓝色杯子，旁边是一本合上的深绿色书", expected: "mug_blue_marble.png" },
  { id: "red-marble-en", text: "a matte red mug on a white marble counter beside a closed dark green book", expected: "mug_red_marble.png" },
  { id: "red-marble-zh", text: "白色大理石台面上的哑光红色杯子，旁边是一本合上的深绿色书", expected: "mug_red_marble.png" },
  { id: "blue-wood-en", text: "a matte blue mug on a warm wooden desk beside an open blank notebook", expected: "mug_blue_wood.png" },
  { id: "blue-wood-zh", text: "温暖木质桌面上的哑光蓝色杯子，旁边有一本摊开的空白笔记本", expected: "mug_blue_wood.png" },
  { id: "red-wood-en", text: "a matte red mug on a warm wooden desk beside an open blank notebook", expected: "mug_red_wood.png" },
  { id: "red-wood-zh", text: "温暖木质桌面上的哑光红色杯子，旁边有一本摊开的空白笔记本", expected: "mug_red_wood.png" },
  { id: "cat-window-en", text: "a cat sitting by a window", expected: "distractor_cat_window.png" },
  { id: "cat-chair-en", text: "a cat sitting on a chair", expected: "distractor_cat_chair.png" },
  { id: "red-car-en", text: "a red sports car", expected: "distractor_red_car.png" },
  { id: "red-car-zh", text: "一辆红色跑车", expected: "distractor_red_car.png" },
];

function argument(name: string, fallback = ""): string {
  const index = process.argv.indexOf(name);
  return index >= 0 ? String(process.argv[index + 1] || fallback) : fallback;
}

function required(name: string): string {
  const value = argument(name);
  if (!value) throw new Error(`缺少 ${name}`);
  return path.resolve(value);
}

function percentile(values: number[], fraction: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * fraction) - 1))] ?? 0;
}

function directoryBytes(root: string): number {
  if (!fs.existsSync(root)) return 0;
  return fs.readdirSync(root, { withFileTypes: true }).reduce((sum, entry) => {
    const target = path.join(root, entry.name);
    return sum + (entry.isDirectory() ? directoryBytes(target) : entry.isFile() ? fs.statSync(target).size : 0);
  }, 0);
}

async function json(url: string, init?: RequestInit): Promise<any> {
  const response = await fetch(url, init);
  const text = await response.text();
  const value = text ? JSON.parse(text) : {};
  if (!response.ok) throw new Error(`${url} 返回 ${response.status}: ${text.slice(0, 500)}`);
  return value;
}

async function closeServer(started: any): Promise<void> {
  await started.close();
}

async function main(): Promise<void> {
  const binary = required("--binary");
  const modelPackage = required("--package");
  const bundle = required("--bundle");
  const assets = path.resolve(argument("--assets", path.join(path.dirname(fileURLToPath(import.meta.url)), "mixed-query-set/assets")));
  const cache = path.resolve(argument("--coreml-cache", path.join(os.tmpdir(), "indexed-apple-coreml-cache")));
  const output = path.resolve(argument("--output", "managed-indexed-backend.json"));
  const mode = argument("--mode", "ane");
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-managed-apple-"));
  const storage = path.join(workspace, "zvec");
  process.env.INDEXED_CONFIG = path.join(workspace, "config.json");
  process.env.INDEXED_STATE_DIR = path.join(workspace, "state");

  const config = loadConfig();
  config.library = {
    ...config.library,
    roots: [assets],
    rootKinds: { [assets]: ["image"] },
    kinds: ["image"],
    autoScan: false,
    maxAssetsPerScan: 100,
  };
  config.profiles.default = {
    ...config.profiles.default,
    label: `WeMM Apple managed ${mode}`,
    spaceId: "",
    embedding: {
      ...config.profiles.default.embedding,
      provider: "apple-native",
      baseUrl: "",
      apiKey: "",
      model: "wemm-embedding-2b-apple-2048",
      dimension: 2048,
      inputStyle: "wemm",
      native: {
        ...config.profiles.default.embedding.native,
        binary,
        modelPackage,
        decoderBundles: mode === "fast" ? [] : [bundle],
        coreMLCache: cache,
        mode,
        visionCompute: "ane",
        decoderMinimumTokens: mode === "ane" ? 1 : 128,
        maxQueuedRequests: 16,
        autoRestart: true,
        maxRestarts: 3,
      },
    },
    storage: { ...config.profiles.default.storage, provider: "local", path: storage },
  };
  writeConfig(config);

  const serverStartedAt = performance.now();
  const started = await startServer({ host: "127.0.0.1", port: 0 });
  const serverListeningMs = performance.now() - serverStartedAt;
  await started.embeddingReady;
  const startupMs = performance.now() - serverStartedAt;
  let report: any;
  let helperPid: number | null = null;
  try {
    const backendBefore = await json(`${started.url}api/embedding-backend`);
    helperPid = backendBefore.pid;
    const publicConfig = await json(`${started.url}api/config`);
    const extensionConfig = await json(`${started.url}api/extension-config`);
    const proxyModels = await json(`${extensionConfig.embeddingBaseUrl}/v1/models`);
    const profile = activeProfile();
    const direct = directConfig(profile);

    const scanStartedAt = performance.now();
    const scan = await scanLibrary(assets, {
      config: loadConfig(),
      limit: 100,
      recordPerformance: true,
    });
    const scanWallMs = performance.now() - scanStartedAt;
    const indexedAssets = await listAssets({ kind: "image", limit: 100, config: loadConfig() });

    const rows = [];
    for (const item of QUERIES) {
      const startedAt = performance.now();
      const result = await json(`${started.url}api/assets/search`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: item.text, kind: "image", limit: 8, includeLegacy: false }),
      });
      const elapsedMs = performance.now() - startedAt;
      const files = (result.hits || []).map((hit: any) => path.basename(hit.path || ""));
      const index = files.indexOf(item.expected);
      rows.push({ ...item, elapsedMs, rank: index < 0 ? null : index + 1, top: files.slice(0, 5) });
    }
    const latencies = rows.map((row) => row.elapsedMs);
    const ranks = rows.map((row) => row.rank ?? Number.POSITIVE_INFINITY);
    const backendAfter = await json(`${started.url}api/embedding-backend`);
    report = {
      schema: "indexed-managed-apple-embedding/v1",
      measuredAt: new Date().toISOString(),
      machine: { platform: os.platform(), arch: os.arch(), cpu: os.cpus()[0]?.model, memoryBytes: os.totalmem() },
      configuration: {
        mode,
        dimension: 2048,
        assets,
        modelPackage,
        decoderBundle: bundle,
        configuredBaseUrl: "",
        configuredApiKey: "",
        configuredSpaceId: "",
      },
      lifecycle: {
        startupMs,
        serverListeningMs,
        managed: backendBefore.managed,
        state: backendBefore.state,
        helperPid,
        helperUrlIsLoopback: /^http:\/\/127\.0\.0\.1:\d+$/.test(backendBefore.baseUrl),
        publicConfigContainsRuntimeToken: JSON.stringify(publicConfig).includes("Bearer "),
        extensionUsesIndexedProxy: extensionConfig.embeddingBaseUrl.endsWith("/api/embedding"),
      },
      contract: {
        configuredModel: profile.embedding.model,
        runtimeModel: direct.embeddingModel,
        runtimeDimension: direct.embeddingDimension,
        runtimeEmbeddingSpace: direct.embeddingSpace,
        proxyEmbeddingSpace: proxyModels.data?.[0]?.embedding_space,
        spacesMatch: direct.embeddingSpace === proxyModels.data?.[0]?.embedding_space,
      },
      backend: { before: backendBefore, after: backendAfter },
      ingest: {
        ok: scan.ok,
        discovered: scan.discovered,
        indexed: scan.indexed,
        failed: scan.failed,
        skipped: scan.skipped,
        wallMs: scanWallMs,
        throughputPerSecond: scan.indexed / (scanWallMs / 1000),
        indexedRows: indexedAssets.count,
        storageBytes: directoryBytes(storage),
      },
      retrieval: {
        queryCount: rows.length,
        recallAt1: ranks.filter((rank) => rank <= 1).length / ranks.length,
        recallAt3: ranks.filter((rank) => rank <= 3).length / ranks.length,
        recallAt5: ranks.filter((rank) => rank <= 5).length / ranks.length,
        latencyMs: {
          median: percentile(latencies, 0.5),
          p95: percentile(latencies, 0.95),
          min: Math.min(...latencies),
          max: Math.max(...latencies),
        },
        queries: rows,
      },
    };
  } finally {
    await closeServer(started);
    let helperStopped = true;
    if (helperPid) {
      try { process.kill(helperPid, 0); helperStopped = false; } catch { helperStopped = true; }
    }
    if (report) report.lifecycle.helperStoppedWithServer = helperStopped;
    fs.rmSync(workspace, { recursive: true, force: true });
  }
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${String((error as Error)?.stack || error)}\n`);
  process.exitCode = 1;
});
