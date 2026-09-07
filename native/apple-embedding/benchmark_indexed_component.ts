#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { activeProfile, directConfig, loadConfig, spaceId } from "@indexed/config";
import {
  assetModelInfo,
  listAssets,
  readIngestPerformanceHistory,
  searchAllAssets,
} from "@indexed/core";

interface QueryCase {
  id: string;
  text: string;
  expected: string;
}

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

function argument(name: string): string {
  const index = process.argv.indexOf(name);
  return index >= 0 ? String(process.argv[index + 1] || "") : "";
}

function percentile(values: number[], fraction: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * fraction) - 1))] ?? 0;
}

function directoryBytes(root: string): number {
  if (!root || !fs.existsSync(root)) return 0;
  let total = 0;
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const target = path.join(root, entry.name);
    if (entry.isDirectory()) total += directoryBytes(target);
    else if (entry.isFile()) total += fs.statSync(target).size;
  }
  return total;
}

async function main(): Promise<void> {
  const output = path.resolve(argument("--output") || "indexed-component-benchmark.json");
  const serverUrl = argument("--server-url").replace(/\/+$/, "");
  const config = loadConfig();
  const profile = activeProfile(config);
  const direct = directConfig(profile);
  const model = await assetModelInfo(config);
  const helperHealthBefore = await fetch(`${direct.embeddingBaseUrl}/health`).then((response) => response.json());
  const assets = await listAssets({ kind: "image", limit: 100, config });

  // Warm zvec and the resident query path without including it in the latency distribution.
  await searchAllAssets("warmup image retrieval", {
    kind: "image", limit: 8, includeLegacy: false, config,
  });

  const results = [];
  for (const item of QUERIES) {
    const started = performance.now();
    const searched = await searchAllAssets(item.text, {
      kind: "image", limit: 8, includeLegacy: false, config,
    });
    const elapsedMs = performance.now() - started;
    const hits = searched.hits.map((hit) => ({
      file: path.basename(hit.path || ""),
      score: Number(hit.score.toFixed(6)),
    }));
    const rankIndex = hits.findIndex((hit) => hit.file === item.expected);
    results.push({
      ...item,
      elapsedMs: Number(elapsedMs.toFixed(3)),
      rank: rankIndex < 0 ? null : rankIndex + 1,
      top: hits.slice(0, 5),
    });
  }

  const latencies = results.map((item) => item.elapsedMs);
  const ranked = results.map((item) => item.rank ?? Number.POSITIVE_INFINITY);
  const helperHealthAfter = await fetch(`${direct.embeddingBaseUrl}/health`).then((response) => response.json());
  const history = readIngestPerformanceHistory();
  let dashboardApi: Record<string, unknown> | null = null;
  if (serverUrl) {
    const modelInfo = await fetch(`${serverUrl}/api/assets/model-info`).then((response) => response.json());
    const listed = await fetch(`${serverUrl}/api/assets?kind=image&limit=100`).then((response) => response.json());
    const apiQueries = [];
    for (const item of QUERIES) {
      const started = performance.now();
      const searched = await fetch(`${serverUrl}/api/assets/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: item.text, kind: "image", limit: 8, includeLegacy: false }),
      }).then((response) => response.json());
      const elapsedMs = performance.now() - started;
      const files = (searched.hits || []).map((hit: { path?: string }) => path.basename(hit.path || ""));
      const rankIndex = files.indexOf(item.expected);
      apiQueries.push({
        id: item.id,
        elapsedMs: Number(elapsedMs.toFixed(3)),
        rank: rankIndex < 0 ? null : rankIndex + 1,
        top: files.slice(0, 5),
      });
    }
    const apiLatencies = apiQueries.map((item) => item.elapsedMs);
    const apiRanks = apiQueries.map((item) => item.rank ?? Number.POSITIVE_INFINITY);
    dashboardApi = {
      url: serverUrl,
      modelInfo,
      imageRows: listed.count,
      queryCount: apiQueries.length,
      recallAt1: apiRanks.filter((rank) => rank <= 1).length / apiRanks.length,
      latencyMs: {
        median: Number(percentile(apiLatencies, 0.5).toFixed(3)),
        p95: Number(percentile(apiLatencies, 0.95).toFixed(3)),
        min: Number(Math.min(...apiLatencies).toFixed(3)),
        max: Number(Math.max(...apiLatencies).toFixed(3)),
      },
      queries: apiQueries,
    };
  }
  const report = {
    schemaVersion: 1,
    measuredAt: new Date().toISOString(),
    machine: {
      arch: os.arch(),
      platform: os.platform(),
      release: os.release(),
      cpu: os.cpus()[0]?.model || "unknown",
      logicalCpuCount: os.cpus().length,
      memoryBytes: os.totalmem(),
    },
    indexed: {
      configPath: process.env.INDEXED_CONFIG || "",
      storageProvider: direct.storageProvider,
      storagePath: direct.localStorePath,
      storageBytes: directoryBytes(direct.localStorePath),
      model: profile.embedding.model,
      dimension: profile.embedding.dimension,
      inputStyle: profile.embedding.inputStyle,
      embeddingSpace: spaceId(profile),
      discoveredModel: model,
      imageRows: assets.count,
      imageFiles: [...new Set(assets.assets.map((asset) => asset.path))].length,
      ingestSamples: history.samples,
    },
    helper: {
      url: direct.embeddingBaseUrl,
      before: helperHealthBefore,
      after: helperHealthAfter,
    },
    retrieval: {
      mode: "Indexed resident core, text-to-image, kind=image, no legacy, topK=8",
      queryCount: results.length,
      recallAt1: ranked.filter((rank) => rank <= 1).length / ranked.length,
      recallAt3: ranked.filter((rank) => rank <= 3).length / ranked.length,
      recallAt5: ranked.filter((rank) => rank <= 5).length / ranked.length,
      latencyMs: {
        median: Number(percentile(latencies, 0.5).toFixed(3)),
        p95: Number(percentile(latencies, 0.95).toFixed(3)),
        min: Number(Math.min(...latencies).toFixed(3)),
        max: Number(Math.max(...latencies).toFixed(3)),
      },
      queries: results,
    },
    dashboardApi,
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${String((error as Error)?.stack || error)}\n`);
  process.exitCode = 1;
});
