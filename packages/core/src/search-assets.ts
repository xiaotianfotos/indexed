import { withOperation, throwIfAborted } from "@indexed/clients/operation";
import { loadConfig } from "@indexed/config";
import {
  searchAssets,
  type AssetHit,
  type SearchAssetsResult,
  type SearchOptions,
} from "./asset-library.js";
import { search as searchVideoMemory } from "./library.js";

type WebMemoryHit = {
  id?: string;
  score?: number;
  videoId?: string;
  sourceSite?: string;
  title?: string;
  channel?: string;
  sourceUrl?: string;
  startTime?: number;
  endTime?: number;
  transcript?: string;
  preview?: string;
  embeddingModel?: string;
};

type WebMemorySearch = {
  embeddingSpace?: string;
  reranked?: boolean;
  visual?: WebMemoryHit[];
  transcript?: WebMemoryHit[];
};

export interface SearchAllAssetsResult extends Omit<SearchAssetsResult, "reranked"> {
  reranked: boolean;
  sources: { local: number; web: number };
}

function finite(value: unknown, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function sourceAt(hit: WebMemoryHit, seconds: number) {
  const site = String(hit.sourceSite || "youtube").toLowerCase();
  const videoId = String(hit.videoId || "");
  const fallback = site === "bilibili"
    ? `https://www.bilibili.com/video/${encodeURIComponent(videoId)}/`
    : `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}`;
  try {
    const target = new URL(String(hit.sourceUrl || fallback));
    if (!new Set(["http:", "https:"]).has(target.protocol)) return "";
    const position = Math.max(0, Math.floor(seconds));
    target.searchParams.set("t", site === "bilibili" ? String(position) : `${position}s`);
    return target.toString();
  } catch {
    return "";
  }
}

function webAsset(hit: WebMemoryHit, modality: "visual" | "transcript"): AssetHit | null {
  const startSeconds = Math.max(0, finite(hit.startTime));
  const openUrl = sourceAt(hit, startSeconds);
  if (!openUrl) return null;
  const sourceSite = String(hit.sourceSite || "youtube").toLowerCase();
  const videoId = String(hit.videoId || "");
  const preview = String(hit.preview || (sourceSite === "youtube" && videoId
    ? `https://i.ytimg.com/vi/${encodeURIComponent(videoId)}/hqdefault.jpg`
    : ""));
  return {
    id: `web:${modality}:${String(hit.id || `${sourceSite}:${videoId}:${startSeconds}`)}`,
    source: "web",
    path: "",
    openUrl,
    name: String(hit.title || "未命名网页视频"),
    kind: "video",
    libraryPath: "",
    libraryName: String(hit.channel || (sourceSite === "bilibili" ? "哔哩哔哩" : "YouTube")),
    size: 0,
    modifiedAt: 0,
    score: finite(hit.score),
    segmentIndex: 0,
    startSeconds,
    endSeconds: Math.max(startSeconds, finite(hit.endTime, startSeconds)),
    duration: 0,
    embeddingBasis: String(hit.embeddingModel || modality),
    previewUrl: preview,
    sidecarText: modality === "transcript" ? String(hit.transcript || "") : "",
    legacy: false,
  };
}

function webAssets(result: WebMemorySearch) {
  const unique = new Map<string, AssetHit>();
  for (const [modality, items] of [
    ["visual", result.visual || []],
    ["transcript", result.transcript || []],
  ] as const) {
    for (const raw of items) {
      const item = webAsset(raw, modality);
      if (!item) continue;
      const identity = `${raw.sourceSite || "youtube"}:${raw.videoId || ""}:${Math.floor(item.startSeconds)}`;
      const current = unique.get(identity);
      if (!current) {
        unique.set(identity, item);
        continue;
      }
      if (item.score > current.score) {
        unique.set(identity, { ...item, sidecarText: item.sidecarText || current.sidecarText });
      } else if (!current.sidecarText && item.sidecarText) {
        unique.set(identity, { ...current, sidecarText: item.sidecarText });
      }
    }
  }
  return [...unique.values()].sort((left, right) => right.score - left.score);
}

/**
 * Search the local file library and the web-video memory written by the Chrome
 * extension. Both are returned as assets; only their opening locator differs.
 */
export async function searchAllAssets(query: string, options: SearchOptions = {}): Promise<SearchAllAssetsResult> {
  return withOperation(options, scope => searchAllAssetsWithin(query, { ...options, signal: scope.signal }));
}

async function searchAllAssetsWithin(query: string, options: SearchOptions): Promise<SearchAllAssetsResult> {
  const text = String(query || "").trim();
  if (!text) throw new Error("搜索文字不能为空");
  const limit = Math.max(1, Math.min(200, Math.floor(finite(options.limit, 30))));
  const config = options.config || loadConfig();
  const kind = String(options.kind || "").trim().toLowerCase();
  if (kind && !["video", "image", "document"].includes(kind)) throw new Error(`不支持的素材类型：${kind}`);
  const includeWeb = !kind || kind === "video";
  if (!includeWeb) {
    const local = await searchAssets(text, { ...options, kind, config, limit });
    return { ...local, sources: { local: local.hits.length, web: 0 } };
  }
  const [localResult, memoryResult] = await Promise.allSettled([
    searchAssets(text, { ...options, kind, config, limit }),
    searchVideoMemory(text, { limit: Math.min(200, Math.max(limit * 2, 30)), signal: options.signal }, config),
  ]);
  throwIfAborted(options.signal);
  if (localResult.status === "rejected" && memoryResult.status === "rejected") {
    throw new Error(String(localResult.reason?.message || memoryResult.reason?.message || "素材检索失败"));
  }
  const local = localResult.status === "fulfilled"
    ? localResult.value
    : { query: text, embeddingSpace: "", reranked: false as const, hits: [], hiddenMissing: 0 };
  const memory = memoryResult.status === "fulfilled" ? memoryResult.value as WebMemorySearch : {};
  const web = webAssets(memory);
  const hits = [...local.hits, ...web]
    .sort((left, right) => right.score - left.score)
    .slice(0, limit);
  return {
    query: text,
    embeddingSpace: local.embeddingSpace || String(memory.embeddingSpace || ""),
    reranked: Boolean(memory.reranked),
    hits,
    hiddenMissing: local.hiddenMissing,
    sources: { local: local.hits.length, web: web.length },
  };
}
