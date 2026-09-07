import { withOperation, type OperationOptions } from "@indexed/clients/operation";
import fs from "node:fs";
import path from "node:path";
import { embedDocument, embedTextWithInstruction, embeddingCorpusInputVersion, metadataFilter, scoreFromDistance, uuid5Url } from "@indexed/clients/direct-cloud";
import { listLocalVectors, putLocalVectors, queryLocalVectors } from "@indexed/clients/local-vectors";
import { activeProfile, directConfig, loadConfig } from "@indexed/config";

const TEXT_EXTENSIONS = new Set([".md", ".mdx", ".txt", ".json", ".jsonl", ".csv", ".srt", ".vtt", ".ts", ".tsx", ".js", ".jsx", ".py", ".html", ".css"]);
const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".avif"]);
const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".mkv", ".webm", ".m4v"]);
const SKIPPED_DIRECTORIES = new Set([".git", "node_modules", ".next", "dist", "build"]);
const ASSET_QUERY_INSTRUCTION =
  "Retrieve personal creative assets relevant to the user's request across videos, images, documents, audio, and project files.";

function context(config = loadConfig()) {
  const profile = activeProfile(config);
  const direct = directConfig(profile);
  if (direct.storageProvider !== "local") throw new Error("本地资料库需要使用本地 zvec 配置档案");
  return { config, profile, direct };
}

function kind(extension: string) {
  if (TEXT_EXTENSIONS.has(extension)) return "document";
  if (IMAGE_EXTENSIONS.has(extension)) return "image";
  if (VIDEO_EXTENSIONS.has(extension)) return "video";
  return "file";
}

function walk(root: string, limit: number) {
  const output: string[] = [];
  const visit = (directory: string) => {
    if (output.length >= limit) return;
    let entries: fs.Dirent[] = [];
    try { entries = fs.readdirSync(directory, { withFileTypes: true }); } catch { return; }
    for (const entry of entries) {
      if (output.length >= limit) break;
      if (entry.name.startsWith(".") || SKIPPED_DIRECTORIES.has(entry.name)) continue;
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(target);
      else if (entry.isFile()) output.push(target);
    }
  };
  visit(root);
  return output;
}

function readableText(file: string, extension: string, size: number) {
  if (!TEXT_EXTENSIONS.has(extension) || size > 2 * 1024 * 1024) return "";
  try { return fs.readFileSync(file, "utf8").slice(0, 64 * 1024).replace(/\0/g, " "); } catch { return ""; }
}

function fileResult(row: { key?: string; distance?: number; metadata?: Record<string, unknown> }) {
  const metadata = row.metadata || {};
  const sourcePath = String(metadata.file_path || metadata.source_path || metadata.source_uri || "");
  const title = String(metadata.file_name || metadata.title || path.basename(sourcePath) || row.key || "Untitled");
  return {
    id: String(row.key || ""),
    assetId: String(metadata.asset_id || row.key || ""),
    score: row.distance === undefined ? undefined : scoreFromDistance(row.distance),
    path: sourcePath,
    name: title,
    kind: String(metadata.file_kind || metadata.modality || metadata.kind || "file"),
    extension: String(metadata.extension || path.extname(sourcePath)),
    size: Number(metadata.file_size || metadata.size_bytes || 0),
    modifiedAt: Number(metadata.modified_at_ms || 0),
    root: String(metadata.library_root || metadata.project_name || ""),
    projectId: String(metadata.project_id || ""),
    projectName: String(metadata.project_name || ""),
    role: String(metadata.role || ""),
    preview: String(metadata.text_preview || metadata.content_excerpt || ""),
  };
}

function dedupeAssets(rows: Array<ReturnType<typeof fileResult>>, limit: number) {
  const unique = new Map<string, ReturnType<typeof fileResult>>();
  for (const row of rows) {
    const identity = row.assetId || row.path || row.id;
    if (!unique.has(identity)) unique.set(identity, row);
    if (unique.size >= limit) break;
  }
  return [...unique.values()];
}

export async function scanLocalLibrary({ limit = 500 } = {}, config = loadConfig()) {
  const { profile, direct } = context(config);
  const roots = (config.library?.roots || []).map((value: unknown) => path.resolve(String(value))).filter((value: string) => fs.existsSync(value));
  if (!roots.length) throw new Error("请先添加本地资料库目录");
  const files = roots.flatMap((root: string) => walk(root, Math.max(1, limit - 1))).slice(0, limit);
  let indexed = 0;
  for (const file of files) {
    const stat = fs.statSync(file);
    const extension = path.extname(file).toLowerCase();
    const text = readableText(file, extension, stat.size);
    const relative = roots.find((root: string) => file.startsWith(`${root}${path.sep}`)) || path.dirname(file);
    const description = [
      path.basename(file),
      path.relative(relative, file),
      kind(extension),
      text,
    ].filter(Boolean).join("\n").slice(0, 64 * 1024);
    const vector = await embedDocument(description, direct);
    const key = await uuid5Url(`local-file:${file}:${direct.embeddingSpace}`);
    await putLocalVectors(direct.documentIndex, [{
      key,
      data: { float32: vector },
      metadata: {
        record_type: "local_file",
        file_path: file,
        file_name: path.basename(file),
        file_kind: kind(extension),
        extension,
        file_size: stat.size,
        modified_at_ms: stat.mtimeMs,
        library_root: relative,
        text_preview: text.slice(0, 280),
        embedding_model: direct.embeddingModel,
        embedding_space: direct.embeddingSpace,
        embedding_input_version: embeddingCorpusInputVersion(direct, "text"),
        indexed_at_ms: Date.now(),
      },
    }], direct);
    indexed += 1;
  }
  return { ok: true, roots, discovered: files.length, indexed, limit };
}

export async function listLocalFiles({ limit = 200 } = {}, config = loadConfig()) {
  const { direct } = context(config);
  const rows = await listLocalVectors(direct.documentIndex, direct, {
    filter: metadataFilter({ embedding_model: direct.embeddingModel, embedding_space: direct.embeddingSpace }),
    maxItems: Math.max(Number(limit || 200) * 4, Number(limit || 200)),
  });
  const files = dedupeAssets(
    rows.map(fileResult).sort((left, right) => right.modifiedAt - left.modifiedAt),
    Number(limit || 200),
  );
  return { count: files.length, roots: config.library?.roots || [], files };
}

export async function searchLocalFiles(query: string, options: OperationOptions & { limit?: number } = {}, config = loadConfig()) {
  return withOperation(options, scope => searchLocalFilesWithin(query, { ...options, signal: scope.signal }, config));
}
async function searchLocalFilesWithin(query: string, { limit = 30, signal }: OperationOptions & { limit?: number }, config = loadConfig()) {
  const text = String(query || "").trim();
  if (!text) throw new Error("搜索文字不能为空");
  const { direct } = context(config);
  const vector = await embedTextWithInstruction(text, ASSET_QUERY_INSTRUCTION, direct, { signal });
  const page = await queryLocalVectors(direct.documentIndex, vector, direct, {
    limit: Math.min(500, Math.max(Number(limit || 30) * 4, Number(limit || 30))),
    filter: metadataFilter({ embedding_model: direct.embeddingModel, embedding_space: direct.embeddingSpace }),
  });
  return { query: text, files: dedupeAssets(page.vectors.map(fileResult), Number(limit || 30)) };
}
