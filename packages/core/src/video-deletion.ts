import { createHash } from "node:crypto";
import type { SourceSite } from "@indexed/contracts";

export interface VideoDeletionTarget {
  profileId: string;
  storageProvider: "local" | "aliyun";
  embeddingModel: string;
  embeddingSpace: string;
  sourceSite: SourceSite;
  videoId: string;
  storage: { region: string; accountId: string; bucket: string; localPath: string; visualIndex: string; transcriptIndex: string };
}

export interface VideoDeletionKeys { visualKeys: string[]; transcriptKeys: string[] }
export interface VideoDeletionRepository {
  listKeys(): Promise<VideoDeletionKeys>;
  deleteKeys(indexName: string, keys: string[]): Promise<unknown>;
}

export class VideoDeletionError extends Error {
  constructor(public readonly status: number, message: string) { super(message); }
}

function validateTarget(target: VideoDeletionTarget): void {
  if (!["local", "aliyun"].includes(target.storageProvider) || !["youtube", "bilibili"].includes(target.sourceSite) || !target.videoId?.trim()
    || !target.profileId || !target.embeddingModel || !target.embeddingSpace
    || !target.storage.visualIndex || !target.storage.transcriptIndex) {
    throw new VideoDeletionError(400, "删除目标必须包含确切的 profile、空间、索引、来源站点和视频 ID");
  }
  if (target.storageProvider === "aliyun" && (!target.storage.region || !target.storage.accountId || !target.storage.bucket)) {
    throw new VideoDeletionError(400, "云端删除目标缺少 region、accountId 或 bucket");
  }
}

function normalizeKeys(keys: VideoDeletionKeys): VideoDeletionKeys {
  return { visualKeys: [...new Set(keys.visualKeys.filter(Boolean))].sort(),
    transcriptKeys: [...new Set(keys.transcriptKeys.filter(Boolean))].sort() };
}

function deletionPlan(target: VideoDeletionTarget, keys: VideoDeletionKeys) {
  return { schema: 1 as const, target, visualCount: keys.visualKeys.length, transcriptCount: keys.transcriptKeys.length,
    token: createHash("sha256").update(JSON.stringify({ schema: 1, target, keys })).digest("hex") };
}

export async function prepareVideoDeletionWithRepository(target: VideoDeletionTarget, repository: VideoDeletionRepository) {
  validateTarget(target);
  return deletionPlan(target, normalizeKeys(await repository.listKeys()));
}

export async function deleteVideoWithRepository(target: VideoDeletionTarget, repository: VideoDeletionRepository, confirmation?: unknown) {
  validateTarget(target);
  const supplied = confirmation && typeof confirmation === "object" ? confirmation as Record<string, unknown> : {};
  const needsConfirmation = target.storageProvider === "aliyun" || confirmation !== undefined;
  if (needsConfirmation && (supplied.confirmed !== true || supplied.schema !== 1 || typeof supplied.token !== "string")) {
    throw new VideoDeletionError(428, "云端删除需要针对确切目标的显式确认；请先获取 deletion-preview，再提交 confirmed、schema 和 token");
  }
  const keys = normalizeKeys(await repository.listKeys());
  const plan = deletionPlan(target, keys);
  if (needsConfirmation && supplied.token !== plan.token) {
    throw new VideoDeletionError(409, "删除目标或索引记录已经变化，请重新预览并确认");
  }
  for (const [indexName, selected] of [[target.storage.visualIndex, keys.visualKeys],
    [target.storage.transcriptIndex, keys.transcriptKeys]] as const) {
    for (let offset = 0; offset < selected.length; offset += 500) await repository.deleteKeys(indexName, selected.slice(offset, offset + 500));
  }
  return { ok: true, sourceSite: target.sourceSite, videoId: target.videoId,
    visualDeleted: keys.visualKeys.length, transcriptDeleted: keys.transcriptKeys.length };
}
