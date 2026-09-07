// @ts-nocheck -- compatibility port; provider response types will replace permissive payloads next.
import { withOperation, throwIfAborted } from "./operation.js";

function document(item) {
  const content = [];
  if (String(item.preview || "").startsWith("data:image/")) {
    content.push({ type: "image_url", image_url: { url: item.preview } });
  }
  const text = [item.title, item.channel, item.transcript, item.sourceSite]
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .join("\n");
  content.push({ type: "text", text: text || "video segment" });
  return { content };
}

export async function rerankerHealth(profile) {
  if (!profile.reranker?.enabled) return { ok: true, enabled: false };
  const baseUrl = String(profile.reranker.baseUrl || "").replace(/\/+$/, "");
  if (!baseUrl) return { ok: false, enabled: true, error: "精排服务地址为空" };
  const response = await fetch(`${baseUrl}/v1/models`);
  if (!response.ok) return { ok: false, enabled: true, error: `HTTP ${response.status}` };
  return { ok: true, enabled: true, model: profile.reranker.model };
}

export async function rerank(query, items, profile, options = {}) {
  return withOperation(options, scope => rerankWithin(query, items, profile, scope.signal));
}
async function rerankWithin(query, items, profile, signal) {
  if (!profile.reranker?.enabled || !items.length) return { items, applied: false, warning: "" };
  const baseUrl = String(profile.reranker.baseUrl || "").replace(/\/+$/, "");
  const count = Math.min(Number(profile.reranker.candidates || 20), items.length);
  const head = items.slice(0, count);
  try {
    const response = await fetch(`${baseUrl}/v1/rerank`, {
      method: "POST",
      signal,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: profile.reranker.model,
        query,
        documents: head.map(document),
        top_n: head.length,
        instruction: "Judge how well each candidate video segment matches the complete user request using its visible frame, title, subtitle and source context.",
      }),
    });
    const text = await response.text();
    if (!response.ok) throw new Error(`精排服务返回 ${response.status}: ${text.slice(0, 240)}`);
    const payload = JSON.parse(text);
    const ranked = [];
    const used = new Set();
    for (const row of payload.results || []) {
      const index = Number(row.index);
      if (!Number.isInteger(index) || index < 0 || index >= head.length) continue;
      ranked.push({
        ...head[index],
        vectorScore: head[index].score,
        score: Number(row.relevance_score || 0),
        scoreType: "rerank",
      });
      used.add(index);
    }
    head.forEach((item, index) => { if (!used.has(index)) ranked.push(item); });
    return { items: [...ranked, ...items.slice(count)], applied: true, warning: "" };
  } catch (error) {
    throwIfAborted(signal);
    return { items, applied: false, warning: String(error?.message || error) };
  }
}
