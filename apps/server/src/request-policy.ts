import type { IncomingMessage } from "node:http";
import { isIP } from "node:net";
import { throwIfAborted } from "@indexed/clients/operation";

const JSON_LIMIT = 2 * 1024 * 1024;
const MEDIA_LIMIT = 64 * 1024 * 1024;

/** This preview serves only its owning machine. Resolve localhost without DNS. */
export function localListenHost(value: string): string {
  const host = value.trim().toLowerCase();
  if (host === "localhost" || host === "localhost.") return "127.0.0.1";
  if (host === "::1" || host === "[::1]") return "::1";
  if (isIP(host) === 4 && host.startsWith("127.")) return host;
  throw new Error("当前版本仅支持本机使用；监听地址请设为 127.0.0.1、localhost 或 ::1");
}

export class RequestPolicyError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "RequestPolicyError";
  }
}

export interface RequestSourcePolicy {
  /** The configured listener name, in addition to its actual socket address. */
  host: string;
  /** Exact origins of explicitly configured or bundled extensions. */
  extensionOrigins: readonly string[];
}

function normalizedHost(host: string): string {
  return host.replace(/^::ffff:/, "").replace(/^\[|\]$/g, "").toLowerCase();
}

function requestAuthority(request: IncomingMessage): URL {
  const host = request.headers.host;
  if (!host || /[\s/\\?#@]/.test(host)) {
    throw new RequestPolicyError(400, "请求缺少有效的 Host");
  }
  try {
    return new URL(`http://${host}`);
  } catch {
    throw new RequestPolicyError(400, "请求缺少有效的 Host");
  }
}

/** This browser-origin boundary is independent of API session authentication. */
export function assertRequestSource(request: IncomingMessage, policy: RequestSourcePolicy): void {
  const target = requestAuthority(request);
  const local = normalizedHost(request.socket.localAddress || "");
  const hosts = new Set([local]);
  const configuredHost = normalizedHost(policy.host);
  if (!["0.0.0.0", "::"].includes(configuredHost)) hosts.add(configuredHost);
  if (local === "::1" || /^127\./.test(local)) {
    hosts.add("localhost");
    hosts.add("127.0.0.1");
    hosts.add("::1");
  }
  if (!hosts.has(normalizedHost(target.hostname)) || Number(target.port || 80) !== request.socket.localPort) {
    throw new RequestPolicyError(403, "请求 Host 与 Indexed 服务地址不匹配");
  }

  const origin = request.headers.origin;
  if (origin && policy.extensionOrigins.includes(origin)) return;
  if (origin) {
    if (origin !== target.origin) throw new RequestPolicyError(403, "不允许此来源访问 Indexed");
    return;
  }
  // Older browsers may supply Referer without Origin or Fetch Metadata.
  if (request.headers.referer) {
    let source: URL;
    try { source = new URL(request.headers.referer); }
    catch { throw new RequestPolicyError(403, "请求来源无效"); }
    if (source.origin !== target.origin) throw new RequestPolicyError(403, "不允许此来源访问 Indexed");
  }
  if (request.headers["sec-fetch-site"] === "cross-site") {
    throw new RequestPolicyError(403, "不允许跨站请求访问 Indexed");
  }
}

export function requestBodyLimit(request: IncomingMessage): number {
  const pathname = new URL(request.url || "/", "http://localhost").pathname;
  return pathname.startsWith("/api/ingest/")
    || pathname === "/api/search/image"
    || pathname === "/api/embedding/v1/embeddings" ? MEDIA_LIMIT : JSON_LIMIT;
}

export function assertRequestBodyHeaders(request: IncomingMessage): void {
  const length = Number(request.headers["content-length"] || 0);
  if (length > requestBodyLimit(request)) throw new RequestPolicyError(413, "请求体超过此接口大小限制");
  const hasBody = length > 0 || request.headers["transfer-encoding"] !== undefined;
  if (hasBody && !/^application\/json(?:\s*;|$)/i.test(request.headers["content-type"] || "")) {
    throw new RequestPolicyError(415, "请求体只接受 application/json");
  }
}

export async function readJSONBody(request: IncomingMessage, signal?: AbortSignal): Promise<Record<string, unknown>> {
  assertRequestBodyHeaders(request);
  const limit = requestBodyLimit(request);
  const chunks: Buffer[] = [];
  let bytes = 0;
  throwIfAborted(signal);
  // Pause on failure so a structured error can be written before closing the
  // connection. Always detach listeners, including when a deadline interrupts upload.
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      request.off("data", onData); request.off("end", onEnd); request.off("error", onError);
      request.off("aborted", onAborted); signal?.removeEventListener("abort", onCancel);
    };
    const fail = (error: unknown) => { cleanup(); request.pause(); reject(error); };
    const onData = (value: Buffer | Uint8Array) => {
      const chunk = Buffer.isBuffer(value) ? value : Buffer.from(value);
      bytes += chunk.length;
      if (bytes > limit) { fail(new RequestPolicyError(413, "请求体超过此接口大小限制")); return; }
      chunks.push(chunk);
    };
    const onEnd = () => { cleanup(); resolve(); };
    const onError = (error: Error) => fail(error);
    const onAborted = () => fail(new RequestPolicyError(499, "请求已取消"));
    const onCancel = () => fail(signal?.reason);
    request.on("data", onData); request.once("end", onEnd); request.once("error", onError);
    request.once("aborted", onAborted); signal?.addEventListener("abort", onCancel, { once: true });
    if (signal?.aborted) onCancel();
    else if (request.aborted) onAborted();
    else if (request.readableEnded) onEnd();
    else request.resume();
  });
  if (!bytes) return {};
  let value: unknown;
  try { value = JSON.parse(Buffer.concat(chunks, bytes).toString("utf8")); }
  catch { throw new RequestPolicyError(400, "请求体不是有效的 JSON"); }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new RequestPolicyError(400, "请求体必须是 JSON 对象");
  }
  return value as Record<string, unknown>;
}
