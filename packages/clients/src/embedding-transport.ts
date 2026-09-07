import { OperationScope, throwIfAborted, waitForRetry, type OperationOptions } from "./operation.js";

export interface EmbeddingTransportOptions extends OperationOptions {
  /** Only an explicitly managed Indexed helper uses the private cancellation protocol. */
  native?: boolean;
}
const retryable = new Set([429, 503]);
function retryDelay(response: Response, attempt: number): number {
  const header = response.headers.get("retry-after")?.trim();
  if (header) {
    const seconds = Number(header);
    if (Number.isFinite(seconds) && seconds >= 0) return Math.min(5_000, seconds * 1_000);
    const date = Date.parse(header);
    if (Number.isFinite(date)) return Math.min(5_000, Math.max(0, date - Date.now()));
  }
  return Math.min(2_000, 100 * 2 ** attempt);
}

/** Retries share one deadline; every native attempt has a distinct cancellation ID. */
export async function postEmbedding(baseURL: string, payload: Record<string, unknown>, headers: Record<string, string>,
                                    options: EmbeddingTransportOptions = {}): Promise<{ response: Response; text: string }> {
  const operation = new OperationScope(options);
  try {
    for (let attempt = 0; attempt < 6; attempt++) {
      throwIfAborted(operation.signal);
      const requestID = options.native ? `${operation.requestID}.${crypto.randomUUID()}` : undefined;
      let cancellation: Promise<void> | undefined;
      const cancel = () => {
        if (!requestID || cancellation) return;
        cancellation = (async () => {
          const scope = new OperationScope({ timeoutMs: 2_000 });
          try {
            const reply = await fetch(`${baseURL}/v1/requests/${requestID}`, {
              method: "DELETE", headers, signal: scope.signal,
            });
            await reply.body?.cancel();
          } catch { /* Best effort: helper disconnect/deadline checks are independent. */ }
          finally { scope.dispose(); }
        })();
      };
      operation.signal.addEventListener("abort", cancel, { once: true });
      let result: { response: Response; text: string } | undefined;
      let failure: unknown;
      try {
        const response = await fetch(`${baseURL}/v1/embeddings`, {
          method: "POST", headers: { ...headers,
            ...(requestID ? { "x-indexed-timeout-ms": String(operation.remainingMS) } : {}) },
          body: JSON.stringify(requestID ? { ...payload, request_id: requestID } : payload),
          signal: operation.signal,
        });
        result = { response, text: await response.text() };
        throwIfAborted(operation.signal);
      } catch (error) {
        failure = error;
        cancel();
      } finally {
        operation.signal.removeEventListener("abort", cancel);
        if (cancellation) await cancellation;
      }
      throwIfAborted(operation.signal);
      if (result) {
        if (!retryable.has(result.response.status) || attempt === 5) return result;
        await waitForRetry(retryDelay(result.response, attempt), operation.signal);
      } else {
        if (attempt === 5) throw new Error(`Embedding 服务连接失败：${failure instanceof Error ? failure.message : String(failure)}`);
        await waitForRetry(Math.min(2_000, 100 * 2 ** attempt), operation.signal);
      }
    }
    throw new Error("Embedding 服务重试次数已用尽");
  } finally { operation.dispose(); }
}
