/** Browser/Node shared operation lifetime. No platform-specific imports. */
export interface OperationOptions { signal?: AbortSignal; timeoutMs?: number }
export const DEFAULT_OPERATION_TIMEOUT_MS = 300_000;
export const MAX_OPERATION_TIMEOUT_MS = 600_000;

export function operationError(timeout = false): Error & { status: number } {
  return Object.assign(new Error(timeout ? "请求超时" : "请求已取消"), {
    name: timeout ? "TimeoutError" : "AbortError", status: timeout ? 504 : 499,
  });
}
export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw signal.reason instanceof Error ? signal.reason : operationError();
  const lifetime = signal ? operationLifetimes.get(signal) : undefined;
  if (lifetime && performance.now() >= lifetime.deadline) throw operationError(true);
}

const operationLifetimes = new WeakMap<AbortSignal, { requestID: string; deadline: number }>();

export class OperationScope {
  readonly controller = new AbortController();
  readonly signal = this.controller.signal;
  readonly requestID: string;
  private readonly deadline: number;
  private readonly timer: ReturnType<typeof setTimeout>;
  private readonly parent: AbortSignal | undefined;
  private readonly onParentAbort = () => this.controller.abort(
    this.parent?.reason?.name === "TimeoutError" ? operationError(true) : operationError());

  constructor(options: OperationOptions = {}) {
    const duration = options.timeoutMs ?? DEFAULT_OPERATION_TIMEOUT_MS;
    if (!Number.isInteger(duration) || duration < 1 || duration > MAX_OPERATION_TIMEOUT_MS) {
      throw Object.assign(new Error(`timeoutMs 必须为 1–${MAX_OPERATION_TIMEOUT_MS} 的整数`), { status: 400 });
    }
    const parent = options.signal ? operationLifetimes.get(options.signal) : undefined;
    this.requestID = parent?.requestID ?? crypto.randomUUID();
    this.deadline = Math.min(performance.now() + duration, parent?.deadline ?? Infinity);
    operationLifetimes.set(this.signal, { requestID: this.requestID, deadline: this.deadline });
    this.parent = options.signal;
    this.timer = setTimeout(() => this.controller.abort(operationError(true)), Math.max(1, Math.ceil(this.deadline - performance.now())));
    if (performance.now() >= this.deadline) this.controller.abort(operationError(true));
    this.parent?.addEventListener("abort", this.onParentAbort, { once: true });
    if (this.parent?.aborted) this.onParentAbort();
  }
  get remainingMS(): number {
    if (performance.now() >= this.deadline && !this.signal.aborted) this.controller.abort(operationError(true));
    throwIfAborted(this.signal);
    return Math.max(1, Math.ceil(this.deadline - performance.now()));
  }
  dispose(): void {
    clearTimeout(this.timer);
    this.parent?.removeEventListener("abort", this.onParentAbort);
  }
}

export function waitForRetry(milliseconds: number, signal: AbortSignal): Promise<void> {
  throwIfAborted(signal);
  return new Promise((resolve, reject) => {
    const finish = () => { signal.removeEventListener("abort", abort); resolve(); };
    const timer = setTimeout(finish, milliseconds);
    const abort = () => { clearTimeout(timer); signal.removeEventListener("abort", abort); reject(signal.reason); };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
  });
}

export async function withOperation<T>(options: OperationOptions, action: (scope: OperationScope) => Promise<T>): Promise<T> {
  const scope = new OperationScope(options);
  try {
    void scope.remainingMS;
    const result = await action(scope);
    void scope.remainingMS;
    return result;
  } catch (error) {
    void scope.remainingMS;
    throw error;
  } finally {
    // Stop siblings still running after an early failure; completed calls have
    // already detached their listeners. Disposal alone would leave them alive.
    scope.controller.abort(operationError());
    scope.dispose();
  }
}
