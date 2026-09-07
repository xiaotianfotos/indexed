export interface QueueStatusLike {
  queued_total?: number;
  errors?: number;
  paused?: boolean;
  processing?: boolean;
  last_error?: string;
}

export function queueItemReadyNow<T extends Record<string, unknown>>(item: T): T {
  return {
    ...item,
    attempts: 0,
    terminal: false,
    lastError: "",
    lastErrorAt: 0,
    nextAttempt: 0,
  };
}

export function queueStatusPresentation(status: QueueStatusLike) {
  const queued = Math.max(0, Number(status.queued_total || 0));
  const errors = Math.max(0, Number(status.errors || 0));
  const detail = String(status.last_error || "").trim();
  if (status.paused) {
    return { visible: true, tone: "warning", text: `队列已暂停 · ${queued} 项等待处理`, title: detail };
  }
  if (errors) {
    return {
      visible: true,
      tone: "error",
      text: `队列处理失败 · ${errors} 项等待重试${detail ? ` · ${detail}` : ""}`,
      title: detail,
    };
  }
  if (status.processing) {
    return { visible: true, tone: "active", text: `正在处理队列 · ${queued} 项剩余`, title: "" };
  }
  if (queued) {
    return { visible: true, tone: "warning", text: `队列等待重试 · ${queued} 项`, title: detail };
  }
  return { visible: false, tone: "idle", text: "", title: "" };
}
