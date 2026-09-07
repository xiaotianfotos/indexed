export interface ScanPresentationInput {
  state?: string;
  startedAt?: number;
  finishedAt?: number;
  total?: number;
  done?: number;
  indexed?: number;
}

/** Display successful throughput using wall time, never summed worker latency. */
export function scanPresentation(job: ScanPresentationInput, now = Date.now()) {
  const count = (value: number | undefined) => Number.isFinite(value) ? Math.max(0, value || 0) : 0;
  const total = count(job.total);
  const done = Math.min(total, count(job.done));
  const indexed = Math.min(done, count(job.indexed));
  const busy = job.state === "running" || job.state === "queued";
  const startedAt = count(job.startedAt);
  const end = count(job.finishedAt) || (busy ? now : startedAt);
  const elapsedMs = startedAt ? Math.max(0, end - startedAt) : 0;
  return {
    busy, total, done, indexed, elapsedMs,
    percent: total ? Math.round(done / total * 100) : 0,
    filesPerSecond: elapsedMs > 0 && indexed > 0 ? indexed * 1000 / elapsedMs : null,
  };
}

export interface ComparisonRun {
  key: string;
  mode: string;
  startedAt: number;
  elapsedMs: number;
  count: number;
}

export function recordComparison(runs: ComparisonRun[], job: ScanPresentationInput & {
  benchmarkKey?: string; executionMode?: string; failed?: number; skipped?: number;
}): ComparisonRun[] {
  const view = scanPresentation(job);
  if (!job.benchmarkKey || !["a", "b", "c", "d"].includes(job.executionMode || "")
    || job.state !== "done" || !view.total || view.indexed !== view.total
    || job.failed || job.skipped || view.elapsedMs <= 0 || !job.startedAt) return runs;
  if (runs.some((run) => run.key === job.benchmarkKey && run.startedAt === job.startedAt)) return runs;
  return [...runs, { key: job.benchmarkKey, mode: job.executionMode!, startedAt: job.startedAt,
    elapsedMs: view.elapsedMs, count: view.total }].slice(-80);
}

export function compareRuns(runs: ComparisonRun[], key: string) {
  const rows = ["a", "b", "c", "d"].flatMap((mode) => {
    const samples = runs.filter((run) => run.key === key && run.mode === mode
      && Number.isFinite(run.elapsedMs) && run.elapsedMs > 0).slice(-3);
    if (!samples.length) return [];
    const durations = samples.map((run) => run.elapsedMs).sort((a, b) => a - b);
    const middle = Math.floor(durations.length / 2);
    const milliseconds = durations.length % 2 ? durations[middle]! : (durations[middle - 1]! + durations[middle]!) / 2;
    return [{ mode, milliseconds, samples: samples.length }];
  });
  const baseline = rows.find((row) => row.mode === "a")?.milliseconds;
  return rows.map((row) => ({ ...row, speedup: baseline ? baseline / row.milliseconds : null }));
}
