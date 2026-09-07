import fs from "node:fs";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { activeProfile, ingestPerformanceHistoryPath, loadConfig } from "@indexed/config";
import { scanLibrary, type ScanOptions, type ScanResult, type AssetJob } from "@indexed/core";

type Config = ReturnType<typeof loadConfig>;
export interface IngestTestRow {
  mode: string;
  state: "pending" | "loading" | "warming" | "running" | "done" | "error";
  loadMs?: number;
  warmupMs?: number;
  elapsedMs?: number;
  filesPerSecond?: number;
  modelMs?: number;
  visionMs?: number;
  languageMs?: number;
  files?: number;
  error?: string;
  samples: ScanResult[];
  executionEvidence?: Array<{ before: unknown; after: unknown }>;
}
export interface IngestTestReport {
  schema: 1;
  protocol?: "no-warmup-cooldown-v1";
  state: "idle" | "running" | "restoring" | "done" | "cancelled" | "error";
  root: string;
  limit: number;
  repeats: number;
  mode: string;
  phase: string;
  rows: IngestTestRow[];
  progress?: AssetJob;
  startedAt?: number;
  finishedAt?: number;
  error?: string;
}
interface Dependencies {
  runtime: { reconcile(config: Config): Promise<unknown>; stop(): Promise<unknown> };
  scan?: (root: string, options: ScanOptions) => Promise<ScanResult>;
  config?: () => Config;
  storePath?: string;
  onStart?: () => void;
  onFinish?: () => void;
  canStart?: () => void;
  cooldown?: (signal: AbortSignal) => Promise<void>;
  executionSnapshot?: () => Promise<unknown>;
}
const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
};

/** Owns the model temporarily, never persists mode changes or writes vectors. */
export class IngestTestController {
  report: IngestTestReport = { schema: 1, state: "idle", root: "", limit: 10, repeats: 1, mode: "", phase: "", rows: [] };
  private abort?: AbortController;
  private running?: Promise<void>;
  private shuttingDown = false;
  private readonly storePath: string;

  constructor(private readonly deps: Dependencies) {
    this.storePath = deps.storePath || path.join(path.dirname(ingestPerformanceHistoryPath()), "ingest-test.json");
    try {
      const saved = JSON.parse(fs.readFileSync(this.storePath, "utf8"));
      if (saved.schema === 1 && Array.isArray(saved.rows) && typeof saved.root === "string") {
        this.report = saved;
        if (this.busy) this.report = { ...saved, state: "cancelled", phase: "上次测试被中断" };
      }
    } catch { /* First run or an unreadable previous report. */ }
  }

  get busy() { return this.report.state === "running" || this.report.state === "restoring"; }

  start(root: string, limit = 10, repeats = 1) {
    if (this.busy) throw new Error("入库测试正在运行");
    if (!root.trim() || !fs.statSync(path.resolve(root)).isDirectory()) throw new Error("请选择有效的测试素材目录");
    const config = (this.deps.config || loadConfig)();
    if (activeProfile(config).embedding.provider !== "apple-native") throw new Error("入库测试需要 Apple 原生 WeMM 后端");
    this.deps.canStart?.();
    this.abort = new AbortController();
    this.report = { schema: 1, protocol: "no-warmup-cooldown-v1", state: "running", root: path.resolve(root),
      limit: Number.isFinite(limit) ? Math.min(50, Math.max(1, Math.floor(limit))) : 10,
      repeats: Number.isFinite(repeats) ? Math.min(3, Math.max(1, Math.floor(repeats))) : 1,
      mode: "a", phase: "准备测试", startedAt: Date.now(),
      rows: ["a", "b", "c", "d"].map((mode) => ({ mode, state: "pending", samples: [] })),
    };
    this.deps.onStart?.();
    this.save();
    this.running = this.run(config, this.abort.signal);
    return this.report;
  }

  async stop() {
    if (!this.busy) return;
    this.abort?.abort();
    this.report.phase = "正在停止";
    // Also interrupts a native kernel or a model still starting up.
    await this.deps.runtime.stop();
    await this.running;
  }

  async close() { this.shuttingDown = true; await this.stop(); }
  async completed() { await this.running; return this.report; }

  private save() {
    try {
      fs.mkdirSync(path.dirname(this.storePath), { recursive: true });
      const temporary = this.storePath + ".tmp";
      fs.writeFileSync(temporary, JSON.stringify(this.report, null, 2), { mode: 0o600 });
      fs.renameSync(temporary, this.storePath);
    } catch (error) { this.report.error = `测试结果保存失败：${String(error)}`; }
  }

  private async run(original: Config, signal: AbortSignal) {
    const scan = this.deps.scan || scanLibrary;
    let signature = "";
    let attempted = false;
    try {
      for (const row of this.report.rows) {
        if (signal.aborted) break;
        this.report.mode = row.mode;
        delete this.report.progress;
        row.state = "loading";
        this.report.phase = "切换并加载模型";
        try {
          const config = structuredClone(original);
          const profile = config.profiles[config.activeProfile];
          profile.embedding.native = { ...profile.embedding.native, executionMode: row.mode, mode: "fast", visionCompute: row.mode === "a" ? "gpu" : "ane" };
          const loadStarted = performance.now();
          await this.deps.runtime.reconcile(config);
          row.loadMs = performance.now() - loadStarted;
          if (signal.aborted) break;
          const options: ScanOptions = { config, signal, dryRun: true, benchmark: true, recordPerformance: false,
            onProgress: (job) => { this.report.progress = { ...job }; } };
          for (let repeat = 0; repeat < this.report.repeats; repeat += 1) {
            if (signal.aborted) break;
            if (attempted) {
              delete this.report.progress;
              this.report.phase = "散热等待 5 秒 · 不计入成绩";
              await (this.deps.cooldown ? this.deps.cooldown(signal) : delay(5000, undefined, { signal }));
              if (signal.aborted) break;
            }
            attempted = true;
            row.state = "running";
            this.report.phase = `正式测试 ${repeat + 1} / ${this.report.repeats}`;
            const before = await this.deps.executionSnapshot?.().catch(() => null);
            const sample = await scan(this.report.root, { ...options, limit: this.report.limit });
            const after = await this.deps.executionSnapshot?.().catch(() => null);
            if (this.deps.executionSnapshot) (row.executionEvidence ??= []).push({ before, after });
            if (sample.failed || !sample.indexed) throw new Error(sample.errors[0] || "没有成功处理的素材");
            if (signature && sample.inputSignature !== signature) throw new Error("测试素材或处理参数发生变化，不能参与同组比较");
            signature = sample.inputSignature || "";
            row.samples.push(sample);
          }
          if (signal.aborted) break;
          row.files = row.samples[0]!.indexed;
          row.elapsedMs = median(row.samples.map((sample) => sample.elapsedMs));
          row.filesPerSecond = row.files * 1000 / row.elapsedMs;
          if (row.samples.every((sample) => sample.modelTimings?.requests)) {
            // Backend request totals can include time waiting for its inference lock.
            // Only measured compute stages belong in the model-compute comparison.
            row.modelMs = median(row.samples.map((sample) => sample.modelTimings!.visionMs + sample.modelTimings!.languageMs));
            row.visionMs = median(row.samples.map((sample) => sample.modelTimings!.visionMs));
            row.languageMs = median(row.samples.map((sample) => sample.modelTimings!.languageMs));
          }
          row.state = "done";
        } catch (error) {
          if (signal.aborted) break;
          row.state = "error";
          row.error = String((error as Error).message || error);
        }
        this.save();
      }
    } catch (error) {
      if (!signal.aborted) this.report.error = String(error);
    } finally {
      this.report.state = "restoring";
      this.report.phase = "恢复原计算模式";
      try {
        if (!this.shuttingDown) await this.deps.runtime.reconcile(original);
      } catch (error) { this.report.error = `恢复原模式失败：${String(error)}`; }
      this.report.state = signal.aborted ? "cancelled" : this.report.error || this.report.rows.some((row) => row.state === "error") ? "error" : "done";
      this.report.phase = signal.aborted ? "测试已停止" : this.report.state === "done" ? "测试完成 · 未写入数据库" : "测试结束，请查看错误";
      this.report.finishedAt = Date.now();
      this.save();
      if (!this.shuttingDown) this.deps.onFinish?.();
    }
  }
}
