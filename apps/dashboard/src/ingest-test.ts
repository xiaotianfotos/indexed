import type { IngestTestReport, IngestTestRow } from "../../server/src/ingest-test.js";

const names: Record<string, string> = { a: "纯 GPU", b: "ANE 视觉 + GPU 语言", c: "ANE 视觉＋部分语言计算", d: "ANE 视觉＋更多语言计算" };
const placement: Record<string, string> = {
  a: "GPU：视觉和语言模型；ANE：不参与",
  b: "ANE：视觉模型；GPU：语言模型",
  c: "ANE：视觉＋部分语言前馈计算（MLP）；GPU：其余语言计算，短输入也由 GPU 处理",
  d: "ANE：C 的分工＋部分语言递推计算（GDN）；GPU：其余语言计算及不满足条件的输入",
};
const states: Record<string, string> = { pending: "等待测试", loading: "加载模型", warming: "预热中", running: "测试中", done: "完成", error: "失败" };
const html = (value: unknown) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
type Metric = "elapsedMs" | "averageFileMs" | "filesPerSecond" | "modelMs" | "visionMs" | "languageMs" | "loadMs" | "warmupMs";

export function testMetricValue(row: IngestTestRow, metric: Metric): number | undefined {
  if (metric !== "averageFileMs") return row[metric];
  if (!Number.isFinite(row.files) || !Number.isInteger(row.files) || row.files! <= 0 || !Number.isFinite(row.elapsedMs) || row.elapsedMs! <= 0) return undefined;
  return row.elapsedMs! / row.files!;
}

export function bestTestModes(rows: IngestTestRow[], metric: Metric): string[] {
  const valid = rows.filter((row) => row.state === "done" && Number.isFinite(testMetricValue(row, metric)) && testMetricValue(row, metric)! > 0);
  if (valid.length < 2) return [];
  const best = (metric === "filesPerSecond" ? Math.max : Math.min)(...valid.map((row) => testMetricValue(row, metric)!));
  return valid.filter((row) => Math.abs(testMetricValue(row, metric)! - best) <= best * 0.00001).map((row) => row.mode);
}

export function renderIngestTest(report: IngestTestReport) {
  const get = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
  const detailsOpen = document.querySelector<HTMLDetailsElement>(".test-details")?.open || false;
  const busy = ["running", "restoring"].includes(report.state);
  for (const id of ["test-start", "test-pick-directory", "test-directory", "test-limit", "test-repeats"]) {
    (get(id) as HTMLInputElement).disabled = busy;
  }
  get("test-stop").hidden = !busy;
  (get("test-stop") as HTMLButtonElement).disabled = report.state === "restoring" || report.phase === "正在停止";
  const progress = report.progress;
  const partial = progress?.total === 1 && progress.currentUnitsTotal ? progress.currentUnitsDone / progress.currentUnitsTotal : 0;
  const percent = progress?.total ? Math.min(100, Math.round(Math.max(progress.done, partial) / progress.total * 100)) : 0;
  const elapsed = progress?.startedAt ? Math.max(0, ((progress.finishedAt || Date.now()) - progress.startedAt) / 1000) : 0;
  const speed = elapsed && progress?.indexed ? `当前吞吐 ${(progress.indexed / elapsed).toFixed(2)} 文件/秒` : "吞吐量计算中";
  const done = report.rows.filter((row) => row.state === "done").length;
  get("test-progress").innerHTML = report.state === "idle" ? '<span>选择少量素材，开始一组对照测试。</span>' : `
    <div class="test-status"><strong>${busy ? `${html(report.mode.toUpperCase())} · ${html(names[report.mode])} · ` : ""}${html(report.phase)}</strong><span>${done} / 4 种模式完成</span></div>
    ${busy ? `<progress max="100" value="${percent}" aria-label="当前模式素材处理进度"></progress><span>${progress?.total ? `${progress.done} / ${progress.total} 个素材 · 本轮 ${elapsed.toFixed(1)} 秒 · ${speed}` : "等待模型就绪"}${progress?.currentUnitsTotal ? ` · ${progress.currentUnitsDone}/${progress.currentUnitsTotal} ${html(progress.currentUnitLabel)}` : ""}${progress?.current ? ` · ${html(progress.current)}` : ""}</span>` : ""}
    ${report.error ? `<p class="test-error">${html(report.error)}</p>` : ""}`;
  const baseline = report.rows.find((row) => row.mode === "a" && row.state === "done")?.elapsedMs;
  const cell = (row: IngestTestRow, metric: Metric) => {
    const value = testMetricValue(row, metric);
    const best = bestTestModes(report.rows, metric).includes(row.mode);
    return `<td class="${best ? "metric-best" : ""}"${best ? ' title="本组最佳"' : ""}>${Number.isFinite(value) && value! > 0 ? metric === "averageFileMs" ? value! < 0.1 ? "&lt;0.1" : value!.toFixed(1) : metric === "filesPerSecond" ? value!.toFixed(2) : value! < 10 ? "&lt;0.01" : (value! / 1000).toFixed(2) : "—"}</td>`;
  };
  get("test-results").innerHTML = !report.rows.length ? "" : `
    <div class="test-result-heading"><strong>本组对照结果</strong><span>${html(report.root)} · 上限 ${report.limit} 个 · ${report.repeats} 轮${report.repeats > 1 ? "中位数" : ""} · ${report.protocol ? "无预热，轮间等待 5 秒" : "旧成绩：含独立预热"}</span></div>
    <div class="test-table-scroll"><table class="test-table"><thead><tr><th>计算模式</th><th>整批总耗时 / 秒</th><th title="整批总耗时 ÷ 完成文件数；不是单个请求的延迟">平均每文件 / 毫秒</th><th title="纯 GPU 整批耗时 ÷ 当前模式整批耗时；1.20× 表示吞吐量提升 20%">速度倍率</th><th>视觉累计 / 秒</th><th>语言累计 / 秒</th></tr></thead><tbody>${report.rows.map((row) => `<tr>
      <th><span class="test-mode-letter">${html(row.mode.toUpperCase())}</span>${html(names[row.mode])}<small>${html(row.state === "done" ? `${row.files} 个素材完成` : !busy && ["running", "loading", "warming"].includes(row.state) ? "已中断" : states[row.state])}${row.error ? ` · ${html(row.error)}` : ""}</small></th>
      ${cell(row, "elapsedMs")}${cell(row, "averageFileMs")}
      <td class="${bestTestModes(report.rows, "filesPerSecond").includes(row.mode) ? "metric-best" : ""}">${row.state === "done" && baseline && row.elapsedMs ? row.mode === "a" ? "基线" : `${(baseline / row.elapsedMs).toFixed(2)}×` : "—"}</td>
      ${cell(row, "visionMs")}${cell(row, "languageMs")}</tr><tr><td colspan="6" class="test-placement">${html(row.mode.toUpperCase())} · ${html(placement[row.mode])}</td></tr>`).join("")}</tbody></table></div>
    <details class="test-details"${detailsOpen ? " open" : ""}><summary>吞吐量、计算与加载明细</summary><p class="panel-note">平均每文件＝整批总耗时÷文件数，不代表单个请求延迟。视觉、语言为各请求阶段耗时之和，不是硬件占用时间；流水线有重叠，累计分项不能相加当作整批耗时。多轮测试各列取中位数，分项相加可能不等于总项。</p><div class="test-table-scroll"><table class="test-table"><thead><tr><th>模式</th><th>吞吐量 / 文件每秒</th><th>视觉＋语言累计 / 秒</th><th>切换加载 / 秒</th>${report.protocol ? "" : "<th>历史预热 / 秒</th>"}</tr></thead><tbody>${report.rows.map((row) => `<tr><th>${html(row.mode.toUpperCase())} · ${html(names[row.mode])}</th>${cell(row, "filesPerSecond")}${cell(row, "modelMs")}${cell(row, "loadMs")}${report.protocol ? "" : cell(row, "warmupMs")}</tr>`).join("")}</tbody></table></div></details>`;
}
