// @ts-nocheck -- DOM bindings stay framework-free.
import { APPLE_EMBEDDING_DEFAULT_DIMENSION, appleEmbeddingModel, APPLE_EXECUTION_MODES, requireAppleExecutionMode } from "@indexed/contracts";
import { openDialog, closeDialog } from "./dialog-focus.js";
import { LatestRequest } from "./latest-request.js";
import { setupPresentation } from "./setup-presentation.js";
import { scanPresentation } from "./scan-presentation.js";
import { renderIngestTest } from "./ingest-test.js";

/**
 * Indexed dashboard.
 *
 * One screen does the finding: describe a shot, get shots back. Everything else -
 * walking directories, embedding, extracting previews - is the server's job, so the
 * UI never asks anyone to browse an asset list. Directory management lives in
 * settings and reports counts only.
 */
const searches = new LatestRequest();
const healthChecks = new LatestRequest();
const profileLoads = new LatestRequest();
const directoryLoads = new LatestRequest();
const state = {
  readiness: { phase: "checking", detail: "" },
  health: null,
  developer: false,
  view: "search",
  query: "",
  assetKind: "",
  hits: [],
  libraries: [],
  coverage: null,
  status: null,
  config: null,
  settingsTab: "assets",
  editingProfileId: "",
  profileDirty: false,
  scanning: false,
  refreshing: false,
  scanSettingsDirty: false,
  timer: 0,
  browsePath: "",
  browseParent: "",
  browseKindsInitialized: false,
  trackedKinds: ["video", "image", "document"],
  documentFormats: [],
  modal: null,
  document: null,
};
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const views = {
  search: ["检索", "搜索你的图片、视频与文档"],
  settings: ["设置", "素材追踪、模型与存储"],
};
const jobStates = { idle: "空闲", queued: "排队中", running: "扫描中", done: "完成", cancelled: "已停止", error: "出错" };
const assetKinds = { video: "视频", image: "图片", document: "文档" };
const editableDocumentExtensions = new Set(["md", "mdx", "txt", "json", "jsonl", "csv", "srt", "vtt"]);
const appleExecutionModes = {
  a: "视觉与语言计算使用 GPU，作为性能对照。",
  b: "ANE 处理视觉，GPU 处理语言。使用公开接口。",
  c: "ANE：视觉和部分语言前馈计算（MLP）；GPU：其余语言计算，短输入也由 GPU 处理。使用私有 ANE 接口。",
  d: "ANE：C 的分工加部分语言递推计算（GDN）；GPU：其余语言计算及不满足条件的输入。不是全语言模型上 ANE。",
};
const appleExecutionModeLabels = {
  a: "纯 GPU（性能基线）",
  b: "ANE 视觉 + GPU 语言（稳定）",
  c: "ANE 视觉＋部分语言计算",
  d: "ANE 视觉＋更多语言计算（研究）",
};

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "content-type": "application/json", ...(options.headers || {}) }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("show"), 2400);
}

function number(value) {
  return Number(value || 0).toLocaleString("zh-CN");
}

function time(seconds) {
  const value = Math.max(0, Math.floor(Number(seconds || 0)));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
}

function duration(milliseconds) {
  const value = Math.max(0, Number(milliseconds || 0));
  if (!value) return "0 秒";
  if (value < 1000) return "少于 1 秒";
  const seconds = Math.round(value / 1000);
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} 分钟`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours} 小时 ${rest} 分钟` : `${hours} 小时`;
}

function preciseDuration(milliseconds) {
  const value = Math.max(0, Number(milliseconds || 0));
  if (!value) return "0 秒";
  if (value < 1000) return `${Math.max(1, Math.round(value))} 毫秒`;
  const seconds = value / 1000;
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} 秒`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  if (minutes < 60) return rest ? `${minutes} 分 ${rest} 秒` : `${minutes} 分钟`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes ? `${hours} 小时 ${remainingMinutes} 分钟` : `${hours} 小时`;
}

function realtimeSpeedLabel(mediaSeconds, processingMs) {
  const speed = Number(mediaSeconds || 0) / Math.max(0.001, Number(processingMs || 0) / 1000);
  if (!Number.isFinite(speed) || speed <= 0) return "";
  const multiple = speed >= 10 ? speed.toFixed(0) : speed >= 1 ? speed.toFixed(1) : speed.toFixed(2);
  const processingMsPerMediaMinute = 60_000 / speed;
  return `${multiple}× 实时 · 处理 1 分钟视频约需 ${preciseDuration(processingMsPerMediaMinute)}`;
}

function bytes(value) {
  const size = Number(value || 0);
  if (size >= 1024 ** 3) return `${(size / 1024 ** 3).toFixed(1)} GB`;
  if (size >= 1024 ** 2) return `${(size / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(size / 1024))} KB`;
}

function relativeTime(value) {
  const timestamp = Number.isFinite(Number(value)) && String(value).trim() !== "" ? Number(value) : Date.parse(value || "");
  if (!timestamp) return "从未";
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 60) return "刚刚";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
  return `${Math.floor(seconds / 86400)} 天前`;
}

/** A search is worth a URL: a refresh, a back button and a shared link all work. */
function currentUrl() {
  const params = new URLSearchParams({ view: state.view });
  if (state.view === "search" && state.query) {
    params.set("q", state.query);
    if (state.assetKind) params.set("kind", state.assetKind);
  }
  if (state.view === "settings" && state.settingsTab !== "assets") params.set("tab", state.settingsTab);
  return `/?${params}`;
}

function showView(name, updateUrl = true) {
  const selected = views[name] ? name : "search";
  if (state.view === "settings" && selected !== "settings" && state.profileDirty
    && !confirm("模型与数据库中有未保存修改，放弃这些修改？")) return false;
  if (selected !== "settings") state.profileDirty = false;
  state.view = selected;
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${selected}`));
  $("#page-title").textContent = views[selected][0];
  $("#page-subtitle").textContent = views[selected][1];
  const toggle = $("#view-toggle");
  toggle.classList.toggle("active", selected === "settings");
  toggle.setAttribute("aria-pressed", String(selected === "settings"));
  toggle.querySelector("span").textContent = selected === "settings" ? "←" : "⚙";
  toggle.querySelector("b").textContent = selected === "settings" ? "返回检索" : "设置";
  if (updateUrl) history.replaceState({}, "", currentUrl());
  if (selected !== "search") { searches.cancel(); closeAssetModal(); }
  void refresh();
  if (selected === "settings") void loadFacts();
  return true;
}

function showSettingsTab(name, updateUrl = true) {
  const selected = name === "test" && !state.developer ? "assets" : ["engine", "test"].includes(name) ? name : "assets";
  if (state.settingsTab === "engine" && selected !== "engine" && state.profileDirty
    && !confirm("模型与数据库中有未保存修改，放弃这些修改？")) return false;
  if (selected !== "engine") state.profileDirty = false;
  state.settingsTab = selected;
  $$("[data-settings-tab]").forEach((button) => {
    const active = button.dataset.settingsTab === selected;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  $$("[data-settings-page]").forEach((page) => {
    const active = page.dataset.settingsPage === selected;
    page.classList.toggle("active", active);
    page.hidden = !active;
  });
  if (updateUrl) history.replaceState({}, "", currentUrl());
  if (selected === "engine") void loadFacts();
  void refresh();
  return true;
}

/* -------------------------------------------------------------------------- */
/* Search                                                                      */
/* -------------------------------------------------------------------------- */

async function runSearch(event) {
  event?.preventDefault();
  const request = searches.begin();
  const query = $("#asset-search-query").value.trim();
  const stage = $("#search-stage");
  state.query = query;
  history.replaceState({}, "", currentUrl());
  if (!query) {
    state.hits = [];
    $("#asset-results").innerHTML = "";
    $("#asset-search-status").textContent = "";
    stage.classList.remove("searching");
    $("#asset-results").setAttribute("aria-busy", "false");
    return;
  }
  if (!searchSetup().searchable) { $("#asset-search-status").textContent = searchSetup().title; return; }
  stage.classList.add("searching");
  $("#asset-results").setAttribute("aria-busy", "true");
  const started = performance.now();
  $("#asset-search-status").textContent = "检索中";
  try {
    const data = await api("/api/assets/search", {
      signal: request.signal,
      method: "POST",
      body: JSON.stringify({ query, ...(state.assetKind ? { kind: state.assetKind } : {}), limit: 60 }),
    });
    if (!request.current()) return;
    state.hits = data.hits || [];
    renderHits();
    const elapsed = Math.round(performance.now() - started);
    const hidden = Number(data.hiddenMissing || 0);
    const sourceSummary = Number(data.sources?.web || 0)
      ? ` · 本机 ${number(data.sources?.local)} / 网页 ${number(data.sources?.web)}`
      : "";
    $("#asset-search-status").textContent = `${state.hits.length ? `${number(state.hits.length)} 个结果` : "没有匹配的结果"} · ${elapsed}ms`
      + sourceSummary
      + (hidden ? ` · 另有 ${number(hidden)} 条结果的文件已不在磁盘上` : "");
  } catch (error) {
    if (!request.current()) return;
    state.hits = [];
    renderHits();
    $("#asset-search-status").textContent = error.message;
  } finally {
    if (request.current()) $("#asset-results").setAttribute("aria-busy", "false");
  }
}

function renderHits() {
  $("#asset-results").innerHTML = state.hits.length ? state.hits.map((asset, index) => `
    <article class="asset-card" role="button" tabindex="0" aria-label="打开 ${escapeHtml(asset.name)}" data-kind="${escapeHtml(asset.kind)}" data-source="${escapeHtml(asset.source || "local")}" data-index="${index}">
      ${asset.previewUrl && asset.kind !== "document" ? `<img src="${escapeHtml(asset.previewUrl)}" alt="" loading="lazy" />` : `<div class="media-empty" aria-hidden="true">${asset.kind === "document" ? "文档" : "暂无预览"}</div>`}
      <div class="asset-card-body">
        <strong title="${escapeHtml(asset.openUrl || asset.path)}">${escapeHtml(asset.name)}</strong>
        <div class="asset-card-meta">
          <span class="asset-kind">${escapeHtml(assetKinds[asset.kind] || "素材")}</span>
          ${asset.openUrl ? `<span class="asset-origin">网页 ↗</span>` : ""}
          ${asset.kind === "video" && asset.endSeconds ? `<span class="asset-timecode">${time(asset.startSeconds)}–${time(asset.endSeconds)}</span>` : ""}
        </div>
        ${asset.sidecarText ? `<p class="asset-sidecar">${escapeHtml(asset.sidecarText)}</p>` : ""}
        <p class="asset-path" title="${escapeHtml(asset.openUrl || asset.path)}">${escapeHtml(asset.libraryName || (asset.openUrl ? "网页收藏" : "本机素材"))}</p>
      </div>
    </article>`).join("") : `<div class="empty">${state.query
      ? `没有找到匹配素材。试试其他描述或文件类型。`
      : ""}</div>`;
  $$("#asset-results img").forEach((image) => image.addEventListener("error", () => image.replaceWith(Object.assign(document.createElement("div"), { className: "media-empty", textContent: "预览不可用" })), { once: true }));
  $$("#asset-results .asset-card").forEach((card) => {
    card.addEventListener("click", () => openAsset(state.hits[Number(card.dataset.index)]));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); card.click(); }
    });
  });
}

function openAsset(asset) {
  if (!asset) return;
  if (asset.openUrl) {
    window.open(asset.openUrl, "_blank", "noopener,noreferrer");
    return;
  }
  openAssetModal(asset);
}

function selectAssetKind(chip) {
  state.assetKind = chip.dataset.kind || "";
  $$("#asset-kind-chips .chip").forEach((item) => item.classList.toggle("active", item === chip));
  history.replaceState({}, "", currentUrl());
  if (state.query) void runSearch();
}

/* -------------------------------------------------------------------------- */
/* Asset viewer                                                                */
/* -------------------------------------------------------------------------- */

function openAssetModal(asset) {
  if (!asset) return;
  state.modal = asset;
  state.document = null;
  const start = Math.max(0, Number(asset.startSeconds || 0));
  const preview = asset.previewUrl || `/api/assets/preview?asset=${encodeURIComponent(asset.path)}${asset.kind === "video" && start ? `&at=${Math.floor(start)}` : ""}`;
  const media = $("#asset-modal-media");
  if (asset.kind === "video") {
    media.innerHTML = `<video controls autoplay playsinline preload="metadata" src="/api/assets/stream?asset=${encodeURIComponent(asset.path)}#t=${Math.floor(start)}"></video>`;
    const video = media.querySelector("video");
    video.addEventListener("loadedmetadata", () => { video.currentTime = start; }, { once: true });
  } else if (asset.kind === "image") {
    media.innerHTML = `<img src="${escapeHtml(preview)}" alt="" />`;
    media.querySelector("img").addEventListener("error", () => { media.innerHTML = `<div class="media-empty"></div>`; }, { once: true });
  } else {
    // Documents are the editable kind: a textarea the loaded text is dropped into.
    media.innerHTML = `<textarea class="asset-document" id="asset-document" spellcheck="false" readonly placeholder="读取中"></textarea>`;
    void loadAssetDocument(asset);
  }
  $("#asset-modal-title").textContent = asset.name || asset.path;
  $("#asset-modal-meta").innerHTML = `
    <p class="modal-path">${escapeHtml(asset.path)}</p>
    <div class="modal-stats">
      <span>${bytes(asset.size)}</span>
      ${asset.duration ? `<span>${time(asset.duration)}</span>` : ""}
      ${asset.kind === "video" && asset.segmentIndex != null ? `<span>分片 ${escapeHtml(asset.segmentIndex)}</span>` : ""}
      ${asset.kind === "document" && asset.segmentIndex != null ? `<span>命中片段 ${number(Number(asset.segmentIndex) + 1)}</span>` : ""}
      ${asset.libraryName ? `<span>${escapeHtml(asset.libraryName)}</span>` : ""}
      ${Number.isFinite(asset.score) ? `<span title="相似度分数，不是准确率">相似度 ${Number(asset.score).toFixed(3)}</span>` : ""}
    </div>
    ${asset.sidecarText ? `<p class="asset-sidecar"><em>${asset.kind === "document" ? "命中文本" : "同期文本"}</em>${escapeHtml(asset.sidecarText)}</p>` : ""}`;
  $("#asset-copy-status").textContent = "";
  $("#asset-save-status").textContent = "";
  $("#asset-save").hidden = true;
  openDialog($("#asset-modal"), $("#asset-modal-close"));
}

async function loadAssetDocument(asset) {
  const editor = $("#asset-document");
  const status = $("#asset-save-status");
  const extension = String(asset.name || asset.path || "").split(".").pop()?.toLowerCase() || "";
  if (!editableDocumentExtensions.has(extension)) {
    editor.value = asset.sidecarText || "已建立检索索引，但当前片段没有可显示的文字预览。";
    editor.readOnly = true;
    status.textContent = `${extension.toUpperCase() || "文档"} 源文件只读 · 显示最相关的索引片段`;
    return;
  }
  try {
    const doc = await api(`/api/assets/document?asset=${encodeURIComponent(asset.path)}`);
    if (state.modal?.path !== asset.path || !editor.isConnected) return;
    state.document = doc;
    editor.value = doc.text;
    editor.readOnly = false;
    $("#asset-save").hidden = false;
    status.textContent = `${bytes(doc.bytes)} · ${doc.vectors ? "已有向量" : "尚未索引"}`;
  } catch (error) {
    if (state.modal?.path !== asset.path || !status) return;
    status.textContent = error.message;
  }
}

async function saveAssetDocument() {
  const editor = $("#asset-document");
  const asset = state.modal;
  if (!editor || !asset || editor.readOnly) return;
  const status = $("#asset-save-status");
  const button = $("#asset-save");
  button.disabled = true;
  status.textContent = "保存中";
  try {
    const result = await api("/api/assets/document", {
      method: "PUT",
      body: JSON.stringify({ path: asset.path, text: editor.value }),
    });
    state.document = { ...state.document, text: editor.value, bytes: result.bytes };
    status.textContent = `${bytes(result.bytes)} · ${result.notice}${result.previews ? ` · 已同步 ${result.previews} 条素材摘要` : ""}`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function closeAssetModal() {
  const editor = $("#asset-document");
  if (editor && !editor.readOnly && editor.value !== (state.document?.text ?? "")) {
    if (!confirm("有未保存的修改，确定关闭？")) return;
  }
  $("#asset-modal-media video")?.pause();
  closeDialog($("#asset-modal"));
  state.modal = null;
  state.document = null;
}

async function copyAssetPath() {
  const path = state.modal?.path;
  if (!path) return;
  try {
    await navigator.clipboard.writeText(path);
    $("#asset-copy-status").textContent = "路径已复制";
  } catch { $("#asset-copy-status").textContent = path; }
}

/* -------------------------------------------------------------------------- */
/* Library accounting                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Counts, refreshed on one timer: the search screen shows a single quiet line,
 * settings expands it per directory. While a scan runs we poll faster so the
 * numbers visibly climb instead of making the user hunt for progress.
 */
async function refresh() {
  if (state.refreshing) return;
  state.refreshing = true;
  clearTimeout(state.timer);
  state.timer = 0;
  try {
    state.ingestTest = await api("/api/ingest-test");
    renderIngestTest(state.ingestTest);
    if (state.settingsTab === "test" && state.view === "settings") {
      if (!state.testFormInitialized) {
        if (state.ingestTest.root) $("#test-directory").value = state.ingestTest.root;
        $("#test-limit").value = state.ingestTest.limit;
        $("#test-repeats").value = String(state.ingestTest.repeats);
        state.testFormInitialized = true;
      }
      return;
    }
    const status = await api("/api/assets/libraries");
    state.status = status;
    state.libraries = status.libraries || [];
    state.scanning = state.libraries.some((library) => ["queued", "running"].includes(library.job?.state));
    // Progress is cheap; directory traversal and vector coverage are not.
    // Keep coverage out of the running inference path and reconcile on completion.
    if (!state.scanning || !state.coverage) state.coverage = await api("/api/assets/coverage");
    renderLibraryFoot();
    renderReadiness();
    if (state.view === "settings") {
      renderDirectories();
      renderScanSettings();
    }
  } catch (error) {
    if (state.view === "settings") $("#directory-note").textContent = error.message;
  } finally {
    state.refreshing = false;
    void checkStatus();
    const testBusy = ["running", "restoring"].includes(state.ingestTest?.state);
    if (!document.hidden) state.timer = setTimeout(() => void refresh(), testBusy || state.scanning ? 500 : state.settingsTab === "test" ? 2000 : 15000);
  }
}

function renderLibraryFoot() {
  const foot = $("#library-foot");
  if (!state.libraries.length) {
    foot.innerHTML = `还没有素材目录。<button class="link" type="button" data-goto="settings">添加目录</button>`;
  } else {
    const totals = state.coverage?.totals || { files: 0, cached: 0, pending: 0, stale: 0, missing: 0 };
    const waiting = Number(totals.pending || 0) + Number(totals.stale || 0);
    foot.innerHTML = `${state.scanning ? `<span class="foot-live">索引中</span> · ` : ""}`
      + `${number(state.libraries.length)} 个目录 · ${number(totals.files)} 个素材 · 已缓存 ${number(totals.cached)}`
      + (waiting ? ` · 待缓存 ${number(waiting)}` : "")
      + (Number(totals.missing || 0) ? ` · <span class="foot-warn">${number(totals.missing)} 个文件已失效</span>` : "")
      + ` · <button class="link" type="button" data-goto="settings">管理目录</button>`;
  }
  foot.querySelector("[data-goto]")?.addEventListener("click", () => showView("settings"));
}

/* -------------------------------------------------------------------------- */
/* Settings                                                                    */
/* -------------------------------------------------------------------------- */

function renderDirectories() {
  const coverage = new Map((state.coverage?.libraries || []).map((item) => [item.path, item]));
  const profile = state.config?.profiles?.[state.config?.activeProfile];
  $("#scan-engine-label").textContent = profile
    ? `当前配置 · ${profile.embedding?.provider === "apple-native" ? appleExecutionModeLabels[profile.embedding?.native?.executionMode || "b"] : profile.embedding?.model || "外部模型"}`
    : "读取计算配置…";
  $("#directory-list").innerHTML = state.libraries.length ? state.libraries.map((library) => {
    const job = library.job || {};
    const { total, done, busy, percent, elapsedMs, filesPerSecond } = scanPresentation(job);
    const videoRun = total === 1 && Number(job.currentMediaDurationSeconds) > 0;
    const displayedSpeed = videoRun
      ? (elapsedMs > 0 && job.currentProcessedMediaSeconds > 0 && !job.failed ? job.currentProcessedMediaSeconds * 1000 / elapsedMs : null)
      : filesPerSecond;
    const unitDone = Number(job.currentUnitsDone || 0);
    const unitTotal = Number(job.currentUnitsTotal || 0);
    const completedUnitMs = Number(job.currentCompletedUnitMs || 0);
    const averageUnitMs = unitDone && completedUnitMs ? completedUnitMs / unitDone : 0;
    // Shared per-file counters are only reliable with one worker.
    const singleWorker = Math.min(total || 1, Number(job.concurrency || 1)) === 1;
    const unitSummary = singleWorker && unitTotal
      ? `${escapeHtml(job.currentUnitLabel || "步骤")} ${number(unitDone)} / ${number(unitTotal)}`
        + (averageUnitMs ? ` · 平均 ${preciseDuration(averageUnitMs)}` : "")
        + (job.currentProcessedMediaSeconds ? ` · ${realtimeSpeedLabel(job.currentProcessedMediaSeconds, completedUnitMs)}` : "")
      : "";
    const processingMode = job.processingMode === "background" ? "后台处理" : "满速处理";
    const item = coverage.get(library.path) || {};
    const startingWaiting = Number(item.pending || 0) + Number(item.stale || 0);
    const waiting = busy && Number.isFinite(Number(job.remaining))
      ? Number(job.remaining)
      : startingWaiting;
    // While busy, show this run's live count instead of cached whole-library coverage.
    const cached = busy ? Number(job.indexed || 0) : Number(item.cached || 0);
    return `
    <article class="directory-row" data-path="${escapeHtml(library.path)}" data-busy="${busy}">
      <div class="directory-main">
        <div class="directory-title"><strong title="${escapeHtml(library.path)}">${escapeHtml(library.name || library.path)}</strong><span class="job-pill" data-state="${escapeHtml(job.state || "idle")}">${escapeHtml(jobStates[job.state] || "空闲")}</span></div>
        <p class="directory-path" title="${escapeHtml(library.path)}">${escapeHtml(library.path)}</p>
        ${job.error ? `<p class="job-error">${escapeHtml(job.error)}</p>` : ""}
      </div>
      <div class="directory-stats">
        <button class="link tracked-kinds" data-action="kinds" type="button">${escapeHtml((library.kinds || item.kinds || []).map((kind) => assetKinds[kind] || kind).join(" · ") || "文件类型")}</button>
        <span class="stat"><b>${number(item.files ?? library.assetCount)}</b> 个素材</span>
        <span class="stat stat-ok"><b>${number(cached)}</b> ${busy ? "本轮入库" : "已入库"}</span>
        ${waiting ? `<span class="stat stat-warn"><b>${number(waiting)}</b> 待入库</span>` : ""}
        ${Number(item.missing) ? `<span class="stat stat-danger"><b>${number(item.missing)}</b> 文件已失效</span>` : ""}
        ${item.truncated ? `<span class="stat stat-warn">盘点达到 ${number(state.status?.maxFilesPerLibrary)} 个上限，实际素材更多</span>` : ""}
        <span class="stat stat-dim">上次扫描 ${relativeTime(library.lastScanAt)}</span>
      </div>
      <div class="directory-actions">
        <button class="quiet${busy ? " danger" : ""}" data-action="${busy ? "stop" : "scan"}" type="button">${busy ? (job.cancelRequested ? "停止中" : "停止") : "扫描"}</button>
        <button class="quiet" data-action="prune" type="button">清理失效</button>
        <button class="quiet danger" data-action="remove" type="button" title="移除目录不会删除磁盘文件">移除</button>
      </div>
      ${busy || total ? `<div class="scan-progress-card">
        <p class="scan-run-label">${busy ? "当前任务" : "上次任务"} · ${escapeHtml(appleExecutionModeLabels[job.executionMode] || (job.executionMode === "remote" ? "外部模型" : "未记录计算模式"))}</p>
        <div class="scan-metrics">
          <div><span>本轮已处理</span><strong>${number(done)} <small>/ ${number(total)}</small></strong></div>
          <div><span>处理用时</span><strong>${elapsedMs ? (elapsedMs / 1000).toFixed(2) : "—"}<small> 秒</small></strong></div>
          <div><span>${videoRun ? "视频处理速度" : "平均入库速度"}</span><strong>${displayedSpeed === null ? "—" : displayedSpeed.toFixed(2)}<small> ${videoRun ? "秒视频/秒" : "个/秒"}</small></strong></div>
        </div>
        <div class="directory-progress${total ? "" : " indeterminate"}" role="progressbar" aria-label="${escapeHtml(library.name || "目录")}入库进度" aria-valuemin="0" aria-valuemax="100" ${total ? `aria-valuenow="${percent}"` : ""}><i style="width:${total ? percent : 35}%"></i></div>
        <div class="progress-meta"><span>${total ? `${processingMode} · 成功 ${number(job.indexed)}${job.failed ? ` · 失败 ${number(job.failed)}` : ""}` : escapeHtml(job.phase || "正在准备")}</span><span>${total ? `${percent}%` : ""}</span></div>
        ${unitSummary ? `<p class="progress-unit">${unitSummary}</p>` : ""}
        ${busy && job.current ? `<p class="progress-file">${singleWorker ? "正在处理" : `并发 ${number(job.concurrency)} · 最近启动`} · ${escapeHtml(job.current)}</p>` : ""}
      </div>` : ""}
    </article>`;
  }).join("") : `<div class="empty">还没有素材目录。点右上角「添加目录」，在服务器上选中放素材的文件夹。</div>`;
  $$("#directory-list .directory-row").forEach((row) => {
    const library = state.libraries.find((item) => item.path === row.dataset.path);
    row.querySelectorAll("[data-action]").forEach((button) => button.addEventListener("click", () => {
      if (button.dataset.action === "scan") void scanDirectory(library);
      if (button.dataset.action === "stop") void stopDirectory(library);
      if (button.dataset.action === "prune") void pruneDirectory(library);
      if (button.dataset.action === "remove") void removeDirectory(library);
      if (button.dataset.action === "kinds") void openBrowse(library.path);
    }));
  });
  const totals = state.coverage?.totals;
  $("#directory-note").textContent = totals && state.libraries.length
    ? `共 ${number(totals.files)} 个素材 · 已入库 ${number(totals.cached)} · 待入库 ${number(totals.pending + totals.stale)}`
      + (Number(totals.missing) ? ` · ${number(totals.missing)} 个文件已失效，可在对应目录上清理` : "")
    : "";
}

async function scanDirectory(library) {
  if (!library) return;
  try {
    const result = await api("/api/assets/libraries/scan", { method: "POST", body: JSON.stringify({ path: library.path }) });
    toast(result.queued?.length ? `已排队 ${result.queued[0].split("/").pop()}` : "这个目录正在扫描");
    await refresh();
  } catch (error) { toast(error.message); }
}

async function stopDirectory(library) {
  if (!library) return;
  try {
    const result = await api("/api/assets/libraries/stop", { method: "POST", body: JSON.stringify({ path: library.path }) });
    toast(result.stopping?.length ? "正在停止当前入库请求" : "扫描已停止");
    await refresh();
  } catch (error) { toast(error.message); }
}

/**
 * Dead rows are counted before asking, so the confirmation states the number
 * it is about to delete. Pruning touches the index only, never the files.
 */
async function pruneDirectory(library) {
  if (!library) return;
  const name = library.name || library.path;
  try {
    const preview = await api("/api/assets/prune", { method: "POST", body: JSON.stringify({ path: library.path, dryRun: true }) });
    const missing = preview.missingAssets || [];
    if (!missing.length) return toast(`${name} 没有失效记录`);
    const rows = missing.reduce((sum, item) => sum + Number(item.rows || 0), 0);
    if (!confirm(`“${name}”里 ${missing.length} 个素材的文件已经不在磁盘上了（例如 ${missing[0]?.path?.split("/").pop()}），对应 ${rows} 条索引记录。删除这些记录？磁盘文件不受影响。`)) return;
    const result = await api("/api/assets/prune", { method: "POST", body: JSON.stringify({ path: library.path }) });
    toast(`已清理 ${(result.missingAssets || []).length} 个失效素材、${number(result.removedRows)} 条记录`);
    await refresh();
  } catch (error) { toast(error.message); }
}

async function removeDirectory(library) {
  if (!library) return;
  const name = library.name || library.path;
  if (!confirm(`移除素材库“${name}”？只删除本地索引记录，磁盘文件不受影响。`)) return;
  const purge = confirm(`同时删除“${name}”的向量记录？此操作不可恢复。`);
  try {
    await api(`/api/assets/libraries?id=${encodeURIComponent(library.path)}${purge ? "&purge=true" : ""}`, { method: "DELETE" });
    toast(purge ? "已移除并删除向量" : "已移除素材库");
    await refresh();
  } catch (error) { toast(error.message); }
}

function renderScanSettings() {
  const status = state.status || {};
  const interval = Math.max(1, Math.round(Number(status.scanIntervalSeconds || 900) / 60));
  if (!state.scanSettingsDirty) {
    $("#auto-scan").checked = Boolean(status.autoScan);
    $("#scan-interval").value = interval;
    $("#scan-limit").value = Number(status.maxAssetsPerScan || 400);
    $("#scan-discovery-limit").value = Number(status.maxFilesPerLibrary || 100000);
  }
  const activeJobs = state.libraries.map((library) => library.job || {}).filter((job) => ["queued", "running"].includes(job.state));
  const activeDone = activeJobs.reduce((sum, job) => sum + Number(job.done || 0), 0);
  const activeTotal = activeJobs.reduce((sum, job) => sum + Number(job.total || 0), 0);
  $("#scan-summary").textContent = state.scanning
    ? `正在处理${activeTotal ? ` ${number(activeDone)} / ${number(activeTotal)}` : "，准备中"}`
    : status.autoScan
      ? `每 ${interval} 分钟扫描一轮，每轮最多 ${number(status.maxAssetsPerScan || 400)} 个素材`
      : "手动扫描";
  $("#scan-all").textContent = state.scanning ? "停止全部" : "满速扫描";
  $("#background-scan-all").disabled = state.scanning;
  renderIngestEstimate();
}

function renderIngestEstimate() {
  const estimate = state.coverage?.ingestEstimate;
  const source = $("#ingest-estimate-source");
  const summary = $("#ingest-estimate-summary");
  const rates = $("#ingest-estimate-rates");
  if (!estimate) {
    source.textContent = "等待扫描数据";
    summary.textContent = "尚未读取到入库性能历史。";
    rates.textContent = "";
    return;
  }
  const sourceLabel = estimate.source === "exact" ? "同配置历史"
    : estimate.source === "provider" ? "同供应商粗估" : "尚无历史";
  source.textContent = `${sourceLabel}${estimate.sampleCount ? ` · ${number(estimate.sampleCount)} 次` : ""}`;
  if (!estimate.sampleCount) {
    summary.textContent = "完成首次入库后显示预估。";
  } else if (!estimate.pendingAssets) {
    summary.textContent = "没有待入库素材。";
  } else if (estimate.estimatedMs !== null) {
    summary.textContent = `待入库 ${number(estimate.pendingAssets)} 个素材，预计 ${duration(estimate.lowerEstimatedMs)} – ${duration(estimate.upperEstimatedMs)}。`;
  } else {
    const missing = (estimate.unavailableKinds || []).map((kind) => assetKinds[kind] || kind).join("、");
    summary.textContent = `待入库 ${number(estimate.pendingAssets)} 个素材，但还缺少${missing ? `“${missing}”` : "对应类型"}的历史样本。`;
  }
  const labels = Object.entries(estimate.ratesByKind || {}).flatMap(([kind, rate]) => {
    if (!rate.millisecondsPerAsset) return [];
    const measured = duration(rate.millisecondsPerAsset);
    return [`${assetKinds[kind] || kind}${rate.millisecondsPerAsset < 1000 ? "" : "约 "}${measured}/个`];
  });
  rates.textContent = labels.length
    ? `${labels.join(" · ")} · 最近样本 ${relativeTime(estimate.lastSampleAt)}`
    : "";
  $("#performance-history-path").textContent = estimate.historyPath || state.config?.performanceHistoryPath || "";
}

async function saveScanSettings() {
  const button = $("#save-scan");
  button.disabled = true;
  try {
    await api("/api/assets/scan-settings", {
      method: "PUT",
      body: JSON.stringify({
        autoScan: $("#auto-scan").checked,
        scanIntervalSeconds: Math.max(1, Number($("#scan-interval").value || 15)) * 60,
        maxAssetsPerScan: Math.max(1, Number($("#scan-limit").value || 400)),
        maxFilesPerLibrary: Math.max(1000, Number($("#scan-discovery-limit").value || 100000)),
      }),
    });
    state.scanSettingsDirty = false;
    toast("已保存后台扫描设置");
    await refresh();
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

async function scanAll(background = false) {
  const button = $("#scan-all");
  button.disabled = true;
  try {
    const stopping = state.scanning;
    const result = await api(stopping ? "/api/assets/libraries/stop" : "/api/assets/libraries/scan", { method: "POST", body: JSON.stringify(background ? { background: true } : {}) });
    toast(stopping
      ? (result.stopping?.length ? "正在停止全部入库任务" : "扫描队列已停止")
      : (result.queued?.length ? `已排队 ${number(result.queued.length)} 个目录${background ? "（后台单文件）" : "（满速）"}` : "所有目录都已在队列中"));
    await refresh();
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

function activeProfileConfig() {
  return state.config?.profiles?.[state.config.activeProfile] || {};
}

function editingProfileConfig() {
  return state.config?.profiles?.[state.editingProfileId] || {};
}

function localProfileEntries() {
  return Object.entries(state.config?.profiles || {}).filter(([, profile]) => profile.storage?.provider !== "aliyun");
}

function safeSpaceId(model, dimension, style, profileId) {
  return `${model || "unknown"}-${dimension || 0}-${style || "auto"}-${profileId || "profile"}-indexed-v1`
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-");
}

function renderProfileActions() {
  const config = state.config || {};
  const id = state.editingProfileId;
  const active = id === config.activeProfile;
  const button = $("#save-profile");
  button.disabled = active && !state.profileDirty;
  button.textContent = active
    ? state.profileDirty ? "保存更改" : "已启用"
    : state.profileDirty ? "保存并启用" : "启用此档案";
  $("#profile-note").textContent = active
    ? state.profileDirty ? "有未保存修改" : ""
    : state.profileDirty ? "保存后将切换到这个档案" : "查看中；启用前不会影响当前检索";
}

function renderAppleBackendFields() {
  const apple = $("#embedding-provider").value === "apple-native";
  const mode = $("#apple-execution-mode").value;
  const model = $("#embedding-model");
  model.readOnly = apple;
  model.title = apple ? "Apple 原生 WeMM 的模型名由输出维度自动确定" : "";
  model.placeholder = apple ? "由 Apple 原生后端固定" : "wemm-embedding-9b";
  if (apple) model.value = appleEmbeddingModel($("#embedding-dimension").value);
  $("#apple-execution-mode-field").hidden = !apple;
  $("#apple-experiments-field").hidden = !apple;
  for (const option of $("#apple-execution-mode").options) {
    const experimental = ["a", "c", "d"].includes(option.value);
    option.hidden = experimental && !$("#apple-experiments").checked;
    option.disabled = !option.value || option.hidden;
  }
  $("#apple-execution-apply-field").hidden = !apple;
  $("#apple-execution-mode-note").hidden = !apple;
  $("#apple-execution-load-status").hidden = !apple;
  $("#apple-execution-mode-note").textContent = apple ? appleExecutionModes[mode] || editingProfileConfig().embedding?.native?.executionModeIssue || "请选择计算模式" : "";
  $("#embedding-base-url-field").hidden = apple;
  $("#apple-native-config").hidden = !apple;
  $("#embedding-space").disabled = apple;
  $("#embedding-space").placeholder = apple
    ? "由模型指纹自动确定，禁止手工混用"
    : "模型或供应商变化时应使用新的向量空间";
  const research = apple && (mode === "c" || mode === "d");
  $("#apple-private-kernel-fields").hidden = !research;
  $("#apple-private-recurrence-max-field").hidden = mode !== "d";
  $("#apple-private-recurrence-block-field").hidden = mode !== "d";
  $("#apple-recurrence-profile-field").hidden = !(apple && mode === "d");
}

function embeddingLoadMessage(status) {
  if (!status?.managed) return "正在切换到 Apple 原生后端";
  if (status.state === "starting") return "正在加载模型和编译缓存，完成前不会开始入库";
  if (status.state === "ready") return `后端已就绪 · ${appleExecutionModeLabels[status.executionMode] || "Apple 原生"}`;
  if (status.state === "failed") return `加载失败 · ${status.error || status.lastError || "请查看诊断日志"}`;
  if (status.state === "stopping") return "正在停止旧后端";
  return `后端状态：${status.state || "准备中"}`;
}

async function waitForEmbeddingReady(timeoutMs = 360_000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const status = await api("/api/embedding-backend");
    $("#apple-execution-load-status").textContent = embeddingLoadMessage(status);
    if (status.state === "ready") return status;
    if (status.state === "failed") throw new Error(status.error || status.lastError || "Apple 原生后端加载失败");
    // A constructor/preflight failure happens before a managed helper exists,
    // so the control plane reports `remote` instead of `failed`. Give a valid
    // restart a short reconciliation window, then surface the failure instead
    // of leaving the Apply button spinning for the full six-minute timeout.
    if (status.state === "remote" && Date.now() - startedAt >= 2_000) {
      throw new Error("Apple 原生后端未能启动，请检查模型和运行时路径");
    }
    await sleep(750);
  }
  throw new Error("Apple 原生后端加载超时");
}

async function applyExecutionMode() {
  const id = state.editingProfileId;
  if (!id) return;
  const previousConfig = structuredClone(state.config || {});
  const previousProfile = previousConfig.profiles?.[id];
  const previousMode = String(previousProfile?.embedding?.native?.executionMode || "b");
  const button = $("#apply-execution-mode");
  const save = $("#save-profile");
  button.disabled = true;
  save.disabled = true;
  button.classList.add("loading");
  button.textContent = "正在应用…";
  $("#apple-execution-load-status").hidden = false;
  try {
    if (state.scanning) {
      $("#apple-execution-load-status").textContent = "正在停止当前入库，避免一轮任务混用两种计算模式";
      await api("/api/assets/libraries/stop", { method: "POST", body: JSON.stringify({}) });
      for (let attempt = 0; attempt < 120; attempt += 1) {
        const status = await api("/api/assets/libraries");
        if (!(status.libraries || []).some((library) => ["queued", "running"].includes(library.job?.state))) break;
        await sleep(500);
      }
    }
    $("#apple-execution-load-status").textContent = "正在停止旧后端并保存计算模式";
    const saved = await api(`/api/profiles/${encodeURIComponent(id)}?defer=true`, {
      method: "PUT",
      body: JSON.stringify(profilePayload(id)),
    });
    if (id !== saved.activeProfile) {
      state.config = await api(`/api/profiles/${encodeURIComponent(id)}/activate`, { method: "POST" });
    } else {
      state.config = saved;
      await waitForEmbeddingReady();
    }
    renderProfileSettings();
    $("#apple-execution-load-status").hidden = false;
    $("#apple-execution-load-status").textContent = `已应用 · ${appleExecutionModeLabels[$("#apple-execution-mode").value]}`;
    toast("计算模式已加载，可以开始入库");
    await Promise.all([checkStatus(), refresh()]);
  } catch (error) {
    let recovered = false;
    if (previousProfile && previousConfig.activeProfile === id) {
      $("#apple-execution-load-status").textContent = `应用失败，正在恢复“${appleExecutionModeLabels[previousMode] || "原计算模式"}”`;
      try {
        state.config = await api(`/api/profiles/${encodeURIComponent(id)}?defer=true`, {
          method: "PUT",
          body: JSON.stringify(previousProfile),
        });
        await waitForEmbeddingReady();
        recovered = true;
        renderProfileSettings();
      } catch (recoveryError) {
        $("#apple-execution-load-status").textContent = `应用失败，恢复原模式也失败 · ${recoveryError.message}`;
      }
    }
    if (recovered) {
      $("#apple-execution-load-status").hidden = false;
      $("#apple-execution-load-status").textContent = `应用失败，已恢复“${appleExecutionModeLabels[previousMode] || "原计算模式"}” · ${error.message}`;
    } else if (!$("#apple-execution-load-status").textContent.includes("恢复原模式也失败")) {
      $("#apple-execution-load-status").textContent = `应用失败 · ${error.message}`;
    }
    toast(error.message);
  } finally {
    button.disabled = false;
    save.disabled = false;
    button.classList.remove("loading");
    button.textContent = "应用计算模式";
    renderProfileActions();
  }
}

function renderProfileSettings({ resetDirty = true } = {}) {
  const config = state.config || {};
  const entries = localProfileEntries();
  const activeId = config.activeProfile || "";
  if (!state.editingProfileId || !entries.some(([id]) => id === state.editingProfileId)) {
    state.editingProfileId = entries.some(([id]) => id === activeId) ? activeId : entries[0]?.[0] || "";
  }
  const id = state.editingProfileId;
  const profile = editingProfileConfig();
  const storage = profile.storage || {};
  const embedding = profile.embedding || {};
  const native = embedding.native || {};
  $("#profile-select").innerHTML = entries.map(([profileId, item]) =>
    `<option value="${escapeHtml(profileId)}"${profileId === id ? " selected" : ""}>${escapeHtml(item.label || profileId)}${profileId === activeId ? " · 已启用" : ""}</option>`).join("");
  $("#profile-delete").disabled = entries.length <= 1 || id === activeId;
  $("#profile-delete").title = id === activeId ? "请先启用另一个档案，再删除当前档案" : "删除这个配置档案，不删除数据库";
  $("#profile-label").value = profile.label || id;
  $("#embedding-provider").value = embedding.provider === "apple-native" ? "apple-native" : "remote";
  $("#embedding-base-url").value = embedding.baseUrl || "";
  $("#embedding-model").value = embedding.provider === "apple-native"
    ? appleEmbeddingModel(embedding.dimension)
    : embedding.model || "";
  $("#embedding-dimension").value = Number(embedding.dimension || 0) || "";
  $("#embedding-input-style").value = ["auto", "wemm", "qwen"].includes(embedding.inputStyle) ? embedding.inputStyle : "auto";
  $("#apple-execution-mode").value = APPLE_EXECUTION_MODES.includes(native.executionMode) ? native.executionMode : "";
  $("#apple-experiments").checked = ["a", "c", "d"].includes(native.executionMode);
  $("#apple-model-package").value = native.modelPackage || "";
  $("#apple-native-binary").value = native.binary || "";
  $("#apple-coreml-cache").value = native.coreMLCache || "";
  $("#apple-recurrence-profile").value = native.privateANE?.recurrenceProfile || "";
  $("#apple-private-sequence-length").value = String(native.privateANE?.sequenceLength || 2112);
  $("#apple-private-mlp-fraction").value = Number(native.privateANE?.mlpFraction ?? 0.75);
  $("#apple-private-mlp-max-layers").value = Number(native.privateANE?.mlpMaxLayers || 24);
  $("#apple-private-mlp-variant").value = Number(native.privateANE?.mlpVariant || 8);
  $("#apple-private-recurrence-max-tokens").value = String(native.privateANE?.recurrenceMaxTokens || 8192);
  $("#apple-private-recurrence-block-size").value = String(native.privateANE?.recurrenceBlockSize || 8);
  $("#embedding-space").value = embedding.provider === "apple-native" ? "" : profile.spaceId || profile.resolvedSpaceId || "";
  $("#local-storage-path").value = storage.path || "";
  $("#resolved-storage-path").textContent = profile.resolvedStoragePath || "保存后由服务器解析";
  $("#config-path").textContent = config.configPath || "config.json";
  $("#performance-history-path").textContent = config.performanceHistoryPath || "";
  $("#profile-id").textContent = id;
  const active = activeProfileConfig();
  $("#active-profile-label").textContent = active.label || activeId || "未启用";
  $("#active-profile-summary").textContent = active.storage?.provider === "aliyun"
    ? "当前档案不是本地 zvec 档案"
    : `${active.embedding?.provider === "apple-native" ? appleExecutionModeLabels[String(active.embedding?.native?.executionMode || "b")] || "Apple 原生" : active.embedding?.model || "未配置模型"}`;
  renderAppleBackendFields();
  if (resetDirty) state.profileDirty = false;
  $("#profile-test-status").textContent = "";
  renderProfileActions();
}

function profilePayload(profileId, creating = false) {
  const current = editingProfileConfig();
  const provider = $("#embedding-provider").value === "apple-native" ? "apple-native" : "remote";
  const apple = provider === "apple-native";
  const dimension = Math.max(1, Math.floor(Number($("#embedding-dimension").value || 0)));
  const model = apple ? appleEmbeddingModel(dimension) : $("#embedding-model").value.trim();
  const inputStyle = $("#embedding-input-style").value;
  const baseUrl = $("#embedding-base-url").value.trim().replace(/\/+$/, "");
  if (!apple && !baseUrl) throw new Error("请填写 Embedding 服务地址");
  if (!model) throw new Error("请填写 Embedding 模型");
  if (!Number.isFinite(dimension)) throw new Error("向量维度无效");
  if (apple && !$("#apple-execution-mode").value) throw new Error("请明确选择 Apple 计算模式，旧 E 模式不会自动切换");
  const executionMode = requireAppleExecutionMode({ executionMode: $("#apple-execution-mode").value || "b" });
  const recurrenceProfile = $("#apple-recurrence-profile").value.trim();
  const privateSequenceLength = Number($("#apple-private-sequence-length").value || 2112);
  const privateMLPFraction = Number($("#apple-private-mlp-fraction").value || 0.75);
  const privateMLPMaxLayers = Number($("#apple-private-mlp-max-layers").value || 24);
  const privateMLPVariant = Number($("#apple-private-mlp-variant").value || 8);
  const privateRecurrenceMaxTokens = Number($("#apple-private-recurrence-max-tokens").value || privateSequenceLength);
  const privateRecurrenceBlockSize = Number($("#apple-private-recurrence-block-size").value || 8);
  // Empty advanced paths intentionally select the managed INDEXED_HOME layout.
  // The server performs the authoritative preflight and returns the exact missing asset.
  const embeddingChanged = creating
    || provider !== String(current.embedding?.provider || "remote")
    || baseUrl !== String(current.embedding?.baseUrl || "").replace(/\/+$/, "")
    || model !== String(current.embedding?.model || "")
    || dimension !== Number(current.embedding?.dimension || 0)
    || inputStyle !== String(current.embedding?.inputStyle || "auto");
  const shownSpace = $("#embedding-space").value.trim();
  const inheritedSpace = String(current.spaceId || current.resolvedSpaceId || "");
  const spaceId = apple ? "" : embeddingChanged && (!shownSpace || shownSpace === inheritedSpace)
    ? safeSpaceId(model, dimension, inputStyle, profileId)
    : shownSpace;
  const storage = {
    provider: "local",
    path: $("#local-storage-path").value.trim(),
  };
  return {
    label: $("#profile-label").value.trim() || profileId,
    spaceId,
    embedding: {
      provider,
      baseUrl: apple ? "" : baseUrl,
      model,
      dimension,
      inputStyle: apple ? "wemm" : inputStyle,
      ...(apple ? {
        native: {
          ...(current.embedding?.native || {}),
          binary: $("#apple-native-binary").value.trim(),
          modelPackage: $("#apple-model-package").value.trim(),
          coreMLCache: $("#apple-coreml-cache").value.trim(),
          executionMode,
          privateANE: {
            ...(current.embedding?.native?.privateANE || {}),
            sequenceLength: privateSequenceLength,
            mlpFraction: privateMLPFraction,
            mlpMaxLayers: privateMLPMaxLayers,
            mlpVariant: privateMLPVariant,
            recurrenceMaxTokens: privateRecurrenceMaxTokens,
            recurrenceBlockSize: privateRecurrenceBlockSize,
            recurrenceLayerSlots: executionMode === "d" ? [0] : (current.embedding?.native?.privateANE?.recurrenceLayerSlots || [0]),
            recurrenceIODtype: "fp16",
            recurrenceProfile,
          },
        },
      } : {}),
    },
    storage,
  };
}

function markProfileDirty(event) {
  state.profileDirty = true;
  $("#profile-test-status").textContent = "";
  if (event?.target?.id === "local-storage-path") $("#resolved-storage-path").textContent = "保存后由服务器解析";
  renderProfileActions();
}

async function loadFacts(force = false) {
  if (state.profileDirty && !force) return;
  const request = profileLoads.begin();
  try {
    const config = await api("/api/config", { signal: request.signal });
    if (!request.current() || (state.profileDirty && !force)) return;
    state.config = config;
    renderProfileSettings();
    if (state.view === "settings") renderDirectories();
  } catch (error) {
    if (!request.current()) return;
    $("#profile-note").textContent = error.message;
  }
}

async function saveProfile() {
  const id = state.editingProfileId;
  if (!id) return;
  const button = $("#save-profile");
  button.disabled = true;
  try {
    const saved = await api(`/api/profiles/${encodeURIComponent(id)}`, {
      method: "PUT",
      body: JSON.stringify(profilePayload(id)),
    });
    state.config = id === saved.activeProfile
      ? saved
      : await api(`/api/profiles/${encodeURIComponent(id)}/activate`, { method: "POST" });
    renderProfileSettings();
    $("#profile-note").textContent = "已保存并启用";
    await Promise.all([checkStatus(true), refresh()]);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

function selectProfileForEditing() {
  const id = $("#profile-select").value;
  if (!id || id === state.editingProfileId) return;
  if (state.profileDirty && !confirm("放弃当前档案中未保存的修改？")) {
    $("#profile-select").value = state.editingProfileId;
    return;
  }
  state.editingProfileId = id;
  renderProfileSettings();
}

async function createProfile() {
  if (state.profileDirty && !confirm("新建档案会放弃当前未保存修改，继续？")) return;
  const id = String(prompt("新档案 ID（小写英文、数字、短横线）") || "").trim();
  if (!id) return;
  if (!/^[a-z0-9][a-z0-9-]{1,48}$/.test(id)) return toast("档案 ID 格式不正确");
  if (state.config?.profiles?.[id]) return toast("这个档案已经存在");
  try {
    const source = editingProfileConfig();
    const embedding = source.embedding || {};
    state.config = await api(`/api/profiles/${encodeURIComponent(id)}`, {
      method: "PUT",
      body: JSON.stringify({
        label: id,
        spaceId: embedding.provider === "apple-native" ? "" : safeSpaceId(embedding.model, embedding.dimension, embedding.inputStyle, id),
        embedding: {
          provider: embedding.provider || "remote",
          baseUrl: embedding.baseUrl || "",
          model: embedding.model || "",
          dimension: Number(embedding.dimension || 0),
          inputStyle: embedding.inputStyle || "auto",
          native: embedding.native || {},
        },
        storage: { provider: "local", path: source.storage?.path || "" },
        video: source.video || {},
      }),
    });
    state.editingProfileId = id;
    renderProfileSettings();
    toast(`已创建 ${id}；确认配置后再启用`);
  } catch (error) { toast(error.message); }
}

async function deleteProfile() {
  const id = state.editingProfileId;
  if (!id || id === state.config?.activeProfile || localProfileEntries().length <= 1) return;
  if (!confirm(`删除配置档案“${id}”？向量数据库不会被删除。`)) return;
  try {
    state.config = await api(`/api/profiles/${encodeURIComponent(id)}`, { method: "DELETE" });
    state.editingProfileId = state.config.activeProfile;
    renderProfileSettings();
    toast("已删除档案");
  } catch (error) { toast(error.message); }
}

async function testProfile() {
  const id = state.editingProfileId;
  if (!id) return;
  const button = $("#test-profile");
  button.disabled = true;
  $("#profile-test-status").textContent = "正在连接";
  try {
    const result = await api("/api/profiles/test", {
      method: "POST",
      body: JSON.stringify({ id, profile: profilePayload(id) }),
    });
    if (result.embeddingSpace) {
      const field = $("#embedding-space");
      const current = editingProfileConfig();
      const inherited = String(current.spaceId || current.resolvedSpaceId || "");
      if (!field.value.trim() || field.value.trim() === inherited) {
        field.value = result.embeddingSpace;
        state.profileDirty = true;
      }
    }
    $("#profile-test-status").textContent = "连接正常";
    $("#profile-test-status").title = `${result.model || ""} · ${result.embeddingSpace || ""}`;
    renderProfileActions();
  } catch (error) {
    $("#profile-test-status").textContent = `连接失败 · ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

/* -------------------------------------------------------------------------- */
/* Server directory picker                                                     */
/* -------------------------------------------------------------------------- */

async function openBrowse(path = "", purpose = "library") {
  state.browsePurpose = purpose;
  state.browseKindsInitialized = false;
  $(".picker-kinds").hidden = purpose === "test";
  openDialog($("#picker"), $("#picker-input"), $("#pick-directory"));
  await browseTo(path || state.browsePath || "");
}

function closeBrowse() {
  directoryLoads.cancel();
  closeDialog($("#picker"));
}

async function browseTo(path) {
  const request = directoryLoads.begin();
  const note = $("#picker-note");
  note.textContent = "读取中";
  state.browsePath = "";
  $("#picker-choose").disabled = true;
  try {
    const data = await api(`/api/assets/directories/browse?path=${encodeURIComponent(path || "")}`, { signal: request.signal });
    if (!request.current()) return;
    state.browsePath = data.path;
    renderBrowse(data);
  } catch (error) {
    if (!request.current()) return;
    note.textContent = error.message;
    $("#picker-list").innerHTML = `<li class="picker-empty">${escapeHtml(error.message)}</li>`;
  }
}

function renderBrowse(data) {
  $("#picker-input").value = data.path;
  state.browseParent = data.parent || "";
  $("#picker-up").disabled = !data.parent;
  if (!state.browseKindsInitialized) {
    state.trackedKinds = (data.kinds || ["video", "image", "document"]).filter((kind) => assetKinds[kind]);
    state.documentFormats = data.documentFormats || [];
    state.browseKindsInitialized = true;
  }
  renderTrackedKinds();
  $("#picker-list").innerHTML = data.entries.length ? data.entries.map((entry) => `
    <li><button class="picker-row" data-path="${escapeHtml(entry.path)}" type="button">
      <span class="dir-name">${escapeHtml(entry.name)}</span>
      ${entry.registered ? `<span class="dir-badge">已添加</span>` : entry.containsLibrary ? `<span class="dir-badge dim">含素材目录</span>` : ""}
    </button></li>`).join("") : `<li class="picker-empty">这个目录下没有子目录</li>`;
  $("#picker-note").textContent = data.truncated
    ? `只显示前 ${number(data.entries.length)} 个子目录`
    : `${number(data.entries.length)} 个子目录`;
  $$("#picker-list .picker-row").forEach((row) => row.addEventListener("click", () => void browseTo(row.dataset.path)));
  const registered = (data.roots || []).includes(data.path);
  $("#picker-choose").textContent = state.browsePurpose === "test" ? "选择此目录" : registered ? "保存并重新扫描" : "添加并扫描";
  $("#picker-choose").disabled = !state.browsePath || !state.trackedKinds.length;
}

function renderTrackedKinds() {
  $$("#picker-kind-chips .kind-option").forEach((button) => {
    button.setAttribute("aria-pressed", String(state.trackedKinds.includes(button.dataset.kind)));
  });
  const formats = $("#document-formats");
  formats.innerHTML = state.trackedKinds.includes("document") ? state.documentFormats.map((format) => `
    <span class="format-note${format.available ? "" : " unavailable"}" title="${escapeHtml(format.note)}">
      <b>${escapeHtml(format.label)}</b>${escapeHtml(format.id === "pdf" ? "有文本层" : (format.extensions || []).join(" · "))}${format.available ? "" : " · 未就绪"}
    </span>`).join("") : "";
  $("#picker-choose").disabled = !state.browsePath || !state.trackedKinds.length;
}

function toggleTrackedKind(button) {
  const kind = button.dataset.kind;
  if (!assetKinds[kind]) return;
  if (state.trackedKinds.includes(kind)) {
    if (state.trackedKinds.length === 1) return toast("至少保留一种文件类型");
    state.trackedKinds = state.trackedKinds.filter((item) => item !== kind);
  } else {
    state.trackedKinds = ["video", "image", "document"].filter((item) => item === kind || state.trackedKinds.includes(item));
  }
  renderTrackedKinds();
}

async function chooseCurrentDirectory() {
  const target = state.browsePath;
  if (!target) return;
  if (state.browsePurpose === "test") { $("#test-directory").value = target; closeBrowse(); return; }
  const registered = state.libraries.some((library) => library.path === target);
  const button = $("#picker-choose");
  button.disabled = true;
  try {
    const result = await api("/api/assets/libraries", {
      method: "POST",
      body: JSON.stringify({ path: target, kinds: state.trackedKinds }),
    });
    toast(registered
      ? `已更新 ${result.library?.name || target} 的追踪类型`
      : result.scanStarted ? `已添加 ${result.library?.name || target}，开始扫描` : `已添加 ${result.library?.name || target}`);
    closeBrowse();
    await refresh();
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

/* -------------------------------------------------------------------------- */
/* Wiring                                                                      */
/* -------------------------------------------------------------------------- */

function searchSetup() {
  const persisted = (state.health?.storage?.collections || []).reduce((sum, item) => sum + Number(item.count || 0), 0);
  return setupPresentation({ ...state.readiness, directories: state.libraries.length,
    indexed: Math.max(persisted, Number(state.coverage?.totals?.cached || 0), ...(state.libraries || []).map(item => Number(item.assetCount || 0)), state.health?.storageProvider === "aliyun" ? 1 : 0),
    scanning: state.scanning });
}

function renderReadiness() {
  const view = searchSetup();
  $("#setup-guide").hidden = view.searchable && !state.scanning;
  $("#setup-title").textContent = view.title;
  $("#setup-detail").textContent = view.detail;
  $("#setup-action").textContent = view.label;
  $("#setup-action").dataset.action = view.action;
  $("#setup-action").hidden = !view.action;
  $("#asset-search-query").disabled = !view.searchable;
  $("#asset-search-form button[type=submit]").disabled = !view.searchable;
}

async function checkStatus(force = false) {
  if (state.healthChecking && force !== true) return;
  state.healthChecking = true;
  const request = healthChecks.begin();
  const pill = $("#connection-pill");
  let online = false;
  try {
    const config = await api("/api/config", { signal: AbortSignal.any([request.signal, AbortSignal.timeout(10_000)]) });
    if (!request.current()) return;
    online = true;
    if (!state.profileDirty) state.config = config;
    const embedding = config.profiles?.[config.activeProfile]?.embedding || {};
    const native = embedding.provider === "apple-native";
    if (!native && !embedding.baseUrl) {
      state.readiness = { phase: "configure", detail: "选择 Apple 原生模型，或填写外部 Embedding 服务地址。" };
    } else if (native && embedding.native?.executionModeIssue) {
      state.readiness = { phase: "configure", detail: embedding.native.executionModeIssue };
    } else {
      const backend = native ? await api("/api/embedding-backend", { signal: AbortSignal.any([request.signal, AbortSignal.timeout(10_000)]) }) : null;
      if (!request.current()) return;
      if (["starting", "restarting"].includes(backend?.state)) {
        state.readiness = { phase: "loading", detail: "正在加载模型和编译缓存，可以到设置中查看进度。" };
      } else {
        if (backend?.state === "failed") throw new Error(backend.lastError || backend.error || "模型加载失败");
        const health = await api("/api/status", { signal: AbortSignal.any([request.signal, AbortSignal.timeout(15_000)]) });
        if (!request.current()) return;
        state.health = health;
        state.readiness = { phase: "ready", detail: "" };
      }
    }
    pill.className = "status-pill online";
    pill.querySelector("span").textContent = "本机服务在线";
  } catch (error) {
    if (!request.current()) return;
    state.readiness = { phase: online ? "error" : "offline", detail: error.message };
    pill.className = online ? "status-pill checking" : "status-pill error";
    pill.querySelector("span").textContent = online ? "本机服务在线 · 待配置" : "本机服务离线";
  } finally {
    if (request.current()) {
      state.healthChecking = false;
      renderReadiness();
    }
  }
}

function handleShortcuts(event) {
  if (event.key === "Escape") {
    if (!$("#picker").hidden) return closeBrowse();
    if (!$("#asset-modal").hidden) return closeAssetModal();
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s" && !$("#asset-modal").hidden) {
    event.preventDefault();
    void saveAssetDocument();
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s"
    && state.view === "settings" && state.settingsTab === "engine" && state.profileDirty) {
    event.preventDefault();
    void saveProfile();
    return;
  }
  if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "")) return;
  event.preventDefault();
  if (state.view !== "search") showView("search");
  $("#asset-search-query").focus();
}

async function init() {
  $("#brand-home").addEventListener("click", (event) => {
    event.preventDefault();
    showView("search");
  });
  $("#view-toggle").addEventListener("click", () => showView(state.view === "settings" ? "search" : "settings"));
  $$("[data-settings-tab]").forEach((button) => button.addEventListener("click", () => showSettingsTab(button.dataset.settingsTab)));
  $("#connection-pill").addEventListener("click", checkStatus);
  $("#asset-search-form").addEventListener("submit", runSearch);
  $("#asset-search-query").addEventListener("input", () => { if (!$("#asset-search-query").value.trim()) void runSearch(); });
  $$("#asset-kind-chips .chip").forEach((chip) => chip.addEventListener("click", () => selectAssetKind(chip)));
  $("#asset-modal-close").addEventListener("click", closeAssetModal);
  $("#asset-modal-backdrop").addEventListener("click", closeAssetModal);
  $("#asset-copy-path").addEventListener("click", () => void copyAssetPath());
  $("#asset-save").addEventListener("click", () => void saveAssetDocument());
  $("#pick-directory").addEventListener("click", () => void openBrowse(""));
  $("#test-pick-directory").addEventListener("click", () => void openBrowse($("#test-directory").value, "test"));
  $("#test-start").addEventListener("click", async () => {
    $("#test-start").disabled = true;
    try {
      state.ingestTest = await api("/api/ingest-test/start", { method: "POST", body: JSON.stringify({ path: $("#test-directory").value.trim(), limit: Number($("#test-limit").value), repeats: Number($("#test-repeats").value) }) });
      renderIngestTest(state.ingestTest); await refresh();
    } catch (error) { toast(error.message); $("#test-start").disabled = false; }
  });
  $("#test-stop").addEventListener("click", async () => {
    $("#test-stop").disabled = true;
    try { state.ingestTest = await api("/api/ingest-test/stop", { method: "POST", body: "{}" }); renderIngestTest(state.ingestTest); }
    catch (error) { toast(error.message); }
  });
  $("#scan-engine-settings").addEventListener("click", () => showSettingsTab("engine"));
  $("#picker-close").addEventListener("click", closeBrowse);
  $("#picker-backdrop").addEventListener("click", closeBrowse);
  $("#picker-choose").addEventListener("click", () => void chooseCurrentDirectory());
  $$("#picker-kind-chips .kind-option").forEach((button) => button.addEventListener("click", () => toggleTrackedKind(button)));
  $("#picker-up").addEventListener("click", () => { if (state.browseParent) void browseTo(state.browseParent); });
  $("#picker-go").addEventListener("click", () => void browseTo($("#picker-input").value.trim()));
  $("#picker-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void browseTo(event.target.value.trim());
    }
  });
  $("#save-scan").addEventListener("click", () => void saveScanSettings());
  $$("#auto-scan, #scan-interval, #scan-limit, #scan-discovery-limit").forEach((input) => input.addEventListener("input", () => { state.scanSettingsDirty = true; }));
  $("#scan-all").addEventListener("click", () => void scanAll(false));
  $("#background-scan-all").addEventListener("click", () => void scanAll(true));
  $("#profile-select").addEventListener("change", selectProfileForEditing);
  $("#profile-new").addEventListener("click", () => void createProfile());
  $("#profile-delete").addEventListener("click", () => void deleteProfile());
  $("#test-profile").addEventListener("click", () => void testProfile());
  $("#save-profile").addEventListener("click", () => void saveProfile());
  $("#apply-execution-mode").addEventListener("click", () => void applyExecutionMode());
  $("#setup-action").addEventListener("click", () => {
    const action = $("#setup-action").dataset.action;
    if (action === "retry") return void checkStatus();
    showView("settings");
    showSettingsTab(action === "engine" ? "engine" : "assets");
    if (action === "add") void openBrowse();
  });
  $("#developer-tools").addEventListener("change", event => {
    state.developer = event.target.checked;
    $("[data-settings-tab=test]").hidden = !state.developer;
    if (!state.developer && state.settingsTab === "test") showSettingsTab("assets");
  });
  $("#apple-experiments").addEventListener("change", () => {
    if (!$("#apple-experiments").checked && ["a", "c", "d"].includes($("#apple-execution-mode").value)) {
      $("#apple-experiments").checked = true;
      toast("请先选择稳定模式 B，再收起实验模式");
    }
    renderAppleBackendFields();
  });
  $("#profile-panel").addEventListener("input", (event) => {
    if (event.target.id === "apple-experiments") return;
    if (event.target.matches("input, select, textarea") && event.target.id !== "profile-select") {
      if (event.target.id === "embedding-dimension" && $("#embedding-provider").value === "apple-native") {
        $("#embedding-model").value = appleEmbeddingModel(event.target.value);
      }
      markProfileDirty(event);
    }
  });
  $("#profile-panel").addEventListener("change", (event) => {
    if (event.target.matches("select") && event.target.id !== "profile-select") {
      if (event.target.id === "embedding-provider") {
        const apple = event.target.value === "apple-native";
        if (apple) {
          $("#embedding-dimension").value = String(APPLE_EMBEDDING_DEFAULT_DIMENSION);
          $("#embedding-model").value = appleEmbeddingModel(APPLE_EMBEDDING_DEFAULT_DIMENSION);
          $("#embedding-input-style").value = "wemm";
          $("#embedding-space").value = "";
        }
      }
      if (event.target.id === "embedding-provider" || event.target.id === "apple-execution-mode") renderAppleBackendFields();
      if (event.target.id === "apple-execution-mode") {
        if (["c", "d"].includes(event.target.value)) {
          $("#apple-private-sequence-length").value = "2112";
          $("#apple-private-mlp-fraction").value = "0.75";
          $("#apple-private-mlp-max-layers").value = "24";
          $("#apple-private-recurrence-max-tokens").value = "8192";
          $("#apple-private-recurrence-block-size").value = "8";
        }
        $("#apple-execution-load-status").hidden = false;
        $("#apple-execution-load-status").textContent = "待应用 · 当前模式尚未改变";
      }
      markProfileDirty(event);
    }
  });
  document.addEventListener("keydown", handleShortcuts);
  window.addEventListener("beforeunload", (event) => {
    if (!state.profileDirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearTimeout(state.timer);
    else void refresh();
  });
  await checkStatus();
  await refresh();
  const params = new URLSearchParams(location.search);
  const initialQuery = String(params.get("q") || "");
  const initialKind = String(params.get("kind") || "");
  state.settingsTab = ["engine", "test"].includes(params.get("tab")) ? params.get("tab") : "assets";
  if (initialQuery) $("#asset-search-query").value = initialQuery;
  const initialChip = $$("#asset-kind-chips .chip").find((item) => item.dataset.kind === initialKind);
  if (initialQuery && initialChip) {
    state.assetKind = initialChip.dataset.kind || "";
    $$("#asset-kind-chips .chip").forEach((item) => item.classList.toggle("active", item === initialChip));
  }
  showView(params.get("view") || "search", false);
  showSettingsTab(state.settingsTab, false);
  // Restoring results is the difference between a search box and a search product.
  if (initialQuery) {
    state.query = initialQuery;
    await runSearch();
  }
}

init().catch((error) => toast(error.message));
