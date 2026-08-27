// @ts-nocheck -- behavior-preserving popup port.
import { DIRECT_DEFAULT_CONFIG, vectorEndpoint } from "@indexed/clients/direct-cloud";
import { PRODUCT_DEFAULTS } from "@indexed/contracts";

const DEFAULT_SEGMENT_SECONDS = PRODUCT_DEFAULTS.capture.segmentSeconds;

const elements = Object.fromEntries(
  [
    "connection",
    "video-count",
    "visual-count",
    "text-count",
    "queue-count",
    "cost-storage",
    "cost-query",
    "cost-size",
    "cost-storage-detail",
    "cost-query-detail",
    "cost-note",
    "source-summary",
    "history-toggle",
    "video-list",
    "history-empty",
    "retain-preview",
    "library",
    "local-library",
    "settings",
    "settings-panel",
    "backend-cloud",
    "backend-local",
    "cloud-settings",
    "local-settings",
    "control-plane-url",
    "sync-control-plane",
    "embedding-url",
    "embedding-model",
    "embedding-dimension",
    "oss-region",
    "oss-account-id",
    "oss-bucket",
    "oss-visual-index",
    "oss-transcript-index",
    "oss-access-key-id",
    "oss-access-key-secret",
    "oss-security-token",
    "reset-memory",
    "save-settings",
    "settings-status",
  ].map((id) => [id, document.getElementById(id)])
);
let activeTab = null;
let pageState = {};
let libraryState = { videos: [] };
let queueState = {};
let historyExpanded = false;
let refreshInFlight = false;
let savedAccessKeyId = "";
let hasSavedAccessKeySecret = false;
let savedMaskedAccessKeySecret = "";
let draftAccessKeyId = "";
let draftAccessKeySecret = "";
let validatedConfigFingerprint = "";
let validationInFlight = false;
let credentialAutoSaveTimer = 0;
let accessKeyIdAutoSaveTimer = 0;
let credentialSaveChain = Promise.resolve();
let backendMode = "cloud";
let resolvedEmbeddingInputStyle = "auto";
let resolvedEmbeddingSpace = "";
let discoveredEmbeddingUrl = "";
let embeddingDiscoveryTimer = 0;
let embeddingDiscoveryInFlight = false;

function setBackendMode(mode) {
  backendMode = mode === "local" ? "local" : "cloud";
  elements["backend-cloud"].classList.toggle("active", backendMode === "cloud");
  elements["backend-local"].classList.toggle("active", backendMode === "local");
  elements["cloud-settings"].hidden = backendMode !== "cloud";
  elements["local-settings"].hidden = backendMode !== "local";
  elements["cost-note"].textContent = backendMode === "local"
    ? "本地 LanceDB 不产生云存储费用。"
    : "写入每月前 20GB 免费；金额为当前规模估算。";
  invalidateCredentialValidation();
}

function maskCredential(value) {
  const text = String(value || "");
  if (!text) return "";
  if (text.length <= 4) return "*".repeat(text.length);
  return `${text.slice(0, 2)}${"*".repeat(text.length - 4)}${text.slice(-2)}`;
}

function setCredentialValue(element, value, masked = false) {
  element.value = value;
  element.dataset.masked = masked ? "true" : "false";
}

function showSavedCredentials({ accessKeyId, maskedAccessKeyId, maskedAccessKeySecret }) {
  savedAccessKeyId = String(accessKeyId || "");
  draftAccessKeyId = savedAccessKeyId;
  draftAccessKeySecret = "";
  savedMaskedAccessKeySecret = String(maskedAccessKeySecret || "");
  hasSavedAccessKeySecret = Boolean(maskedAccessKeySecret);
  setCredentialValue(
    elements["oss-access-key-id"],
    maskedAccessKeyId || maskCredential(savedAccessKeyId),
    Boolean(savedAccessKeyId)
  );
  setCredentialValue(
    elements["oss-access-key-secret"],
    savedMaskedAccessKeySecret,
    Boolean(savedMaskedAccessKeySecret)
  );
  elements["oss-access-key-secret"].placeholder = hasSavedAccessKeySecret
    ? "已保存；粘贴可轮转"
    : "粘贴后自动验证";
}

function invalidateCredentialValidation() {
  validatedConfigFingerprint = "";
  elements["save-settings"].textContent = "应用其他设置";
}

function message(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) return resolve({ error: chrome.runtime.lastError.message });
      resolve(response || {});
    });
  });
}

function tabMessage(messageValue) {
  return new Promise((resolve) => {
    if (!activeTab?.id) return resolve({ error: "没有活动标签页" });
    chrome.tabs.sendMessage(activeTab.id, messageValue, (response) => {
      if (chrome.runtime.lastError) return resolve({ error: chrome.runtime.lastError.message });
      resolve(response || {});
    });
  });
}

async function pageMessage(messageValue) {
  const response = await tabMessage(messageValue);
  if (!response.error) return response;
  try {
    await chrome.scripting.executeScript({
      target: { tabId: activeTab.id },
      files: ["content-script.js"],
    });
  } catch (error) {
    return { error: String(error.message || error) };
  }
  return tabMessage(messageValue);
}

function formatTime(seconds) {
  const value = Math.max(0, Math.floor(Number(seconds || 0)));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const rest = value % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `${minutes}:${String(rest).padStart(2, "0")}`;
}

function formatSize(gib) {
  const value = Math.max(0, Number(gib || 0));
  if (value >= 1) return `${value.toFixed(value >= 10 ? 1 : 2)} GB`;
  const mib = value * 1024;
  if (mib >= 1) return `${mib.toFixed(mib >= 10 ? 1 : 2)} MB`;
  return `${Math.max(0, mib * 1024).toFixed(1)} KB`;
}

function formatCny(value) {
  const amount = Math.max(0, Number(value || 0));
  if (amount === 0) return "¥0";
  if (amount < 0.0001) return "<¥0.0001";
  if (amount < 0.01) return `¥${amount.toFixed(4)}`;
  return `¥${amount.toFixed(2)}`;
}

function sourceName(site) {
  return String(site || "").toLowerCase() === "bilibili" ? "哔哩哔哩" : "YouTube";
}

function sourceUrlAt(video, segmentIndex = 0) {
  if (!video?.source_url) return "";
  try {
    const url = new URL(video.source_url);
    const seconds = Math.max(0, Number(segmentIndex || 0) * Number(video.segment_interval || DEFAULT_SEGMENT_SECONDS));
    url.searchParams.set("t", video.source_site === "bilibili" ? String(seconds) : `${seconds}s`);
    return url.href;
  } catch {
    return video.source_url;
  }
}

function videoIdentity(urlValue) {
  try {
    const url = new URL(urlValue);
    if (url.hostname.endsWith("bilibili.com")) {
      return `bilibili:${url.pathname.match(/\/video\/(BV[A-Za-z0-9]+|av\d+)/i)?.[1] || ""}`;
    }
    if (url.hostname.endsWith("youtube.com")) {
      return `youtube:${url.searchParams.get("v") || ""}`;
    }
  } catch {}
  return "";
}

function resumeSegmentIndex(video) {
  const known = [
    ...(video.saved_visual_segments || []),
    ...(video.pending_visual_segments || []),
    ...(video.saved_transcript_segments || []),
  ].map(Number).filter((value) => Number.isInteger(value) && value >= 0);
  if (!known.length) return Number(video.total_blocks || 0) > 0 ? 0 : -1;
  const next = Math.max(...known) + 1;
  return next < Number(video.total_blocks || 0) ? next : -1;
}

async function continueWatching(video, segmentIndex) {
  const url = sourceUrlAt(video, segmentIndex);
  if (!url) return;
  const stored = await chrome.storage.local.get({ videoIndexingOptIns: {} });
  const optIns = { ...(stored.videoIndexingOptIns || {}) };
  optIns[`${String(video.source_site || "youtube").toLowerCase()}:${video.video_id}`] = true;
  await chrome.storage.local.set({ videoIndexingOptIns: optIns, autoIndexWatched: false });
  await message({ type: "SET_PROCESSING_PAUSED", paused: false });
  const targetIdentity = `${String(video.source_site || "youtube").toLowerCase()}:${video.video_id}`;
  const tabs = await chrome.tabs.query({});
  const existing = tabs.find((tab) => videoIdentity(tab.url) === targetIdentity);
  if (existing?.id) {
    await chrome.tabs.update(existing.id, { active: true, url });
    if (existing.windowId) await chrome.windows.update(existing.windowId, { focused: true });
  } else {
    await chrome.tabs.create({ active: true, url });
  }
  window.close();
}

function renderBlocks(container, video) {
  const total = Math.max(0, Number(video?.total_blocks || 0));
  const saved = new Set((video?.saved_visual_segments || []).map(Number));
  const textOnly = new Set((video?.saved_transcript_segments || []).map(Number));
  const pending = new Set((video?.pending_visual_segments || []).map(Number));
  const processing = new Set((video?.processing_segments || []).map(Number));
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < total; index += 1) {
    const block = document.createElement("span");
    let state = "empty-block";
    let label = "空缺";
    if (saved.has(index)) [state, label] = ["saved", "画面已保存"];
    else if (processing.has(index)) [state, label] = ["processing", "正在处理"];
    else if (pending.has(index)) [state, label] = ["pending", "等待处理"];
    else if (textOnly.has(index)) [state, label] = ["text-only", "仅字幕"];
    block.className = `video-block ${state}`;
    block.title = `${formatTime(index * Number(video.segment_interval || DEFAULT_SEGMENT_SECONDS))} · ${label}`;
    fragment.appendChild(block);
  }
  container.replaceChildren(fragment);
}

function renderHistory() {
  const cutoff = Date.now() - 24 * 60 * 60 * 1000;
  const recentVideos = (libraryState.videos || []).filter((video) =>
    Number(video.last_processed_at || 0) >= cutoff
    || video.processing
    || video.pending_count > 0
  );
  const videos = historyExpanded ? recentVideos : recentVideos.slice(0, 3);
  const sources = recentVideos.reduce((counts, video) => {
    const site = sourceName(video.source_site);
    counts[site] = (counts[site] || 0) + 1;
    return counts;
  }, {});
  const sourceText = Object.entries(sources)
    .map(([site, count]) => `${site} ${count}`)
    .join(" · ") || "0 个视频";
  elements["source-summary"].textContent = sourceText;
  elements["history-toggle"].hidden = recentVideos.length <= 3;
  elements["history-toggle"].textContent = historyExpanded ? "收起" : `展开全部 ${recentVideos.length}`;
  elements["history-empty"].hidden = recentVideos.length > 0;
  const fragment = document.createDocumentFragment();
  for (const video of videos) {
    const card = document.createElement("article");
    card.className = `video-card${video.processing ? " is-processing" : ""}`;
    const top = document.createElement("div");
    top.className = "video-card-top";
    const badge = document.createElement("span");
    badge.className = `site-badge ${video.source_site === "bilibili" ? "bilibili" : "youtube"}`;
    badge.textContent = video.source_site === "bilibili" ? "B站" : "YT";
    const title = document.createElement("button");
    title.className = "video-card-title";
    title.textContent = video.title || "未命名视频";
    title.title = video.title || "未命名视频";
    title.disabled = !video.source_url;
    if (video.source_url) title.addEventListener("click", () => chrome.tabs.create({ url: video.source_url }));
    const resumeIndex = resumeSegmentIndex(video);
    let action;
    if (video.active_capture) {
      action = document.createElement("span");
      action.className = "video-state processing";
      action.textContent = video.capture_playing ? "正在记忆" : "等待播放";
    } else if (resumeIndex >= 0 && video.source_url) {
      action = document.createElement("button");
      action.className = "continue-button";
      const resumeTime = formatTime(resumeIndex * Number(video.segment_interval || DEFAULT_SEGMENT_SECONDS));
      action.textContent = `继续看 ${resumeTime}`;
      action.title = `跳到 ${resumeTime}，并继续记忆这个视频`;
      action.addEventListener("click", () => void continueWatching(video, resumeIndex));
    } else {
      action = document.createElement("span");
      const state = video.processing
        ? ["正在处理", "processing"]
        : video.pending_count
          ? ["队列中", "processing"]
          : ["已完成", "complete"];
      action.className = `video-state ${state[1]}`;
      action.textContent = state[0];
    }
    top.append(badge, title, action);
    const details = document.createElement("div");
    details.className = "video-card-details";
    const parts = [
      video.channel_name || sourceName(video.source_site),
      video.duration ? formatTime(video.duration) : "",
      `${video.visual_count} 画面`,
      `${video.transcript_count} 字幕`,
    ];
    if (video.processing) parts.push("正在处理");
    if (video.pending_count) parts.push(`${video.pending_count} 待处理`);
    details.textContent = parts.filter(Boolean).join(" · ");
    const blocks = document.createElement("div");
    blocks.className = "video-blocks history-blocks";
    renderBlocks(blocks, video);
    card.append(top, details, blocks);
    fragment.appendChild(card);
  }
  elements["video-list"].replaceChildren(fragment);
}

function renderDashboard() {
  const estimate = libraryState.billing_estimate || {};
  const monthlyStorage = formatCny(estimate.storage_month_cny);
  const thousandSearches = formatCny(estimate.text_search_1000_cny);
  elements["retain-preview"].checked = pageState?.retainPreview !== false;
  elements["video-count"].textContent = String(libraryState.video_count || 0);
  elements["visual-count"].textContent = String(libraryState.visual_count || 0);
  elements["text-count"].textContent = String(libraryState.transcript_count || 0);
  elements["queue-count"].textContent = String(
    Number(queueState.queued_total ?? libraryState.queued_count ?? 0)
    + Number(libraryState.capturing_count || 0)
  );
  elements["cost-storage"].textContent = `${monthlyStorage}/月`;
  elements["cost-query"].textContent = `千次搜索 · ${thousandSearches}`;
  elements["cost-size"].textContent = formatSize(estimate.estimated_gib);
  elements["cost-storage-detail"].textContent = `${monthlyStorage}/月`;
  elements["cost-query-detail"].textContent = thousandSearches;
  elements["cost-note"].textContent = Number(estimate.estimated_gib || 0) <= Number(estimate.free_write_gib_per_month || 20)
    ? "按当前数据量，写入仍在每月 20GB 免费额度内。"
    : `当前等量写入超出免费额度约 ${formatCny(estimate.current_write_cny_after_free)}。`;
  renderHistory();
}

async function refreshStats() {
  const [queue, library] = await Promise.all([
    message({ type: "QUEUE_STATUS" }),
    message({ type: "VIDEO_LIBRARY" }),
  ]);
  if (library.error) {
    const detail = String(library.error || "");
    elements.connection.textContent = detail.includes("Embedding") ? "模型未连接" : "云端未连接";
    elements.connection.title = detail;
    elements.connection.className = "connection error";
    return;
  }
  queueState = queue;
  libraryState = library;
  elements.connection.textContent = "已连接";
  elements.connection.className = "connection ok";
  renderDashboard();
}

async function refreshProgress() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    if (
      activeTab?.url?.startsWith("https://www.youtube.com/watch")
      || activeTab?.url?.startsWith("https://www.bilibili.com/video/")
    ) {
      const nextState = await pageMessage({ type: "GET_PAGE_STATE" });
      if (!nextState.error) pageState = nextState;
    }
    await refreshStats();
  } finally {
    refreshInFlight = false;
  }
}

async function initialize() {
  [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (
    activeTab?.url?.startsWith("https://www.youtube.com/watch")
    || activeTab?.url?.startsWith("https://www.bilibili.com/video/")
  ) {
    pageState = await pageMessage({ type: "GET_PAGE_STATE" });
  } else {
    pageState = {};
  }
  await refreshStats();
  const config = await message({ type: "GET_CONFIG" });
  setBackendMode(config.backendMode || DIRECT_DEFAULT_CONFIG.backendMode);
  elements["control-plane-url"].value = config.controlPlaneUrl || DIRECT_DEFAULT_CONFIG.controlPlaneUrl;
  elements["embedding-url"].value = config.embeddingBaseUrl || DIRECT_DEFAULT_CONFIG.embeddingBaseUrl;
  elements["embedding-model"].value = config.embeddingModel || DIRECT_DEFAULT_CONFIG.embeddingModel;
  elements["embedding-dimension"].value = config.embeddingModel
    ? String(config.embeddingDimension || DIRECT_DEFAULT_CONFIG.embeddingDimension)
    : "";
  resolvedEmbeddingInputStyle = config.embeddingInputStyle || DIRECT_DEFAULT_CONFIG.embeddingInputStyle;
  resolvedEmbeddingSpace = config.embeddingSpace || "";
  discoveredEmbeddingUrl = config.embeddingModel ? String(config.embeddingBaseUrl || "").replace(/\/+$/, "") : "";
  elements["oss-region"].value = config.ossRegion || DIRECT_DEFAULT_CONFIG.ossRegion;
  elements["oss-account-id"].value = config.ossAccountId || DIRECT_DEFAULT_CONFIG.ossAccountId;
  elements["oss-bucket"].value = config.ossBucket || DIRECT_DEFAULT_CONFIG.ossBucket;
  elements["oss-visual-index"].value = config.ossVisualIndex || DIRECT_DEFAULT_CONFIG.ossVisualIndex;
  elements["oss-transcript-index"].value = config.ossTranscriptIndex || DIRECT_DEFAULT_CONFIG.ossTranscriptIndex;
  showSavedCredentials({
    accessKeyId: config.ossAccessKeyId || DIRECT_DEFAULT_CONFIG.ossAccessKeyId,
    maskedAccessKeyId: config.maskedAccessKeyId,
    maskedAccessKeySecret: config.maskedAccessKeySecret,
  });
  elements["oss-security-token"].placeholder = config.hasSecurityToken
    ? "已保存；留空表示不修改"
    : "使用长期 RAM AK 时留空";
  if (backendMode === "cloud" && elements["embedding-url"].value.trim()) {
    await discoverEmbedding({ force: true });
    await refreshStats();
  }
  const cloudFieldsComplete = [
    elements["embedding-url"].value,
    elements["embedding-model"].value,
    Number(elements["embedding-dimension"].value) > 0 ? elements["embedding-dimension"].value : "",
    config.ossRegion,
    config.ossAccountId,
    config.ossBucket,
    config.ossVisualIndex,
    config.ossTranscriptIndex,
  ].every((value) => String(value || "").trim());
  const configurationComplete = backendMode === "local"
    ? Boolean(String(config.controlPlaneUrl || "").trim())
    : cloudFieldsComplete && Boolean(config.ossAccessKeyId) && Boolean(config.hasAccessKeySecret);
  if (!configurationComplete) {
    elements["settings-panel"].hidden = false;
    elements["settings-status"].textContent = backendMode === "local" ? "填写本地服务地址" : "补全阿里云与模型配置";
  }
}

function formConfig() {
  return {
    backendMode,
    controlPlaneUrl: elements["control-plane-url"].value.trim().replace(/\/+$/, ""),
    embeddingBaseUrl: elements["embedding-url"].value.trim().replace(/\/+$/, ""),
    embeddingModel: elements["embedding-model"].value.trim(),
    embeddingDimension: Number(elements["embedding-dimension"].value),
    embeddingInputStyle: resolvedEmbeddingInputStyle,
    embeddingSpace: resolvedEmbeddingSpace,
    ossRegion: elements["oss-region"].value.trim(),
    ossAccountId: elements["oss-account-id"].value.trim(),
    ossBucket: elements["oss-bucket"].value.trim(),
    ossVisualIndex: elements["oss-visual-index"].value.trim(),
    ossTranscriptIndex: elements["oss-transcript-index"].value.trim(),
    ossAccessKeyId: String(draftAccessKeyId || savedAccessKeyId).trim(),
    ossAccessKeySecret: draftAccessKeySecret,
    ossSecurityToken: elements["oss-security-token"].value,
  };
}

function configFingerprint(config) {
  return JSON.stringify(config);
}

async function discoverEmbedding({ force = false } = {}) {
  if (backendMode !== "cloud" || embeddingDiscoveryInFlight) return false;
  const baseUrl = elements["embedding-url"].value.trim().replace(/\/+$/, "");
  if (!baseUrl) return false;
  if (!force && baseUrl === discoveredEmbeddingUrl && elements["embedding-model"].value.trim()) return true;
  embeddingDiscoveryInFlight = true;
  elements["settings-status"].textContent = "正在识别向量模型…";
  try {
    const origin = `${new URL(baseUrl).origin}/*`;
    const allowed = await chrome.permissions.request({ origins: [origin] });
    if (!allowed) throw new Error("未获得向量服务访问权限");
    const discovered = await message({
      type: "DISCOVER_EMBEDDING",
      config: { embeddingBaseUrl: baseUrl, embeddingModel: "", embeddingDimension: 0, embeddingInputStyle: "auto", embeddingSpace: "" },
    });
    if (discovered.error || !discovered.ok) throw new Error(discovered.error || "无法识别向量模型");
    elements["embedding-model"].value = discovered.model;
    elements["embedding-dimension"].value = String(discovered.dimension);
    resolvedEmbeddingInputStyle = discovered.inputStyle || "auto";
    resolvedEmbeddingSpace = discovered.space || "";
    discoveredEmbeddingUrl = baseUrl;
    const saved = await message({
      type: "SAVE_CONFIG",
      config: {
        backendMode,
        embeddingBaseUrl: baseUrl,
        embeddingModel: discovered.model,
        embeddingDimension: Number(discovered.dimension),
        embeddingInputStyle: resolvedEmbeddingInputStyle,
        embeddingSpace: resolvedEmbeddingSpace,
      },
    });
    if (saved.error) throw new Error(saved.error);
    invalidateCredentialValidation();
    elements["settings-status"].textContent = `已识别 ${discovered.model} · ${discovered.dimension} 维`;
    return true;
  } catch (error) {
    discoveredEmbeddingUrl = "";
    resolvedEmbeddingInputStyle = "auto";
    resolvedEmbeddingSpace = "";
    elements["embedding-model"].value = "";
    elements["embedding-dimension"].value = "";
    elements["settings-status"].textContent = String(error.message || error);
    return false;
  } finally {
    embeddingDiscoveryInFlight = false;
  }
}

async function ensureConnectionPermissions(config) {
  try {
    const origins = config.backendMode === "local"
      ? [`${new URL(config.controlPlaneUrl).origin}/*`]
      : [`${new URL(config.embeddingBaseUrl).origin}/*`, `https://${vectorEndpoint(config)}/*`];
    const allowed = await chrome.permissions.request({ origins });
    if (!allowed) throw new Error(config.backendMode === "local" ? "未获得本地服务访问权限" : "未获得模型与阿里云访问权限");
  } catch (error) {
    elements["settings-status"].textContent = String(error.message || error);
    return false;
  }
  return true;
}

async function persistSettings(config, { automatic = false } = {}) {
  elements["save-settings"].disabled = true;
  elements["settings-status"].textContent = automatic
    ? "验证通过，正在自动保存…"
    : "验证通过，正在保存…";
  const saved = await message({ type: "SAVE_CONFIG", config });
  if (saved.error) {
    elements["settings-status"].textContent = `配置保存失败：${saved.error}`;
    return false;
  }
  if (config.backendMode !== "local") {
    showSavedCredentials({
      accessKeyId: config.ossAccessKeyId,
      maskedAccessKeyId: saved.maskedAccessKeyId,
      maskedAccessKeySecret: saved.maskedAccessKeySecret,
    });
  }
  if (automatic && [
    elements["oss-access-key-id"],
    elements["oss-access-key-secret"],
  ].includes(document.activeElement)) {
    document.activeElement.blur();
  }
  elements["oss-security-token"].value = "";
  validatedConfigFingerprint = "";
  elements["save-settings"].textContent = "应用其他设置";
  elements["settings-status"].textContent = "已连接";
  await refreshStats();
  return true;
}

async function validateSettings(config, {
  automatic = false,
  saveOnSuccess = false,
  alreadySaved = false,
} = {}) {
  if (validationInFlight) return false;
  validationInFlight = true;
  elements["save-settings"].disabled = true;
  elements["settings-status"].textContent = automatic
    ? "正在验证…"
    : (config.backendMode === "local" ? "正在连接本地服务…" : "正在连接阿里云…");
  try {
    if (!await ensureConnectionPermissions(config)) return false;
    const tested = await message({ type: "TEST_CONFIG", config });
    if (tested.error || !tested.ok) {
      validatedConfigFingerprint = "";
      elements["save-settings"].textContent = "应用其他设置";
      const detail = tested.error || "服务返回异常状态。";
      elements["settings-status"].textContent = alreadySaved
        ? `凭据已保存在本地，但验证失败：${detail}`
        : detail;
      await refreshStats();
      return false;
    }
    validatedConfigFingerprint = configFingerprint(config);
    const model = tested.embedding?.model
      || tested.embedding?.details?.data?.[0]?.id
      || "Embedding 在线";
    const storage = tested.storage?.provider_label || tested.storage?.provider || (config.backendMode === "local" ? "本地 LanceDB" : "向量库在线");
    if (saveOnSuccess) return persistSettings(config, { automatic: true });
    elements["save-settings"].textContent = "保存并连接";
    elements["settings-status"].textContent = alreadySaved
      ? `已保存并验证成功 · ${model} · ${storage}`
      : `验证成功 · ${model} · ${storage}`;
    return true;
  } finally {
    validationInFlight = false;
    elements["save-settings"].disabled = false;
  }
}

async function saveSettings() {
  if (backendMode === "cloud" && !await discoverEmbedding({ force: true })) return;
  const config = formConfig();
  if (validatedConfigFingerprint !== configFingerprint(config)) {
    const valid = await validateSettings(config);
    if (!valid) return;
  }
  await persistSettings(config);
}

function saveCredentialsLocally(config, {
  clearAccessKeySecret = false,
  validateAfterSave = false,
} = {}) {
  const captured = { ...config };
  credentialSaveChain = credentialSaveChain.then(async () => {
    elements["settings-status"].textContent = validateAfterSave
      ? "正在保存凭据…"
      : "正在保存 AccessKey ID…";
    const saved = await message({
      type: "SAVE_CONFIG",
      config: {
        ossAccessKeyId: captured.ossAccessKeyId,
        ossAccessKeySecret: captured.ossAccessKeySecret,
        clearAccessKeySecret,
      },
    });
    if (saved.error) {
      elements["settings-status"].textContent = `本地保存失败：${saved.error}`;
      return false;
    }
    showSavedCredentials({
      accessKeyId: captured.ossAccessKeyId,
      maskedAccessKeyId: saved.maskedAccessKeyId,
      maskedAccessKeySecret: saved.maskedAccessKeySecret,
    });
    if ([
      elements["oss-access-key-id"],
      elements["oss-access-key-secret"],
    ].includes(document.activeElement)) {
      document.activeElement.blur();
    }
    if (!validateAfterSave) {
      elements["settings-status"].textContent = saved.hasAccessKeySecret
        ? "AccessKey ID 已保存到当前设备"
        : "AccessKey ID 已保存，请粘贴对应 Secret";
      return true;
    }
    elements["settings-status"].textContent = "凭据已保存到当前设备，正在验证连接…";
    return validateSettings(captured, { automatic: true, alreadySaved: true });
  }).catch((error) => {
    elements["settings-status"].textContent = `本地保存失败：${String(error.message || error)}`;
    return false;
  });
  return credentialSaveChain;
}

function pasteText(event) {
  return String(event.clipboardData?.getData("text/plain") || "").trim();
}

function resetSecretForChangedAccessKey(nextAccessKeyId) {
  if (!nextAccessKeyId || nextAccessKeyId === savedAccessKeyId) return;
  draftAccessKeySecret = "";
  setCredentialValue(elements["oss-access-key-secret"], "", false);
  elements["oss-access-key-secret"].placeholder = "粘贴新 AccessKey 对应的 Secret";
}

elements["oss-access-key-id"].addEventListener("focus", (event) => {
  setCredentialValue(event.currentTarget, draftAccessKeyId || savedAccessKeyId, false);
  event.currentTarget.select();
});
elements["oss-access-key-secret"].addEventListener("focus", (event) => {
  window.clearTimeout(credentialAutoSaveTimer);
  draftAccessKeySecret = "";
  setCredentialValue(event.currentTarget, "", false);
});
elements["oss-access-key-id"].addEventListener("paste", (event) => {
  const value = pasteText(event);
  if (!value) return;
  event.preventDefault();
  draftAccessKeyId = value;
  setCredentialValue(elements["oss-access-key-id"], value, false);
  const changed = value !== savedAccessKeyId;
  resetSecretForChangedAccessKey(value);
  invalidateCredentialValidation();
  elements["settings-status"].textContent = !changed
    ? "AccessKey ID 未变化；可继续使用已保存的 Secret"
    : "已粘贴新的 AccessKey ID，请粘贴对应 Secret";
  void saveCredentialsLocally(formConfig(), { clearAccessKeySecret: changed });
});
elements["oss-access-key-secret"].addEventListener("paste", (event) => {
  const value = pasteText(event);
  if (!value) return;
  event.preventDefault();
  draftAccessKeySecret = value;
  setCredentialValue(elements["oss-access-key-secret"], value, false);
  invalidateCredentialValidation();
  void saveCredentialsLocally(formConfig(), { validateAfterSave: true });
});
elements["oss-access-key-id"].addEventListener("input", () => {
  elements["oss-access-key-id"].dataset.masked = "false";
  draftAccessKeyId = elements["oss-access-key-id"].value.trim();
  resetSecretForChangedAccessKey(draftAccessKeyId);
  invalidateCredentialValidation();
  window.clearTimeout(accessKeyIdAutoSaveTimer);
  if (!draftAccessKeyId) return;
  accessKeyIdAutoSaveTimer = window.setTimeout(() => {
    const changed = draftAccessKeyId !== savedAccessKeyId;
    void saveCredentialsLocally(formConfig(), { clearAccessKeySecret: changed });
  }, 450);
});
elements["oss-access-key-id"].addEventListener("blur", () => {
  window.clearTimeout(accessKeyIdAutoSaveTimer);
  const value = draftAccessKeyId || savedAccessKeyId;
  setCredentialValue(elements["oss-access-key-id"], maskCredential(value), Boolean(value));
  if (value && value !== savedAccessKeyId) {
    void saveCredentialsLocally(formConfig(), { clearAccessKeySecret: true });
  }
});
elements["oss-access-key-secret"].addEventListener("input", () => {
  elements["oss-access-key-secret"].dataset.masked = "false";
  draftAccessKeySecret = elements["oss-access-key-secret"].value;
  invalidateCredentialValidation();
  window.clearTimeout(credentialAutoSaveTimer);
  if (!draftAccessKeySecret.trim()) return;
  credentialAutoSaveTimer = window.setTimeout(() => {
    void saveCredentialsLocally(formConfig(), { validateAfterSave: true });
  }, 650);
});
elements["oss-access-key-secret"].addEventListener("blur", () => {
  window.clearTimeout(credentialAutoSaveTimer);
  const pendingSecret = draftAccessKeySecret;
  const canReuseSavedSecret = (draftAccessKeyId || savedAccessKeyId) === savedAccessKeyId
    && hasSavedAccessKeySecret;
  const masked = draftAccessKeySecret
    ? maskCredential(draftAccessKeySecret)
    : canReuseSavedSecret
      ? savedMaskedAccessKeySecret
      : "";
  setCredentialValue(elements["oss-access-key-secret"], masked, Boolean(masked));
  if (pendingSecret.trim()) {
    void saveCredentialsLocally(formConfig(), { validateAfterSave: true });
  }
});
for (const id of [
  "control-plane-url",
  "embedding-url",
  "embedding-model",
  "embedding-dimension",
  "oss-region",
  "oss-account-id",
  "oss-bucket",
  "oss-visual-index",
  "oss-transcript-index",
  "oss-security-token",
]) {
  elements[id].addEventListener("input", invalidateCredentialValidation);
}

elements["embedding-url"].addEventListener("input", () => {
  window.clearTimeout(embeddingDiscoveryTimer);
  discoveredEmbeddingUrl = "";
  resolvedEmbeddingInputStyle = "auto";
  resolvedEmbeddingSpace = "";
  elements["embedding-model"].value = "";
  elements["embedding-dimension"].value = "";
  const value = elements["embedding-url"].value.trim();
  if (!value) return;
  embeddingDiscoveryTimer = window.setTimeout(() => void discoverEmbedding(), 650);
});
elements["embedding-url"].addEventListener("blur", () => {
  window.clearTimeout(embeddingDiscoveryTimer);
  void discoverEmbedding();
});

async function resetVideoMemory() {
  const button = elements["reset-memory"];
  button.disabled = true;
  elements["settings-status"].textContent = "正在读取索引…";
  try {
    const info = await message({ type: "RESET_VIDEO_MEMORY_INFO" });
    if (info.error || !info.ok) throw new Error(info.error || "无法读取索引");
    const confirmed = window.confirm(
      `永久清空 ${info.bucket} 中的两个索引？\n\n${info.visual_index}：${info.visual_count} 条\n${info.transcript_index}：${info.transcript_count} 条\n\n云端向量和本地队列都会删除，无法恢复。`
    );
    if (!confirmed) {
      elements["settings-status"].textContent = "已取消";
      return;
    }
    elements["settings-status"].textContent = "正在清空视频索引…";
    const result = await message({ type: "RESET_ALL_VIDEO_MEMORY", payload: { target: info.target } });
    if (result.error || !result.ok) throw new Error(result.error || "清空失败");
    elements["settings-status"].textContent = `已清空 ${result.visual_deleted + result.transcript_deleted} 条云端向量`;
    await refreshStats();
  } catch (error) {
    elements["settings-status"].textContent = String(error.message || error);
  } finally {
    button.disabled = false;
  }
}

async function syncFromControlPlane() {
  const baseUrl = elements["control-plane-url"].value.trim().replace(/\/+$/, "");
  if (!baseUrl) return;
  try {
    const origin = `${new URL(baseUrl).origin}/*`;
    const allowed = await chrome.permissions.request({ origins: [origin] });
    if (!allowed) throw new Error("未获得访问本地 Dashboard 的权限");
    const response = await fetch(`${baseUrl}/api/extension-config`);
    const remote = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(remote.error || `Dashboard 返回 ${response.status}`);
    const synced = {
      ...formConfig(),
      backendMode: "local",
      controlPlaneUrl: baseUrl,
      embeddingBaseUrl: remote.embeddingBaseUrl,
      embeddingModel: remote.embeddingModel,
      embeddingDimension: Number(remote.embeddingDimension),
      embeddingInputStyle: remote.embeddingInputStyle,
      embeddingSpace: remote.embeddingSpace,
    };
    const saved = await message({ type: "SAVE_CONFIG", config: synced });
    if (saved.error) throw new Error(saved.error);
    elements["embedding-url"].value = synced.embeddingBaseUrl;
    elements["embedding-model"].value = synced.embeddingModel;
    elements["embedding-dimension"].value = String(synced.embeddingDimension);
    resolvedEmbeddingInputStyle = synced.embeddingInputStyle || "auto";
    resolvedEmbeddingSpace = synced.embeddingSpace || "";
    discoveredEmbeddingUrl = synced.embeddingBaseUrl;
    elements["settings-status"].textContent = `已读取 ${remote.label || remote.profileId}`;
    invalidateCredentialValidation();
  } catch (error) {
    elements["settings-status"].textContent = String(error.message || error);
  }
}

async function openLocalFrontend(view) {
  const url = elements["control-plane-url"].value.trim().replace(/\/+$/, "");
  if (!url) {
    elements["settings-panel"].hidden = false;
    setBackendMode("local");
    elements["settings-status"].textContent = "先填写本地服务地址";
    return;
  }
  await chrome.tabs.create({ url: `${url}/?view=${encodeURIComponent(view)}` });
}

elements.library.addEventListener("click", () => {
  if (backendMode === "local") void openLocalFrontend("videos");
  else void chrome.tabs.create({ url: chrome.runtime.getURL("search.html") });
});
elements["local-library"].addEventListener("click", () => void openLocalFrontend("files"));
elements["backend-cloud"].addEventListener("click", () => setBackendMode("cloud"));
elements["backend-local"].addEventListener("click", () => setBackendMode("local"));
elements.settings.addEventListener("click", () => {
  elements["settings-panel"].hidden = !elements["settings-panel"].hidden;
});
elements["history-toggle"].addEventListener("click", () => {
  historyExpanded = !historyExpanded;
  renderHistory();
});
elements["save-settings"].addEventListener("click", () => void saveSettings());
elements["sync-control-plane"].addEventListener("click", () => void syncFromControlPlane());
elements["reset-memory"].addEventListener("click", () => void resetVideoMemory());
void initialize().then(() => window.setInterval(() => void refreshProgress(), 3000));
