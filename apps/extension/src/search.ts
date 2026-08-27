// @ts-nocheck -- behavior-preserving search UI port.

const elements = Object.fromEntries(
  [
    "video-count",
    "visual-count",
    "text-count",
    "connection",
    "search-form",
    "query",
    "image-input",
    "image-query-preview",
    "image-preview-image",
    "image-preview-name",
    "remove-image",
    "search-submit",
    "search-state",
    "visual-lane",
    "visual-result-count",
    "visual-results",
    "visual-pagination",
    "text-lane",
    "text-result-count",
    "text-results",
    "text-pagination",
    "show-library",
    "library-section",
    "library-video-count",
    "library-state",
    "library-videos",
    "library-pagination",
    "delete-dialog",
    "delete-dialog-name",
    "delete-dialog-summary",
    "delete-dialog-status",
    "delete-cancel",
    "delete-confirm",
  ].map((id) => [id, document.getElementById(id)])
);
let selectedImage = null;
let previewUrl = "";
const PAGE_SIZE = 12;
const LIBRARY_PAGE_SIZE = 8;
const resultState = {
  visual: [],
  transcript: [],
  visualPage: 1,
  transcriptPage: 1,
};
const libraryState = {
  videos: [],
  page: 1,
};
let deleteTarget = null;

function setStatus(text = "", tone = "") {
  elements["search-state"].textContent = text;
  elements["search-state"].className = `search-state${tone ? ` ${tone}` : ""}`;
  elements["search-state"].hidden = !text;
}

function hasExtensionRuntime() {
  return typeof chrome !== "undefined" && Boolean(chrome.runtime?.id);
}

async function message(messageValue) {
  if (hasExtensionRuntime()) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(messageValue, (response) => {
        if (chrome.runtime.lastError) return resolve({ error: chrome.runtime.lastError.message });
        resolve(response || {});
      });
    });
  }
  return { error: "请从 Indexed Chrome 扩展打开此页面" };
}

function openExternal(url) {
  if (!url) return;
  if (hasExtensionRuntime()) {
    chrome.tabs.create({ url });
  } else {
    window.open(url, "_blank", "noopener");
  }
}

function fileToBase64(file) {
  return file.arrayBuffer().then((buffer) => {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 32768) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
    }
    return btoa(binary);
  });
}

function clearSelectedImage() {
  selectedImage = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = "";
  elements["image-input"].value = "";
  elements["image-preview-image"].removeAttribute("src");
  elements["image-query-preview"].hidden = true;
  elements.query.placeholder = "描述画面或台词，也可以粘贴图片";
  elements["search-submit"].textContent = "搜索";
}

function selectImage(file, sourceLabel = "已选择") {
  if (!file || !String(file.type || "").startsWith("image/")) {
    setStatus("请选择 JPEG、PNG、WebP 或 GIF 图片", "error");
    return false;
  }
  if (file.size > 12 * 1024 * 1024) {
    setStatus("图片不能超过 12MB", "error");
    return false;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedImage = file;
  previewUrl = URL.createObjectURL(file);
  elements["image-preview-image"].src = previewUrl;
  elements["image-preview-name"].textContent = `${sourceLabel} · ${file.name || "clipboard.png"}`;
  elements["image-query-preview"].hidden = false;
  elements.query.placeholder = "补充文字条件（可选）";
  elements["search-submit"].textContent = "以图搜索";
  setStatus();
  return true;
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

function sourceName(site) {
  return String(site || "").toLowerCase() === "bilibili" ? "哔哩哔哩" : "YouTube";
}

function formatActivity(timestamp) {
  const value = Number(timestamp || 0);
  if (!value) return "云端记录";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function setLibraryStatus(text = "", tone = "") {
  elements["library-state"].textContent = text;
  elements["library-state"].className = `library-state${tone ? ` ${tone}` : ""}`;
  elements["library-state"].hidden = !text;
}

function openDeleteDialog(video) {
  deleteTarget = video;
  const total = Number(video.visual_count || 0) + Number(video.transcript_count || 0);
  elements["delete-dialog-name"].textContent = video.title || "未命名视频";
  elements["delete-dialog-summary"].textContent = `${video.visual_count || 0} 条画面索引 · ${video.transcript_count || 0} 条字幕索引 · 共 ${total} 条云端记录`;
  elements["delete-dialog-status"].textContent = "";
  elements["delete-dialog-status"].className = "delete-dialog-status";
  elements["delete-confirm"].disabled = false;
  elements["delete-confirm"].textContent = total ? `永久删除 ${total} 条索引` : "清理本地记录";
  elements["delete-dialog"].showModal();
}

function createLibraryCard(video) {
  const card = document.createElement("article");
  card.className = "memory-video-card";
  const media = document.createElement("div");
  media.className = "memory-video-media";
  if (video.thumbnail_url) {
    const image = document.createElement("img");
    image.src = video.thumbnail_url;
    image.alt = "";
    image.loading = "lazy";
    media.appendChild(image);
  }
  const badge = document.createElement("span");
  badge.className = `memory-site ${video.source_site === "bilibili" ? "bilibili" : "youtube"}`;
  badge.textContent = video.source_site === "bilibili" ? "B站" : "YT";
  media.appendChild(badge);

  const body = document.createElement("div");
  body.className = "memory-video-body";
  const heading = document.createElement("div");
  heading.className = "memory-video-heading";
  const title = document.createElement("h3");
  title.textContent = video.title || "未命名视频";
  title.title = video.title || "未命名视频";
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "memory-delete";
  remove.textContent = "删除";
  remove.setAttribute("aria-label", `删除 ${video.title || "这个视频"} 的全部索引`);
  remove.addEventListener("click", () => openDeleteDialog(video));
  heading.append(title, remove);

  const meta = document.createElement("p");
  meta.className = "memory-video-meta";
  meta.textContent = [
    video.channel_name || sourceName(video.source_site),
    video.duration ? formatTime(video.duration) : "",
    formatActivity(video.last_processed_at),
  ].filter(Boolean).join(" · ");

  const counts = document.createElement("div");
  counts.className = "memory-counts";
  counts.innerHTML = `<span><strong>${Number(video.visual_count || 0)}</strong>画面</span><span><strong>${Number(video.transcript_count || 0)}</strong>字幕</span><span><strong>${Number(video.pending_count || 0)}</strong>待处理</span>`;

  const actions = document.createElement("div");
  actions.className = "memory-card-actions";
  const state = document.createElement("span");
  state.className = `memory-state${video.processing ? " processing" : ""}`;
  state.textContent = video.processing ? "正在处理" : video.failed_count ? `${video.failed_count} 个分片失败` : "云端已保存";
  const open = document.createElement("button");
  open.type = "button";
  open.textContent = "打开视频 ↗";
  open.disabled = !video.source_url;
  open.addEventListener("click", () => openExternal(video.source_url));
  actions.append(state, open);

  body.append(heading, meta, counts, actions);
  card.append(media, body);
  return card;
}

function renderLibrary() {
  const videos = libraryState.videos || [];
  const totalPages = Math.max(1, Math.ceil(videos.length / LIBRARY_PAGE_SIZE));
  libraryState.page = Math.min(totalPages, Math.max(1, libraryState.page));
  const offset = (libraryState.page - 1) * LIBRARY_PAGE_SIZE;
  const pageItems = videos.slice(offset, offset + LIBRARY_PAGE_SIZE);
  const fragment = document.createDocumentFragment();
  pageItems.forEach((video) => fragment.appendChild(createLibraryCard(video)));
  elements["library-videos"].replaceChildren(fragment);
  elements["library-video-count"].textContent = `${videos.length} 个`;
  if (!videos.length) setLibraryStatus("还没有保存过视频。在播放器上点击 Indexed 小图标后，它会出现在这里。", "empty");

  const pagination = elements["library-pagination"];
  pagination.hidden = videos.length <= LIBRARY_PAGE_SIZE;
  if (pagination.hidden) {
    pagination.replaceChildren();
    return;
  }
  const previous = document.createElement("button");
  previous.type = "button";
  previous.textContent = "上一页";
  previous.disabled = libraryState.page <= 1;
  previous.addEventListener("click", () => {
    libraryState.page -= 1;
    renderLibrary();
    elements["library-section"].scrollIntoView({ behavior: "smooth", block: "start" });
  });
  const label = document.createElement("span");
  label.textContent = `${libraryState.page} / ${totalPages}`;
  const next = document.createElement("button");
  next.type = "button";
  next.textContent = "下一页";
  next.disabled = libraryState.page >= totalPages;
  next.addEventListener("click", () => {
    libraryState.page += 1;
    renderLibrary();
    elements["library-section"].scrollIntoView({ behavior: "smooth", block: "start" });
  });
  pagination.replaceChildren(previous, label, next);
}

async function refreshLibrary() {
  setLibraryStatus("正在读取云端记录…");
  const response = await message({ type: "VIDEO_LIBRARY" });
  if (response.error) {
    setLibraryStatus(response.error, "error");
    elements.connection.textContent = "云端未连接";
    elements.connection.title = response.error;
    elements.connection.className = "connection error";
    return response;
  }
  libraryState.videos = response.videos || [];
  elements.connection.textContent = "已连接";
  elements.connection.title = "OSS Vectors 直连";
  elements.connection.className = "connection ok";
  elements["video-count"].textContent = String(response.video_count || 0);
  elements["visual-count"].textContent = String(response.visual_count || 0);
  elements["text-count"].textContent = String(response.transcript_count || 0);
  setLibraryStatus();
  renderLibrary();
  return response;
}

function showLibrary() {
  document.body.classList.remove("has-results");
  elements["visual-lane"].hidden = true;
  elements["text-lane"].hidden = true;
  elements["library-section"].scrollIntoView({ behavior: "smooth", block: "start" });
}

async function deleteVideo() {
  if (!deleteTarget) return;
  elements["delete-confirm"].disabled = true;
  elements["delete-cancel"].disabled = true;
  elements["delete-confirm"].textContent = "正在删除…";
  elements["delete-dialog-status"].textContent = "正在停止该视频的后台任务，并从两个云端 Index 分批删除…";
  try {
    const response = await message({
      type: "DELETE_VIDEO",
      payload: {
        video_id: deleteTarget.video_id,
        source_site: deleteTarget.source_site,
      },
    });
    if (response.error || !response.ok) {
      const detail = String(response.error || "删除失败");
      if (/403|AccessDenied/i.test(detail)) {
        throw new Error("当前 RAM 用户没有删除权限。请绑定 IndexedOSSVectorsDelete 策略后再重试。");
      }
      throw new Error(detail);
    }
    const removedVideoId = deleteTarget.video_id;
    const removedSourceSite = deleteTarget.source_site;
    resultState.visual = resultState.visual.filter((item) =>
      item.video_id !== removedVideoId || item.source_site !== removedSourceSite
    );
    resultState.transcript = resultState.transcript.filter((item) =>
      item.video_id !== removedVideoId || item.source_site !== removedSourceSite
    );
    elements["delete-dialog"].close();
    deleteTarget = null;
    libraryState.page = 1;
    await refreshLibrary();
    setLibraryStatus(`已删除 ${response.visual_deleted || 0} 条画面索引和 ${response.transcript_deleted || 0} 条字幕索引。`, "success");
  } catch (error) {
    elements["delete-dialog-status"].textContent = error instanceof Error ? error.message : String(error);
    elements["delete-dialog-status"].className = "delete-dialog-status error";
    elements["delete-confirm"].disabled = false;
    elements["delete-confirm"].textContent = "重试删除";
  } finally {
    elements["delete-cancel"].disabled = false;
  }
}

function thumbnailUrl(value) {
  if (!value) return "";
  return value.startsWith("/") ? "" : value;
}

function apiUrl(value) {
  if (!value) return "";
  return value.startsWith("/") ? "" : value;
}

function addRank(wrapper, rankValue) {
  const rank = document.createElement("span");
  rank.className = "rank";
  rank.textContent = `#${rankValue}`;
  wrapper.appendChild(rank);
}

function createImage(item, rankValue) {
  const wrapper = document.createElement("div");
  wrapper.className = "visual-thumb";
  const image = document.createElement("img");
  image.alt = "";
  image.loading = "lazy";
  image.src = thumbnailUrl(item.thumbnail_url);
  const time = document.createElement("span");
  time.className = "time";
  time.textContent = formatTime(item.playback_time);
  wrapper.append(image, time);
  addRank(wrapper, rankValue);
  return wrapper;
}

function createVideo(item, rankValue) {
  const wrapper = document.createElement("div");
  wrapper.className = "visual-thumb";
  const video = document.createElement("video");
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.poster = thumbnailUrl(item.thumbnail_url);
  video.src = apiUrl(item.clip_url);
  wrapper.addEventListener("mouseenter", () => void video.play().catch(() => {}));
  wrapper.addEventListener("mouseleave", () => {
    video.pause();
    video.currentTime = 0;
  });
  const time = document.createElement("span");
  time.className = "time";
  time.textContent = `${formatTime(item.playback_time)} · 10s`;
  wrapper.append(video, time);
  addRank(wrapper, rankValue);
  return wrapper;
}

function meta(item) {
  const row = document.createElement("div");
  row.className = "card-meta";
  const channel = document.createElement("span");
  channel.textContent = item.channel_name || "未知频道";
  const score = document.createElement("span");
  const scoreValue = Number(item.score || 0);
  score.className = `score${scoreValue >= 0.4 ? " high" : scoreValue < 0.3 ? " low" : ""}`;
  score.title = "原始向量分数";
  score.textContent = scoreValue.toFixed(3);
  row.append(channel, score);
  return row;
}

function visualCard(item, index) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "visual-card";
  const body = document.createElement("div");
  body.className = "card-body";
  const title = document.createElement("div");
  title.className = "card-title";
  title.textContent = item.title || "未命名视频";
  body.append(title, meta(item));
  card.append(item.clip_url ? createVideo(item, index + 1) : createImage(item, index + 1), body);
  card.addEventListener("click", () => openExternal(item.open_url || item.youtube_url));
  return card;
}

function textCard(item, index) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "text-card";
  const body = document.createElement("div");
  body.className = "card-body";
  const title = document.createElement("div");
  title.className = "card-title";
  title.textContent = item.title || "未命名视频";
  const snippet = document.createElement("div");
  snippet.className = "snippet";
  snippet.textContent = item.transcript_text || "字幕片段";
  body.append(title, snippet, meta(item));
  card.append(createImage(item, index + 1), body);
  card.addEventListener("click", () => openExternal(item.open_url || item.youtube_url));
  return card;
}

function renderResults(container, items, factory, rankOffset = 0) {
  container.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "这一条索引暂时没有命中";
    container.appendChild(empty);
    return;
  }
  items.forEach((item, index) => container.appendChild(factory(item, index + rankOffset)));
}

function renderPagination(kind) {
  const items = kind === "visual" ? resultState.visual : resultState.transcript;
  const pageKey = kind === "visual" ? "visualPage" : "transcriptPage";
  const laneKey = kind === "visual" ? "visual" : "text";
  const container = elements[`${laneKey}-pagination`];
  const results = elements[`${laneKey}-results`];
  const factory = kind === "visual" ? visualCard : textCard;
  const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
  const page = Math.min(totalPages, Math.max(1, resultState[pageKey]));
  resultState[pageKey] = page;
  const offset = (page - 1) * PAGE_SIZE;
  renderResults(results, items.slice(offset, offset + PAGE_SIZE), factory, offset);
  container.hidden = items.length <= PAGE_SIZE;
  if (container.hidden) {
    container.replaceChildren();
    return;
  }
  const previous = document.createElement("button");
  previous.type = "button";
  previous.textContent = "上一页";
  previous.disabled = page <= 1;
  previous.addEventListener("click", () => {
    resultState[pageKey] -= 1;
    renderPagination(kind);
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  const label = document.createElement("span");
  label.textContent = `${page} / ${totalPages}`;
  const next = document.createElement("button");
  next.type = "button";
  next.textContent = "下一页";
  next.disabled = page >= totalPages;
  next.addEventListener("click", () => {
    resultState[pageKey] += 1;
    renderPagination(kind);
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  container.replaceChildren(previous, label, next);
}

async function runSearch(query) {
  elements["search-submit"].disabled = true;
  setStatus("搜索中…");
  try {
    const response = await message({ type: "SEARCH", payload: { query, limit: 60 } });
    if (response.error) {
      setStatus(response.error, "error");
      return;
    }
    const visual = response.visual || [];
    const transcript = response.transcript || [];
    resultState.visual = visual;
    resultState.transcript = transcript;
    resultState.visualPage = 1;
    resultState.transcriptPage = 1;
    document.body.classList.add("has-results");
    elements["visual-lane"].hidden = false;
    elements["text-lane"].hidden = false;
    elements["visual-result-count"].textContent = `${visual.length} 条`;
    elements["text-result-count"].textContent = `${transcript.length} 条`;
    renderPagination("visual");
    renderPagination("transcript");
    setStatus();
    const url = new URL(location.href);
    url.searchParams.set("q", query);
    history.replaceState(null, "", url);
  } finally {
    elements["search-submit"].disabled = false;
  }
}

async function runImageSearch(file, query = "") {
  elements["search-submit"].disabled = true;
  setStatus("搜索相似画面…");
  try {
    const imageBase64 = await fileToBase64(file);
    const response = await message({
      type: "IMAGE_SEARCH",
      payload: {
        image_base64: imageBase64,
        image_filename: file.name || "clipboard.png",
        query,
        limit: 60,
      },
    });
    if (response.error) {
      setStatus(response.error, "error");
      return;
    }
    const visual = response.visual || [];
    resultState.visual = visual;
    resultState.transcript = [];
    resultState.visualPage = 1;
    document.body.classList.add("has-results");
    elements["visual-lane"].hidden = false;
    elements["text-lane"].hidden = true;
    elements["visual-result-count"].textContent = `${visual.length} 条`;
    renderPagination("visual");
    setStatus();
    const url = new URL(location.href);
    url.searchParams.delete("q");
    history.replaceState(null, "", url);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), "error");
  } finally {
    elements["search-submit"].disabled = false;
  }
}

async function initialize() {
  await refreshLibrary();
  const query = new URL(location.href).searchParams.get("q") || "";
  if (query) {
    elements.query.value = query;
    await runSearch(query);
  }
}

elements["search-form"].addEventListener("submit", (event) => {
  event.preventDefault();
  const query = elements.query.value.trim();
  if (selectedImage) {
    void runImageSearch(selectedImage, query);
  } else if (query) {
    void runSearch(query);
  } else {
    setStatus("输入文字或添加图片", "error");
  }
});

elements["image-input"].addEventListener("change", () => {
  const [file] = elements["image-input"].files || [];
  if (file) selectImage(file, "已选择");
});

elements["remove-image"].addEventListener("click", () => {
  clearSelectedImage();
  setStatus();
});

elements["show-library"].addEventListener("click", showLibrary);
elements["delete-confirm"].addEventListener("click", () => void deleteVideo());
elements["delete-dialog"].addEventListener("close", () => {
  deleteTarget = null;
  elements["delete-cancel"].disabled = false;
});

document.addEventListener("paste", (event) => {
  const imageItem = Array.from(event.clipboardData?.items || []).find((item) =>
    String(item.type || "").startsWith("image/")
  );
  const file = imageItem?.getAsFile();
  if (!file) return;
  event.preventDefault();
  selectImage(file, "来自剪贴板");
});

elements["search-form"].addEventListener("dragover", (event) => {
  event.preventDefault();
  elements["search-form"].classList.add("drag-active");
});

elements["search-form"].addEventListener("dragleave", (event) => {
  if (!elements["search-form"].contains(event.relatedTarget)) {
    elements["search-form"].classList.remove("drag-active");
  }
});

elements["search-form"].addEventListener("drop", (event) => {
  event.preventDefault();
  elements["search-form"].classList.remove("drag-active");
  const file = Array.from(event.dataTransfer?.files || []).find((item) =>
    String(item.type || "").startsWith("image/")
  );
  if (file) selectImage(file, "已拖入");
  else setStatus("没有找到可用图片", "error");
});

window.addEventListener("beforeunload", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
void initialize();
