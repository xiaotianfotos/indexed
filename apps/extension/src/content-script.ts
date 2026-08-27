// @ts-nocheck -- behavior-preserving site adapter port; YouTube/Bilibili typing follows separately.
import { PRODUCT_DEFAULTS, SEGMENT_SCHEMAS } from "@indexed/contracts";

(() => {
  const CONTENT_SCRIPT_VERSION = "0.12.13";
  if (globalThis.__sceneMemoryContentScriptLoaded === CONTENT_SCRIPT_VERSION) return;
  globalThis.__sceneMemoryContentScriptLoaded = CONTENT_SCRIPT_VERSION;

  const SEGMENT_SECONDS = PRODUCT_DEFAULTS.capture.segmentSeconds;
  const VIDEO_SEGMENT_SCHEMA = SEGMENT_SCHEMAS.video;
  const TRANSCRIPT_SEGMENT_SCHEMA = SEGMENT_SCHEMAS.transcript;
  const CAPTURE_FPS = PRODUCT_DEFAULTS.capture.framesPerSecond;
  const MAX_CAPTURE_WIDTH = PRODUCT_DEFAULTS.capture.maxWidth;
  const MAX_CAPTURE_HEIGHT = PRODUCT_DEFAULTS.capture.maxHeight;
  const VIDEO_BITS_PER_SECOND = PRODUCT_DEFAULTS.capture.videoBitsPerSecond;
  const TICK_MILLISECONDS = 125;
  const VIDEO_OPT_INS_KEY = "videoIndexingOptIns";
  const FLOATING_BUTTON_ID = "indexed-video-toggle";

  const runtime = {
    running: false,
    retainPreview: true,
    captureInterval: SEGMENT_SECONDS,
    activeSegment: null,
    indexedVisual: 0,
    indexedText: 0,
    skippedVisual: 0,
    watchedVideoId: "",
    watchedSegments: new Set(),
    lastError: "",
    timer: null,
    tickInFlight: false,
    rebuildJob: null,
    lastCompletedVideoId: "",
    resumeOnVisible: false,
    optedIn: false,
    activeIdentity: "",
    floatingHost: null,
    floatingButton: null,
    lastFloatingCheck: 0,
    lastSessionSyncAt: 0,
  };

  function extensionContextAvailable() {
    return typeof chrome !== "undefined"
      && Boolean(chrome.runtime?.id)
      && Boolean(chrome.storage?.local);
  }

  function invalidateExtensionContext(message = "扩展已重新加载，请刷新当前视频页面") {
    runtime.running = false;
    runtime.lastError = message;
    if (runtime.timer) {
      window.clearInterval(runtime.timer);
      runtime.timer = null;
    }
    const segment = runtime.activeSegment;
    runtime.activeSegment = null;
    if (segment) {
      try {
        if (segment.recorder?.state !== "inactive") segment.recorder.stop();
      } catch {}
      try {
        segment.stream?.getTracks().forEach((track) => track.stop());
      } catch {}
    }
  }

  async function extensionStorageGet(defaults) {
    if (!extensionContextAvailable()) {
      invalidateExtensionContext();
      return null;
    }
    try {
      return await chrome.storage.local.get(defaults);
    } catch (error) {
      invalidateExtensionContext(String(error?.message || error));
      return null;
    }
  }

  function sourceSite() {
    return location.hostname.endsWith("bilibili.com") ? "bilibili" : "youtube";
  }

  function videoId() {
    if (sourceSite() === "bilibili") {
      return location.pathname.match(/\/video\/(BV[A-Za-z0-9]+|av\d+)/i)?.[1] || "";
    }
    return new URL(location.href).searchParams.get("v") || "";
  }

  function currentVideoIdentity() {
    const id = videoId();
    return id ? `${sourceSite()}:${id}` : "";
  }

  function videoElement() {
    return document.querySelector("video.html5-main-video")
      || document.querySelector(".bpx-player-video-wrap video")
      || document.querySelector("video");
  }

  function playerControlSlot() {
    if (sourceSite() === "bilibili") {
      const container = document.querySelector(".bpx-player-control-bottom-right");
      return container
        ? {
            container,
            anchor: container.querySelector(".immersive-translate-quick-button-container")
              || container.firstElementChild,
            width: 40,
            pageClass: "bpx-player-ctrl-btn indexed-player-control",
          }
        : null;
    }
    const container = document.querySelector(".ytp-right-controls-left")
      || document.querySelector(".ytp-right-controls");
    return container
      ? {
          container,
          anchor: container.querySelector(".ytp-subtitles-button")
            || container.querySelector(".ytp-settings-button")
            || container.firstElementChild,
          width: 48,
          pageClass: "ytp-button indexed-player-control",
        }
      : null;
  }

  function updateFloatingButton() {
    const button = runtime.floatingButton;
    if (!button) return;
    const state = runtime.lastError ? "error" : runtime.running && runtime.optedIn ? "active" : "idle";
    button.dataset.state = state;
    const label = state === "active"
      ? "Indexed 正在记忆当前视频，点击暂停"
      : state === "error"
        ? `Indexed 发生错误，点击重试：${runtime.lastError}`
        : "用 Indexed 记忆当前视频";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.setAttribute("aria-pressed", String(state === "active"));
  }

  function ensureFloatingButton(force = false) {
    const now = Date.now();
    if (!force && now - runtime.lastFloatingCheck < 1000) return;
    runtime.lastFloatingCheck = now;
    const slot = playerControlSlot();
    if (!slot || !videoId()) return;
    if (runtime.floatingHost?.isConnected && runtime.floatingHost.parentElement === slot.container) {
      if (slot.anchor && runtime.floatingHost.nextElementSibling !== slot.anchor) {
        slot.container.insertBefore(runtime.floatingHost, slot.anchor);
      }
      updateFloatingButton();
      return;
    }
    const staleHost = document.getElementById(FLOATING_BUTTON_ID);
    if (staleHost && staleHost !== runtime.floatingHost) {
      // Clicking an invalidated older instance makes it run its own cleanup,
      // including stopping the timer that would otherwise recreate the host.
      if (staleHost.dataset.indexedVersion !== CONTENT_SCRIPT_VERSION) {
        try { staleHost.shadowRoot?.querySelector("button")?.click(); } catch {}
      }
      staleHost.remove();
    }
    runtime.floatingHost?.remove();
    const host = document.createElement("div");
    host.id = FLOATING_BUTTON_ID;
    host.className = slot.pageClass;
    host.dataset.site = sourceSite();
    host.dataset.indexedVersion = CONTENT_SCRIPT_VERSION;
    host.style.cssText = `display:flex;align-items:center;justify-content:center;position:relative;width:${slot.width}px;height:100%;flex:0 0 ${slot.width}px;padding:0;margin:0;pointer-events:auto;`;
    const shadow = host.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = `
      :host { color: rgba(255,255,255,.82); }
      button {
        position: relative; box-sizing: border-box; display: grid; place-items: center; width: 100%; height: 100%;
        margin: 0; padding: 0; overflow: visible; border: 0; border-radius: 0; color: inherit;
        background: transparent; cursor: pointer; opacity: .74; transform: translateZ(0);
        transition: opacity .16s, transform .16s, color .16s, filter .16s;
      }
      button:hover, button:focus-visible { opacity: 1; transform: scale(1.08); outline: none; }
      :host([data-site="bilibili"]) button { align-items: start; justify-items: center; }
      svg { display: block; width: 21px; height: 21px; pointer-events: none; filter: drop-shadow(0 1px 2px rgba(0,0,0,.38)); }
      .frame { fill: none; stroke: currentColor; stroke-width: 1.5; opacity: .74; }
      .tile { fill: currentColor; opacity: .84; }
      .tile.muted { opacity: .3; }
      button[data-state="active"] { color: #6be3b2; opacity: 1; }
      button[data-state="active"] svg { filter: drop-shadow(0 0 5px rgba(77,225,166,.58)); }
      button[data-state="active"]::after { content: ""; position: absolute; right: 5px; top: 7px; width: 4px; height: 4px; border-radius: 50%; background: #6be3b2; box-shadow: 0 0 6px #6be3b2; }
      button[data-state="error"] { color: #ff858b; opacity: 1; }
    `;
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect class="frame" x="2.75" y="2.75" width="18.5" height="18.5" rx="4.25"/>
        <rect class="tile" x="5.5" y="5.5" width="3" height="3" rx=".8"/>
        <rect class="tile muted" x="10.5" y="5.5" width="3" height="3" rx=".8"/>
        <rect class="tile muted" x="15.5" y="5.5" width="3" height="3" rx=".8"/>
        <rect class="tile muted" x="5.5" y="10.5" width="3" height="3" rx=".8"/>
        <rect class="tile" x="10.5" y="10.5" width="3" height="3" rx=".8"/>
        <rect class="tile muted" x="15.5" y="10.5" width="3" height="3" rx=".8"/>
        <rect class="tile muted" x="5.5" y="15.5" width="3" height="3" rx=".8"/>
        <rect class="tile muted" x="10.5" y="15.5" width="3" height="3" rx=".8"/>
        <rect class="tile" x="15.5" y="15.5" width="3" height="3" rx=".8"/>
      </svg>`;
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      void setCurrentVideoEnabled(!runtime.optedIn, { retainPreview: runtime.retainPreview });
    });
    shadow.append(style, button);
    slot.container.insertBefore(host, slot.anchor || null);
    runtime.floatingHost = host;
    runtime.floatingButton = button;
    updateFloatingButton();
  }

  function textOf(selector) {
    return document.querySelector(selector)?.textContent?.trim() || "";
  }

  function pageMetadata(video = videoElement()) {
    const site = sourceSite();
    const channelLink = site === "bilibili"
      ? document.querySelector("a.up-name, .up-info-container a[href*='space.bilibili.com']")
      : document.querySelector("ytd-watch-metadata ytd-channel-name a, #owner ytd-channel-name a");
    const channelHref = channelLink?.getAttribute("href") || "";
    const channelUrl = channelHref
      ? new URL(channelHref, location.origin).href.replace(/\/$/, "")
      : "";
    const channelId = site === "bilibili"
      ? channelUrl.match(/space\.bilibili\.com\/(\d+)/)?.[1] || ""
      : (channelHref.startsWith("/channel/") ? channelHref.split("/")[2] || "" : "");
    const title = site === "bilibili"
      ? textOf("h1.video-title") || document.title.replace(/_哔哩哔哩_bilibili\s*$/, "")
      : textOf("ytd-watch-metadata h1 yt-formatted-string")
        || textOf("h1.title yt-formatted-string")
        || document.title.replace(/\s*-\s*YouTube\s*$/, "");
    return {
      videoId: videoId(),
      sourceSite: site,
      title,
      channelId,
      channelName: channelLink?.textContent?.trim() || "",
      channelUrl,
      sourceUrl: location.href,
      thumbnailUrl: document.querySelector('meta[property="og:image"]')?.content || "",
      currentTime: Number(video?.currentTime || 0),
      duration: Number.isFinite(video?.duration) ? Number(video.duration) : 0,
    };
  }

  function publicState() {
    const video = videoElement();
    const metadata = pageMetadata(video);
    return {
      ...metadata,
      running: runtime.running,
      optedIn: runtime.optedIn,
      videoIdentity: currentVideoIdentity(),
      retainPreview: runtime.retainPreview,
      captureInterval: runtime.captureInterval,
      playing: Boolean(video && !video.paused && !video.ended),
      indexedVisual: runtime.indexedVisual,
      indexedText: runtime.indexedText,
      skippedVisual: runtime.skippedVisual,
      watchedSegments: runtime.watchedVideoId === metadata.videoId
        ? Array.from(runtime.watchedSegments).sort((a, b) => a - b)
        : [],
      lastError: runtime.lastError,
      captionsDetected: Boolean(runtime.activeSegment?.captionTexts.size),
      recordingSegment: runtime.activeSegment?.segmentIndex ?? null,
      recordingFrameCount: runtime.activeSegment?.frameCount || 0,
      visualInput: "native_video",
      rebuildJob: runtime.rebuildJob
        ? {
            active: Boolean(runtime.rebuildJob.active),
            index: Number(runtime.rebuildJob.index || 0),
            total: Number(runtime.rebuildJob.entries?.length || 0),
            status: runtime.rebuildJob.status || "",
            currentVideoId: runtime.rebuildJob.entries?.[runtime.rebuildJob.index]?.id || "",
          }
        : null,
    };
  }

  function send(message) {
    return new Promise((resolve) => {
      if (!extensionContextAvailable()) {
        invalidateExtensionContext();
        resolve(null);
        return;
      }
      try {
        chrome.runtime.sendMessage(message, (response) => {
          if (chrome.runtime.lastError) {
            runtime.lastError = chrome.runtime.lastError.message;
            resolve(null);
            return;
          }
          resolve(response || null);
        });
      } catch (error) {
        invalidateExtensionContext(String(error?.message || error));
        resolve(null);
      }
    });
  }

  function syncVideoSession(force = false) {
    const identity = currentVideoIdentity();
    if (!identity) return Promise.resolve(null);
    const now = Date.now();
    if (!force && now - runtime.lastSessionSyncAt < 1000) return Promise.resolve(null);
    runtime.lastSessionSyncAt = now;
    const state = publicState();
    return send({
      type: "VIDEO_SESSION",
      payload: {
        videoId: state.videoId,
        sourceSite: state.sourceSite,
        title: state.title,
        channelId: state.channelId,
        channelName: state.channelName,
        sourceUrl: state.sourceUrl,
        thumbnailUrl: state.thumbnailUrl,
        duration: state.duration,
        segmentInterval: SEGMENT_SECONDS,
        optedIn: state.optedIn,
        running: state.running,
        playing: state.playing,
        recordingSegment: state.recordingSegment,
        watchedSegments: state.watchedSegments,
      },
    });
  }

  function recorderMimeType() {
    const candidates = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ];
    return candidates.find((value) => MediaRecorder.isTypeSupported(value)) || "";
  }

  function captureSize(video) {
    const sourceWidth = Math.max(1, Number(video.videoWidth || 0));
    const sourceHeight = Math.max(1, Number(video.videoHeight || 0));
    const scale = Math.min(1, MAX_CAPTURE_WIDTH / sourceWidth, MAX_CAPTURE_HEIGHT / sourceHeight);
    return {
      width: Math.max(2, Math.round((sourceWidth * scale) / 2) * 2),
      height: Math.max(2, Math.round((sourceHeight * scale) / 2) * 2),
    };
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(reader.error || new Error("无法读取视频分片"));
      reader.onload = () => resolve(String(reader.result || "").split(",", 2)[1] || "");
      reader.readAsDataURL(blob);
    });
  }

  function stopRecorder(segment) {
    return new Promise((resolve) => {
      const finish = () => {
        const blob = new Blob(segment.chunks, { type: segment.mimeType || "video/webm" });
        segment.stream.getTracks().forEach((track) => track.stop());
        resolve(blob);
      };
      if (segment.recorder.state === "inactive") {
        finish();
        return;
      }
      segment.recorder.addEventListener("stop", finish, { once: true });
      segment.recorder.stop();
    });
  }

  function targetSegmentDuration(segment) {
    const fixedStart = segment.segmentIndex * SEGMENT_SECONDS;
    if (!segment.metadata.duration) return SEGMENT_SECONDS;
    return Math.max(0.25, Math.min(SEGMENT_SECONDS, segment.metadata.duration - fixedStart));
  }

  async function finalizeActiveSegment(reason) {
    const segment = runtime.activeSegment;
    if (!segment) return null;
    runtime.activeSegment = null;
    const blob = await stopRecorder(segment);
    const capturedSeconds = Math.max(
      0,
      Number(segment.lastVideoTime || 0) - Number(segment.firstVideoTime || 0) + 1 / CAPTURE_FPS
    );
    const expectedSeconds = targetSegmentDuration(segment);
    const requiredSeconds = Math.max(0.75, expectedSeconds * 0.8);
    if (capturedSeconds < requiredSeconds || segment.frameCount < 3 || blob.size < 1024) {
      runtime.skippedVisual += 1;
      return { skipped: true, reason: `incomplete_${reason}`, capturedSeconds, expectedSeconds };
    }

    const videoBase64 = await blobToBase64(blob);
    const captionText = Array.from(segment.captionTexts).join(" ").replace(/\s+/g, " ").trim();
    const response = await send({
      type: "VIDEO_SEGMENT",
      payload: {
        ...segment.metadata,
        startTime: segment.segmentIndex * SEGMENT_SECONDS,
        endTime: segment.metadata.duration > 0
          ? Math.min(segment.metadata.duration, (segment.segmentIndex + 1) * SEGMENT_SECONDS)
          : (segment.segmentIndex + 1) * SEGMENT_SECONDS,
        segmentIndex: segment.segmentIndex,
        segmentInterval: SEGMENT_SECONDS,
        videoSegmentSchema: VIDEO_SEGMENT_SCHEMA,
        transcriptSegmentSchema: TRANSCRIPT_SEGMENT_SCHEMA,
        videoBase64,
        videoFilename: `${segment.metadata.sourceSite}-${segment.metadata.videoId}-${segment.segmentIndex}.webm`,
        videoMimeType: segment.mimeType || "video/webm",
        previewBase64: segment.previewBase64,
        previewFilename: `${segment.metadata.sourceSite}-${segment.metadata.videoId}-${segment.segmentIndex}.jpg`,
        captionText,
        capturedSeconds,
        frameCount: segment.frameCount,
        width: segment.width,
        height: segment.height,
        captureFps: Math.round((segment.frameCount / capturedSeconds) * 1000) / 1000,
        retainPreview: runtime.retainPreview,
      },
    });
    if (response?.video?.queued || response?.video?.cached) runtime.indexedVisual += 1;
    if (response?.transcript?.queued || response?.transcript?.cached) runtime.indexedText += 1;
    if (response?.error) runtime.lastError = response.error;
    return response;
  }

  async function startSegment(video, segmentIndex) {
    if (typeof MediaRecorder === "undefined" || typeof HTMLCanvasElement.prototype.captureStream !== "function") {
      throw new Error("当前浏览器不支持视频分片录制");
    }
    const metadata = pageMetadata(video);
    if (!metadata.videoId) return null;
    if (runtime.watchedVideoId !== metadata.videoId) {
      runtime.watchedVideoId = metadata.videoId;
      runtime.watchedSegments.clear();
    }
    const { width, height } = captureSize(video);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { alpha: false });
    const stream = canvas.captureStream(CAPTURE_FPS);
    const mimeType = recorderMimeType();
    const options = { videoBitsPerSecond: VIDEO_BITS_PER_SECOND };
    if (mimeType) options.mimeType = mimeType;
    const recorder = new MediaRecorder(stream, options);
    const segment = {
      metadata,
      segmentIndex,
      canvas,
      context,
      stream,
      recorder,
      mimeType: recorder.mimeType || mimeType || "video/webm",
      chunks: [],
      captionTexts: new Set(),
      firstVideoTime: Number(video.currentTime || 0),
      lastVideoTime: Number(video.currentTime || 0),
      lastDrawVideoTime: -Infinity,
      frameCount: 0,
      width,
      height,
      previewBase64: "",
    };
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data?.size) segment.chunks.push(event.data);
    });
    recorder.addEventListener("error", (event) => {
      runtime.lastError = String(event.error?.message || "视频分片录制失败");
    });
    recorder.start(1000);
    runtime.activeSegment = segment;
    return segment;
  }

  function collectCaption(segment, video = videoElement()) {
    const cueTexts = [];
    for (const track of Array.from(video?.textTracks || [])) {
      for (const cue of Array.from(track.activeCues || [])) {
        const value = String(cue.text || "").trim();
        if (value) cueTexts.push(value);
      }
    }
    const selectors = sourceSite() === "bilibili"
      ? ".bpx-player-subtitle-panel-text, .bpx-player-subtitle-wrap [class*='subtitle-item']"
      : ".ytp-caption-segment";
    const caption = cueTexts.concat(Array.from(document.querySelectorAll(selectors))
      .map((node) => node.textContent?.trim() || "")
      .filter(Boolean))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    if (caption) segment.captionTexts.add(caption);
  }

  function drawFrame(segment, video) {
    const currentTime = Number(video.currentTime || 0);
    if (currentTime - segment.lastDrawVideoTime < 1 / CAPTURE_FPS - 0.03) return;
    segment.context.drawImage(video, 0, 0, segment.width, segment.height);
    segment.frameCount += 1;
    runtime.watchedSegments.add(segment.segmentIndex);
    segment.lastDrawVideoTime = currentTime;
    segment.lastVideoTime = currentTime;
    if (!segment.previewBase64 && currentTime >= segment.segmentIndex * SEGMENT_SECONDS + 4) {
      segment.previewBase64 = segment.canvas.toDataURL("image/jpeg", 0.8).split(",", 2)[1] || "";
    }
  }

  function pauseRecorder() {
    const recorder = runtime.activeSegment?.recorder;
    if (recorder?.state === "recording") recorder.pause();
  }

  function resumeRecorder() {
    const recorder = runtime.activeSegment?.recorder;
    if (recorder?.state === "paused") recorder.resume();
  }

  function isAdPlaying() {
    return sourceSite() === "youtube" && Boolean(document.querySelector("#movie_player.ad-showing"));
  }

  function disableAutoplay() {
    if (sourceSite() !== "youtube") return;
    const toggle = document.querySelector('.ytp-autonav-toggle-button[aria-checked="true"]');
    if (toggle instanceof HTMLElement) toggle.click();
  }

  function skipAdWhenPossible() {
    if (sourceSite() !== "youtube") return;
    const button = document.querySelector(".ytp-skip-ad-button");
    if (button instanceof HTMLElement) button.click();
  }

  async function notifyRebuildVideoComplete(video) {
    const job = runtime.rebuildJob;
    const metadata = pageMetadata(video);
    const current = job?.entries?.[job.index];
    if (!job?.active || !current || current.id !== metadata.videoId) return;
    if (runtime.lastCompletedVideoId === metadata.videoId) return;
    if (metadata.duration <= 60 || video.currentTime < metadata.duration - 0.75) return;
    runtime.lastCompletedVideoId = metadata.videoId;
    video.pause();
    await send({
      type: "CHANNEL_VIDEO_COMPLETE",
      payload: {
        videoId: metadata.videoId,
        duration: metadata.duration,
      },
    });
  }

  async function tickInternal() {
    if (!runtime.running) return;
    const video = videoElement();
    if (!video || video.readyState < 2) return;
    disableAutoplay();
    if (isAdPlaying()) {
      pauseRecorder();
      skipAdWhenPossible();
      return;
    }
    if (video.ended) {
      await finalizeActiveSegment("ended");
      await notifyRebuildVideoComplete(video);
      return;
    }
    if (
      document.visibilityState !== "visible" ||
      video.paused
    ) {
      pauseRecorder();
      return;
    }

    resumeRecorder();
    const currentTime = Number(video.currentTime || 0);
    const segmentIndex = Math.floor(currentTime / SEGMENT_SECONDS);
    const active = runtime.activeSegment;
    const discontinuity = Boolean(
      active &&
      (active.metadata.videoId !== videoId() ||
        currentTime < active.lastVideoTime - 0.75 ||
        currentTime > active.lastVideoTime + 1.5)
    );
    if (active && (active.segmentIndex !== segmentIndex || discontinuity)) {
      await finalizeActiveSegment(discontinuity ? "seek" : "boundary");
    }
    if (!runtime.activeSegment) await startSegment(video, segmentIndex);
    if (!runtime.activeSegment) return;
    collectCaption(runtime.activeSegment, video);
    drawFrame(runtime.activeSegment, video);
  }

  async function tick() {
    ensureFloatingButton();
    if (runtime.tickInFlight) return;
    runtime.tickInFlight = true;
    try {
      await tickInternal();
      void syncVideoSession();
    } catch (error) {
      runtime.lastError = String(error.message || error);
    } finally {
      runtime.tickInFlight = false;
    }
  }

  function ensureTimer() {
    if (runtime.timer) return;
    runtime.timer = window.setInterval(() => void tick(), TICK_MILLISECONDS);
  }

  async function start(options = {}) {
    runtime.running = true;
    runtime.activeIdentity = currentVideoIdentity();
    runtime.retainPreview = options.retainPreview !== false;
    runtime.captureInterval = SEGMENT_SECONDS;
    runtime.lastError = "";
    ensureTimer();
    updateFloatingButton();
    await tick();
    await syncVideoSession(true);
    return publicState();
  }

  async function stop() {
    runtime.running = false;
    await finalizeActiveSegment("manual_stop");
    runtime.activeIdentity = "";
    updateFloatingButton();
    await syncVideoSession(true);
    return publicState();
  }

  async function setCurrentVideoEnabled(enabled, options = {}) {
    const identity = currentVideoIdentity();
    if (!identity) return publicState();
    const stored = await extensionStorageGet({ [VIDEO_OPT_INS_KEY]: {} });
    if (!stored) return publicState();
    const optIns = { ...(stored[VIDEO_OPT_INS_KEY] || {}) };
    if (enabled) optIns[identity] = true;
    else delete optIns[identity];
    await chrome.storage.local.set({ [VIDEO_OPT_INS_KEY]: optIns, autoIndexWatched: false });
    runtime.optedIn = Boolean(enabled);
    runtime.lastError = "";
    if (enabled) {
      await send({ type: "SET_PROCESSING_PAUSED", paused: false });
      if (!runtime.running) await start(options);
    } else if (runtime.running) {
      await stop();
    }
    updateFloatingButton();
    await syncVideoSession(true);
    return publicState();
  }

  async function syncVideoIndexing() {
    const identity = currentVideoIdentity();
    const stored = await extensionStorageGet({ [VIDEO_OPT_INS_KEY]: {} });
    if (!stored) return publicState();
    runtime.optedIn = Boolean(identity && stored[VIDEO_OPT_INS_KEY]?.[identity]);
    const currentRebuild = runtime.rebuildJob?.active
      && runtime.rebuildJob.entries?.[runtime.rebuildJob.index]?.id === videoId();
    if (runtime.optedIn || currentRebuild) {
      if (!runtime.running) await start({ retainPreview: runtime.retainPreview });
    } else if (runtime.running) {
      await stop();
    }
    ensureFloatingButton(true);
    updateFloatingButton();
    return publicState();
  }

  async function syncRebuildJob() {
    const stored = await extensionStorageGet({ youtubeRebuildJob: null });
    if (!stored) return publicState();
    runtime.rebuildJob = stored.youtubeRebuildJob || null;
    const job = runtime.rebuildJob;
    const current = job?.entries?.[job.index];
    if (!job?.active || !current || current.id !== videoId()) return publicState();
    runtime.lastCompletedVideoId = "";
    disableAutoplay();
    if (!runtime.running) await start({ retainPreview: runtime.retainPreview });
    const video = videoElement();
    if (
      video &&
      video.readyState >= 2 &&
      !video.ended &&
      video.paused &&
      document.visibilityState === "visible" &&
      !isAdPlaying()
    ) {
      try {
        await video.play();
      } catch (error) {
        runtime.lastError = `批量任务等待播放：${String(error.message || error)}`;
      }
    }
    return publicState();
  }

  async function requestChannelRebuild() {
    const metadata = pageMetadata();
    if (!metadata.channelUrl) {
      runtime.lastError = "没有识别到当前视频所属频道";
      return null;
    }
    return send({
      type: "START_CHANNEL_REBUILD",
      payload: {
        channelUrl: metadata.channelUrl,
        currentVideoId: metadata.videoId,
      },
    });
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "GET_PAGE_STATE") {
      sendResponse(publicState());
      return false;
    }
    if (message?.type === "START_INDEXING") {
      void setCurrentVideoEnabled(true, message.options).then(sendResponse);
      return true;
    }
    if (message?.type === "STOP_INDEXING") {
      void setCurrentVideoEnabled(false).then(sendResponse);
      return true;
    }
    if (message?.type === "SET_VIDEO_INDEXING") {
      void setCurrentVideoEnabled(Boolean(message.enabled), message.options).then(sendResponse);
      return true;
    }
    if (message?.type === "SYNC_REBUILD_JOB") {
      void syncRebuildJob().then(sendResponse);
      return true;
    }
    return false;
  });

  document.addEventListener("yt-navigate-start", () => {
    if (runtime.running) void finalizeActiveSegment("navigate");
  });
  document.addEventListener("yt-navigate-finish", () => {
    runtime.lastCompletedVideoId = "";
    void syncVideoIndexing();
    void syncRebuildJob();
  });
  window.addEventListener("popstate", () => {
    if (runtime.running) void finalizeActiveSegment("navigate");
    runtime.lastCompletedVideoId = "";
    void syncVideoIndexing();
    void syncRebuildJob();
  });
  document.addEventListener("seeking", () => {
    if (runtime.running) void finalizeActiveSegment("seek");
  }, true);
  window.addEventListener("pagehide", () => {
    if (runtime.running) void finalizeActiveSegment("pagehide");
  });
  document.addEventListener("visibilitychange", () => {
    const video = videoElement();
    if (!video) return;
    if (document.visibilityState !== "visible" && !video.paused && !video.ended) {
      runtime.resumeOnVisible = true;
      video.pause();
      pauseRecorder();
      return;
    }
    if (document.visibilityState === "visible" && runtime.resumeOnVisible) {
      runtime.resumeOnVisible = false;
      void syncRebuildJob();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.altKey && event.shiftKey && event.code === "KeyV") {
      event.preventDefault();
      void requestChannelRebuild();
    }
  }, true);
  if (chrome.storage?.onChanged) {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local" && changes[VIDEO_OPT_INS_KEY]) void syncVideoIndexing();
      if (area === "local" && changes.youtubeRebuildJob) void syncRebuildJob();
    });
  }
  ensureTimer();
  ensureFloatingButton(true);
  void syncVideoIndexing();
  void syncRebuildJob();
})();
