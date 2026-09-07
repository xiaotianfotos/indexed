import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import {
  backendModePresentation,
  saveBeforeConnectionTest,
} from "../apps/extension/src/settings-flow.js";
import {
  queueItemReadyNow,
  queueStatusPresentation,
} from "../apps/extension/src/queue-state.js";
import { applyVideoActivity } from "../apps/extension/src/library-state.js";

test("backend selector explains that local and cloud are mutually exclusive", () => {
  assert.match(backendModePresentation("local").help, /不需要填写阿里云/);
  assert.match(backendModePresentation("cloud").help, /不需要本地 Indexed 服务/);
  const html = fs.readFileSync("apps/extension/static/popup.html", "utf8");
  assert.match(html, /id="backend-cloud" type="radio" name="backend-mode"/);
  assert.match(html, /id="backend-local" type="radio" name="backend-mode"/);
});

test("settings are saved before an unavailable backend is tested", async () => {
  const events: string[] = [];
  const result = await saveBeforeConnectionTest(
    async () => {
      events.push("save");
      return true;
    },
    async () => {
      events.push("test");
      return false;
    },
  );
  assert.deepEqual(events, ["save", "test"]);
  assert.deepEqual(result, { saved: true, valid: false });
});

test("connection testing does not run when persistence fails", async () => {
  let tested = false;
  const result = await saveBeforeConnectionTest(
    async () => false,
    async () => {
      tested = true;
      return true;
    },
  );
  assert.equal(tested, false);
  assert.deepEqual(result, { saved: false, valid: null });
});

test("changing backend makes every queued item immediately retryable", () => {
  assert.deepEqual(queueItemReadyNow({ id: "one", attempts: 6, nextAttempt: 999, lastError: "fetch failed" }), {
    id: "one",
    attempts: 0,
    terminal: false,
    lastError: "",
    lastErrorAt: 0,
    nextAttempt: 0,
  });
});

test("queue failures are visible instead of looking permanently pending", () => {
  assert.deepEqual(queueStatusPresentation({ queued_total: 4, errors: 4, last_error: "fetch failed" }), {
    visible: true,
    tone: "error",
    text: "队列处理失败 · 4 项等待重试 · fetch failed",
    title: "fetch failed",
  });
});

test("local library timestamps participate in the same recent activity model as cloud", () => {
  const videos = applyVideoActivity([
    { video_id: "local", last_indexed_at: 1_700_000_000_000, visual_count: 2 },
    { video_id: "queued", last_indexed_at: 0, visual_count: 0 },
  ], [
    { videoId: "queued", createdAt: 1_700_000_000_100 },
  ], []);
  assert.deepEqual(videos.map((video) => [video.video_id, video.last_processed_at]), [
    ["queued", 1_700_000_000_100],
    ["local", 1_700_000_000_000],
  ]);
});

test("video memory opens the same management UI for local and cloud backends", () => {
  const popup = fs.readFileSync("apps/extension/src/popup.ts", "utf8");
  const search = fs.readFileSync("apps/extension/static/search.html", "utf8");
  assert.match(popup, /elements\.library\.addEventListener\("click".*search\.html/);
  assert.match(search, /管理当前后端中已经保存的画面与字幕索引/);
  assert.doesNotMatch(search, /云端向量删除后/);
});
