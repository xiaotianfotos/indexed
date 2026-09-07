import assert from "node:assert/strict";
import test from "node:test";
import { LatestRequest } from "../apps/dashboard/src/latest-request.js";
import { setupPresentation, type SetupState } from "../apps/dashboard/src/setup-presentation.js";

test("superseded and cleared searches cannot render even if their transport ignores cancellation", async () => {
  const requests = new LatestRequest();
  const rendered: string[] = [];
  let finishOld!: () => void;
  const old = requests.begin();
  const pending = new Promise<void>(resolve => { finishOld = resolve; }).then(() => {
    if (old.current()) rendered.push("old");
  });
  const fresh = requests.begin();
  assert.equal(old.signal.aborted, true);
  if (fresh.current()) rendered.push("new");
  finishOld();
  await pending;
  assert.deepEqual(rendered, ["new"]);
  requests.cancel();
  assert.equal(fresh.current(), false);
  assert.equal(fresh.signal.aborted, true);
});

test("first-use guidance distinguishes setup, loading, empty libraries and usable partial scans", () => {
  const input: SetupState = { phase: "configure", detail: "填写模型地址", directories: 0, indexed: 0, scanning: false };
  assert.equal(setupPresentation(input).action, "engine");
  assert.equal(setupPresentation({ ...input, phase: "loading" }).searchable, false);
  assert.equal(setupPresentation({ ...input, phase: "ready" }).action, "add");
  assert.equal(setupPresentation({ ...input, phase: "ready", directories: 1 }).action, "assets");
  assert.equal(setupPresentation({ ...input, phase: "ready", scanning: true }).searchable, false);
  assert.equal(setupPresentation({ ...input, phase: "ready", scanning: true, indexed: 1 }).searchable, true);
  assert.equal(setupPresentation({ ...input, phase: "offline", indexed: 1 }).searchable, false);
});
