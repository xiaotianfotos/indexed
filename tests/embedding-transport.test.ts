import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import http from "node:http";
import test, { type TestContext } from "node:test";
import { postEmbedding } from "@indexed/clients/embedding-transport";
import { OperationScope, waitForRetry, withOperation, throwIfAborted } from "@indexed/clients/operation";

async function fixture(t: TestContext, handler: http.RequestListener) {
  const server = http.createServer(handler);
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  t.after(() => { server.closeAllConnections(); server.close(); });
  const address = server.address();
  assert(address && typeof address === "object");
  return `http://127.0.0.1:${address.port}`;
}
const payload = { model: "fixture-model", messages: [{ role: "user", content: "fixed input" }] };
const headers = { "content-type": "application/json", authorization: "Bearer fixture-token" };

test("native abort cancels the exact upstream attempt, including a response body that has not finished", async t => {
  let ready!: () => void;
  const started = new Promise<void>(resolve => { ready = resolve; });
  let requestID = "";
  const cancelled: string[] = [];
  const url = await fixture(t, (req, res) => {
    assert.equal(req.headers.authorization, headers.authorization);
    if (req.method === "DELETE") { cancelled.push(req.url!.split("/").at(-1)!); res.writeHead(202).end(); return; }
    let body = "";
    req.on("data", chunk => { body += String(chunk); });
    req.on("end", () => {
      const input = JSON.parse(body);
      requestID = input.request_id;
      assert.match(requestID, /^[a-f0-9-]{36}\.[a-f0-9-]{36}$/);
      assert.deepEqual(input.messages, payload.messages);
      assert(Number(req.headers["x-indexed-timeout-ms"]) <= 2000);
      res.writeHead(200, { "content-type": "application/json" }); res.write('{"data":'); ready();
    });
  });
  const controller = new AbortController();
  const result = postEmbedding(url, payload, headers, { native: true, signal: controller.signal, timeoutMs: 2000 });
  const rejected = assert.rejects(result, { name: "AbortError" });
  await started; controller.abort(); await rejected;
  assert.deepEqual(cancelled, [requestID]);
});

test("retry attempts keep one total deadline and new native IDs without altering the workload", async t => {
  const ids: string[] = [];
  const budgets: number[] = [];
  const url = await fixture(t, (req, res) => {
    let body = ""; req.on("data", chunk => { body += String(chunk); });
    req.on("end", () => {
      const input = JSON.parse(body); assert.deepEqual(input.messages, payload.messages);
      ids.push(input.request_id); budgets.push(Number(req.headers["x-indexed-timeout-ms"]));
      if (ids.length === 1) res.writeHead(503, { "retry-after": "0.02" }).end("busy");
      else res.writeHead(200).end("ok");
    });
  });
  assert.equal((await postEmbedding(url, payload, headers, { native: true, timeoutMs: 2000 })).text, "ok");
  assert.equal(ids.length, 2); assert.notEqual(ids[0], ids[1]); assert.equal(ids[0]!.split(".")[0], ids[1]!.split(".")[0]); assert(budgets[1]! < budgets[0]!);
  let calls = 0;
  const slow = await fixture(t, (_req, res) => { calls++; res.writeHead(503, { "retry-after": "5" }).end("busy"); });
  await assert.rejects(postEmbedding(slow, payload, headers, { timeoutMs: 60 }), { name: "TimeoutError", status: 504 });
  assert.equal(calls, 1);
});

test("native deadline cancels a stalled helper; external endpoints receive no private protocol", async t => {
  let cancellations = 0;
  const native = await fixture(t, (req, res) => {
    if (req.method === "DELETE") { cancellations++; res.writeHead(202).end(); }
    else req.resume();
  });
  await assert.rejects(postEmbedding(native, payload, headers, { native: true, timeoutMs: 80 }), { name: "TimeoutError" });
  assert.equal(cancellations, 1);
  const remote = await fixture(t, (req, res) => {
    assert.equal(req.method, "POST"); assert.equal(req.headers["x-indexed-timeout-ms"], undefined);
    let body = ""; req.on("data", chunk => { body += String(chunk); });
    req.on("end", () => { assert.deepEqual(JSON.parse(body), payload); res.end("ok"); });
  });
  await postEmbedding(remote, payload, headers);
});

test("operation/retry listeners are removed, parent cancellation propagates, and early failure stops siblings", async () => {
  const parent = new AbortController();
  for (let i = 0; i < 10; i++) await waitForRetry(1, parent.signal);
  assert.equal(getEventListeners(parent.signal, "abort").length, 0);
  const scope = new OperationScope({ signal: parent.signal, timeoutMs: 1000 });
  const nested = new OperationScope({ signal: scope.signal, timeoutMs: 600000 });
  assert.equal(nested.requestID, scope.requestID); assert(nested.remainingMS <= scope.remainingMS + 1); nested.dispose();
  parent.abort(); assert.equal(scope.signal.aborted, true); scope.dispose();
  assert.equal(getEventListeners(parent.signal, "abort").length, 0);
  let sibling: AbortSignal | undefined;
  await assert.rejects(withOperation({}, async operation => { sibling = operation.signal; throw new Error("failure"); }), /failure/);
  assert.equal(sibling?.aborted, true);
  let wrote = false;
  await assert.rejects(withOperation({ timeoutMs: 2 }, async operation => {
    const end = performance.now() + 5; while (performance.now() < end) { /* Simulate a synchronous storage/CPU boundary. */ }
    throwIfAborted(operation.signal); wrote = true;
  }), { name: "TimeoutError" });
  assert.equal(wrote, false);
  for (const timeoutMs of [0, -1, 1.2, NaN, 600001]) assert.throws(() => new OperationScope({ timeoutMs }));
});
