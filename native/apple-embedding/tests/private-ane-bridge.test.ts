import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import test from "node:test";

const run = promisify(execFile);
const binary = path.resolve(process.env.INDEXED_PRIVATE_ANE_CHECK_BINARY || "native/apple-embedding/swift/.build/release/indexed-private-ane-check");

test("compiled Swift bridge validates handles, quantization and frozen GDN graph/quality profile without private hardware", async () => {
  const { stdout } = await run(binary, [], { timeout: 10_000 });
  assert.deepEqual(JSON.parse(stdout), { bounds: "passed", hardware: "not-requested" });
});

test("Swift private GDN matches a sequential CPU oracle and preserves chunk/state/mask/layer/cancellation behavior", { skip: process.env.INDEXED_TEST_PRIVATE_ANE !== "1" }, async () => {
  const { stdout } = await run(binary, ["--gdn"], { timeout: 120_000 });
  assert.deepEqual(JSON.parse(stdout), { gdn: "passed", values: 661504, oracle: "sequential-cpu", routing: "passed" });
});

test("experimental native bridge executes FP16 and INT8 linear graphs on real ANE", { skip: process.env.INDEXED_TEST_PRIVATE_ANE !== "1" }, async () => {
  const { stdout } = await run(binary, ["--hardware"], { timeout: 60_000 });
  assert.deepEqual(JSON.parse(stdout), { bounds: "passed", hardware: "passed", values: 40960, linear: "passed", shared_workspace: "passed", cancel: "passed" });
});

test("Swift hybrid MLP preserves FP16/BF16 channel partition, tiling, fallback and cancellation on hardware", { skip: process.env.INDEXED_TEST_PRIVATE_ANE !== "1" }, async () => {
  const { stdout } = await run(binary, ["--hybrid-mlp"], { timeout: 120_000 });
  assert.deepEqual(JSON.parse(stdout), { hybrid_mlp: "passed", values: 1789696, dtypes: ["float16", "bfloat16"], routing: "passed" });
});

test("Swift stage admission, FIFO/priority, cancellation and CPU patches preserve single-owner inference and exact legacy preprocessing", async () => {
  const { stdout } = await run(binary, ["--pipeline-check"], { timeout: 30_000 });
  assert.deepEqual(JSON.parse(stdout), { scheduling: "passed", cpu_patches: "bit-exact", values: 2420736 });
});
