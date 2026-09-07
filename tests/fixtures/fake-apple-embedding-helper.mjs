#!/usr/bin/env node

import http from "node:http";
import fs from "node:fs";

const args = process.argv.slice(2);
const command = args[0];
const value = (flag, fallback = "") => {
  const index = args.indexOf(flag);
  return index >= 0 && args[index + 1] !== undefined ? args[index + 1] : fallback;
};

if (command === "validate-model") {
  process.stdout.write(`${JSON.stringify({ status: "valid", depth: args.includes("--full") ? "full" : "quick", package_fingerprint: "fake-package", model: "fake-wemm" })}\n`);
} else if (command === "prepare-coreml") {
  process.stdout.write(`${JSON.stringify({ status: "prepared", package_fingerprint: "fake-package", vision_compiled_path: "/tmp/fake.mlmodelc", decoder_bundle_fingerprints: [] })}\n`);
} else if (command === "serve" || args.includes("--execution-mode")) {
  const dimension = Number(value("--default-dimension", "2048"));
  const advertisedDimension = Number(value("--advertised-dimension", String(dimension)));
  const model = `wemm-embedding-2b-apple-${dimension}`;
  const embeddingSpace = `fake-wemm-${dimension}-same-space`;
  const token = String(process.env.INDEXED_APPLE_EMBEDDING_AUTH_TOKEN || "");
  const server = http.createServer(async (request, response) => {
    if (token && request.headers.authorization !== `Bearer ${token}`) {
      response.writeHead(401, { "content-type": "application/json" });
      response.end(JSON.stringify({ error: { message: "unauthorized" } }));
      return;
    }
    if (request.method === "GET" && request.url === "/health") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({
        status: "ready",
        backend: "fake-apple-native",
        execution_mode: value("--execution-mode"),
        vision_compute: value("--vision-compute"),
        language_compute: value("--language-compute", "gpu"),
        private_ane_mode: value("--private-ane-mode"),
        recurrence_block_size: Number(value("--private-ane-recurrence-block-size", "0")),
        video_pipeline: Number(value("--video-pipeline", "1")),
        video_down_projection: value("--video-down-projection", "q8"),
        recurrence_layer_slots: value("--private-ane-recurrence-layer-slots"),
        recurrence_query_scale: Number(value("--private-ane-recurrence-query-scale", "0")),
        recurrence_max_tokens: Number(value("--private-ane-recurrence-max-tokens", "0")),
        recurrence_verify_reference: args.includes("--private-ane-recurrence-verify-reference"),
        max_queued_requests: Number(value("--max-queued-requests", "16")),
      }));
      return;
    }
    if (request.method === "GET" && request.url === "/v1/models") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ object: "list", data: [{ id: model, dimension: advertisedDimension, embedding_space: embeddingSpace, max_model_len: 256 }] }));
      return;
    }
    if (request.method === "POST" && request.url === "/v1/embeddings") {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      const input = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      const vector = Array.from({ length: dimension }, (_, index) => index === 0 ? 1 : 0);
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({
        object: "list",
        model: input.model || model,
        data: [{ object: "embedding", index: 0, embedding: vector }],
        usage: { prompt_tokens: 1, total_tokens: 1 },
        indexed: { embedding_space: embeddingSpace, backend: "fake-apple-native" },
      }));
      return;
    }
    response.writeHead(404, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: { message: "not found" } }));
  });
  server.listen(Number(value("--port", "0")), "127.0.0.1", () => {
    const address = server.address();
    setTimeout(() => {
      process.stdout.write(`${JSON.stringify({
        status: "ready",
        url: `http://127.0.0.1:${address.port}`,
        default_model: model,
        default_embedding_space: embeddingSpace,
        package_fingerprint: "fake-package",
        backend: "fake-apple-native",
        execution_mode: value("--execution-mode"),
        load_seconds: 0.01,
      })}\n`);
      const crashMarker = value("--crash-once-marker");
      if (crashMarker && !fs.existsSync(crashMarker)) {
        fs.writeFileSync(crashMarker, "crashed\n");
        setTimeout(() => process.exit(42), 50).unref();
      }
    }, Number(value("--ready-delay-ms", "0")));
  });
  const stop = () => server.close(() => process.exit(0));
  process.on("SIGTERM", stop);
  process.on("SIGINT", stop);
} else {
  process.stderr.write(`unknown command: ${command}\n`);
  process.exitCode = 64;
}
