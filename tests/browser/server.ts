import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = process.env.INDEXED_HOME;
if (!root) throw new Error("The browser fixture requires an isolated INDEXED_HOME");
const directory = path.join(root, "素材");
fs.mkdirSync(directory, { recursive: true });
fs.writeFileSync(path.join(directory, "窗边的猫.txt"), "一只猫坐在窗边，看着花园。\n");
const embedding = http.createServer(async (request, response) => {
  for await (const _chunk of request) { /* Drain the fixture request. */ }
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify(request.url?.endsWith("/models")
    ? { data: [{ id: "dashboard-fixture", dimension: 3, embedding_space: "dashboard-fixture-space" }] }
    : { data: [{ embedding: [1, 0, 0] }] }));
});
await new Promise<void>(resolve => embedding.listen(0, "127.0.0.1", resolve));
const address = embedding.address();
if (!address || typeof address === "string") throw new Error("Missing embedding fixture port");
// Exercise the same bundled server shipped by the normal build.
const { startServer } = await import(pathToFileURL(path.resolve("dist/server/index.js")).href);
const app = await startServer({ host: "127.0.0.1", port: 0 });
let stopping = false;
async function stop() {
  if (stopping) return;
  stopping = true;
  await app.close();
  embedding.closeAllConnections();
  await new Promise<void>(resolve => embedding.close(() => resolve()));
  process.exit(0);
}
process.once("SIGTERM", () => void stop());
process.once("SIGINT", () => void stop());
process.once("disconnect", () => void stop());
process.send?.({ url: app.url, embeddingUrl: `http://127.0.0.1:${address.port}`, directory });
