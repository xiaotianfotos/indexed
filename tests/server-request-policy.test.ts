import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { loadConfig, writeConfig } from "@indexed/config";
import { startServer } from "@indexed/server";
import { bundledExtensionOrigins } from "../apps/server/src/extension-origin.js";
import { localListenHost } from "../apps/server/src/request-policy.js";

test("preview listener accepts loopback addresses without resolving arbitrary hostnames", () => {
  for (const [input, expected] of [["127.0.0.1", "127.0.0.1"], ["127.0.0.2", "127.0.0.2"],
    ["LOCALHOST", "127.0.0.1"], ["localhost.", "127.0.0.1"], ["::1", "::1"], ["[::1]", "::1"]]) {
    assert.equal(localListenHost(input!), expected);
  }
  for (const host of ["0.0.0.0", "::", "10.0.0.1", "example.com", "127.0.0.1.example.com", "127.0.0.999", "::ffff:10.0.0.1"]) {
    assert.throws(() => localListenHost(host), /仅支持本机/);
  }
});

function request(url: string, options: http.RequestOptions = {}, chunks: readonly string[] = []) {
  return new Promise<{ status: number; text: string }>((resolve, reject) => {
    const outgoing = http.request(url, options, (response) => {
      const result: Buffer[] = [];
      response.on("data", (chunk: Buffer) => result.push(chunk));
      response.on("error", reject);
      response.on("end", () => resolve({ status: response.statusCode || 0, text: Buffer.concat(result).toString("utf8") }));
    });
    outgoing.setTimeout(10_000, () => outgoing.destroy(new Error("request timed out")));
    outgoing.on("error", reject);
    for (const chunk of chunks) outgoing.write(chunk);
    outgoing.end();
  });
}

test("local API enforces browser origins and bounded JSON before side effects", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-request-policy-"));
  const previous = Object.fromEntries(["INDEXED_CONFIG", "INDEXED_STATE_DIR", "INDEXED_DATA_DIR"].map((key) => [key, process.env[key]]));
  process.env.INDEXED_CONFIG = path.join(directory, "config.json");
  process.env.INDEXED_STATE_DIR = path.join(directory, "state");
  process.env.INDEXED_DATA_DIR = path.join(directory, "vectors");
  const config = loadConfig();
  config.library = { ...config.library, roots: [], autoScan: false };
  writeConfig(config);
  await assert.rejects(startServer({ host: "0.0.0.0", port: 0 }), /仅支持本机/);
  const previousHost = config.server.host;
  config.server.host = "0.0.0.0";
  writeConfig(config);
  await assert.rejects(startServer({ port: 0 }), /仅支持本机/);
  config.server.host = previousHost;
  writeConfig(config);
  const started = await startServer({ host: "127.0.0.1", port: 0 });
  context.after(async () => {
    await started.close();
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    fs.rmSync(directory, { recursive: true, force: true });
  });
  const base = started.url;
  const origin = new URL(base).origin;
  const library = path.join(directory, "library");
  fs.mkdirSync(library);
  const payload = JSON.stringify({ path: library });

  await context.test("CLI, same-origin Dashboard and exact bundled extension remain usable", async () => {
    const extensions = bundledExtensionOrigins(path.resolve("dist/dashboard"));
    assert.equal(extensions.length, 1, "build must provide the extension's stable manifest key");
    for (const headers of [
      {},
      { origin },
      { referer: `${base}settings`, "sec-fetch-site": "same-origin" },
      { origin: extensions[0]! },
      { host: `localhost:${started.port}` },
    ]) {
      const result = await request(`${base}api/config`, { headers });
      assert.equal(result.status, 200, JSON.stringify(headers));
    }
    assert.equal((await request(base, { headers: { "sec-fetch-site": "none" } })).status, 200);
    const activated = await request(`${base}api/profiles/default/activate`, { method: "POST" });
    assert.equal(activated.status, 200, "bodyless CLI operations stay compatible");
  });

  await context.test("foreign Host, port, Origin, Referer and opaque origins are rejected", async () => {
    for (const headers of [
      { host: `review.invalid:${started.port}` },
      { host: "127.0.0.1:1" },
      { origin: "https://review.invalid" },
      { origin: "null" },
      { origin: "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
      { referer: "https://review.invalid/page" },
      { "sec-fetch-site": "cross-site" },
    ]) {
      for (const route of ["api/config", "app.js"]) {
        const result = await request(`${base}${route}`, { headers });
        assert.equal(result.status, 403, JSON.stringify(headers));
        assert.equal("configPath" in JSON.parse(result.text), false);
      }
    }
    assert.equal((await request(`${base}api/config`, { headers: { host: "user@127.0.0.1" } })).status, 400);
    const crossSiteWrite = await request(`${base}api/assets/libraries`, {
      method: "POST", headers: { origin: "https://review.invalid", "content-type": "text/plain" },
    }, [payload]);
    assert.equal(crossSiteWrite.status, 403);
    assert.deepEqual(loadConfig().library.roots, []);
  });

  await context.test("JSON type and shape are validated before registering a directory", async () => {
    for (const contentType of ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data", "application/jsonp"]) {
      const result = await request(`${base}api/assets/libraries`, {
        method: "POST", headers: { "content-type": contentType },
      }, [payload]);
      assert.equal(result.status, 415, contentType);
    }
    for (const invalid of ["{", "null", "[]", "true", "42", '"text"']) {
      const result = await request(`${base}api/assets/libraries`, {
        method: "POST", headers: { "content-type": "application/json" },
      }, [invalid]);
      assert.equal(result.status, 400, invalid);
      assert.match(result.text, /JSON/);
    }
    assert.deepEqual(loadConfig().library.roots, []);
  });

  await context.test("declared and streamed oversized bodies return structured 413", async () => {
    const declared = await request(`${base}api/assets/libraries`, {
      method: "POST", headers: { "content-type": "application/json", "content-length": String(2 * 1024 * 1024 + 1) },
    });
    assert.equal(declared.status, 413);
    assert.match(declared.text, /大小限制/);
    const streamed = await request(`${base}api/assets/libraries`, {
      method: "POST", headers: { "content-type": "application/json", "transfer-encoding": "chunked" },
    }, ['{"padding":"', "x".repeat(1024 * 1024), "x".repeat(1024 * 1024), '"}']);
    assert.equal(streamed.status, 413);
    assert.match(streamed.text, /大小限制/);
    const media = await request(`${base}api/ingest/video`, {
      method: "POST", headers: { "content-type": "application/json", "content-length": String(64 * 1024 * 1024 + 1) },
    });
    assert.equal(media.status, 413);
    assert.deepEqual(loadConfig().library.roots, []);
  });

  await context.test("valid JSON still performs the intended write after rejected requests", async () => {
    const result = await request(`${base}api/assets/libraries`, {
      method: "POST", headers: { origin, "content-type": "application/json; charset=utf-8" },
    }, [payload]);
    assert.equal(result.status, 200, result.text);
    assert.deepEqual(loadConfig().library.roots, [library]);
  });
});

test("extension allowlist follows the bundled key and does not trust an unkeyed build", (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-extension-origin-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const dashboard = path.join(directory, "dashboard");
  const extension = path.join(directory, "extension");
  fs.mkdirSync(path.join(extension, "static"), { recursive: true });
  assert.deepEqual(bundledExtensionOrigins(dashboard), []);
  // SHA256("abc") starts ba7816bf8f01cfea414140de5dae2223.
  fs.writeFileSync(path.join(extension, "static/manifest.json"), JSON.stringify({ key: "YWJj" }));
  assert.deepEqual(bundledExtensionOrigins(dashboard), ["chrome-extension://lkhibglpipabmpokebebeanofnkocccd"]);
  fs.writeFileSync(path.join(extension, "manifest.json"), JSON.stringify({}));
  assert.deepEqual(bundledExtensionOrigins(dashboard), [], "do not fall back to a different identity when a build exists");
});
