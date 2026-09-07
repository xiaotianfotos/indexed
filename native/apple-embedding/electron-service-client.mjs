import { randomBytes } from "node:crypto";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { dirname } from "node:path";

/** Electron main-process client for the Indexed Apple embedding helper. */
export class AppleEmbeddingService {
  constructor({
    binary,
    modelPackage,
    decoderSegment,
    decoderBundle,
    decoderBundles = [],
    decoderMinimumTokens,
    coreMLCache,
    languageCompute = "auto",
    visionCompute = "ane",
    defaultDimension = 2048,
    maxQueuedRequests = 16,
    onStderr,
    extraArgs = [],
  } = {}) {
    this.binary = binary;
    this.modelPackage = modelPackage;
    this.decoderSegment = decoderSegment;
    this.decoderBundles = [
      ...decoderBundles,
      ...(decoderBundle ? [decoderBundle] : []),
    ];
    this.decoderMinimumTokens = decoderMinimumTokens;
    this.coreMLCache = coreMLCache;
    this.languageCompute = languageCompute;
    this.visionCompute = visionCompute;
    this.defaultDimension = defaultDimension;
    this.maxQueuedRequests = maxQueuedRequests;
    this.onStderr = onStderr ?? (() => {});
    this.extraArgs = [...extraArgs];
    this.token = randomBytes(32).toString("hex");
    this.child = null;
    this.url = null;
    this.readyInfo = null;
    this.startPromise = null;
  }

  async start({ timeoutMs = 300_000 } = {}) {
    if (this.readyInfo) return this.readyInfo;
    if (this.startPromise) return this.startPromise;
    if (!this.binary || !this.modelPackage) {
      throw new Error("binary and modelPackage are required");
    }
    this.startPromise = this.#start(timeoutMs);
    try {
      return await this.startPromise;
    } finally {
      this.startPromise = null;
    }
  }

  async #start(timeoutMs) {
    const args = [
      "serve",
      "--package", this.modelPackage,
      "--port", "0",
      "--default-dimension", String(this.defaultDimension),
      "--language-compute", this.languageCompute,
      "--vision-compute", this.visionCompute,
      "--max-queued-requests", String(this.maxQueuedRequests),
    ];
    if (this.decoderSegment) args.push("--decoder-segment", this.decoderSegment);
    for (const bundle of this.decoderBundles) args.push("--decoder-bundle", bundle);
    if (this.decoderMinimumTokens !== undefined) {
      args.push("--decoder-minimum-tokens", String(this.decoderMinimumTokens));
    }
    if (this.coreMLCache) args.push("--coreml-cache", this.coreMLCache);
    args.push(...this.extraArgs);
    const child = spawn(this.binary, args, {
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
      cwd: dirname(this.binary),
      env: {
        ...process.env,
        INDEXED_APPLE_EMBEDDING_AUTH_TOKEN: this.token,
      },
    });
    this.child = child;
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", this.onStderr);

    const ready = new Promise((resolve, reject) => {
      let buffer = "";
      child.stdout.setEncoding("utf8");
      child.stdout.on("data", (chunk) => {
        buffer += chunk;
        for (;;) {
          const newline = buffer.indexOf("\n");
          if (newline < 0) break;
          const line = buffer.slice(0, newline).trim();
          buffer = buffer.slice(newline + 1);
          if (!line) continue;
          try {
            const value = JSON.parse(line);
            if (value.status === "ready" && value.url) resolve(value);
          } catch (error) {
            reject(new Error(`Native helper emitted invalid ready JSON: ${error.message}`));
          }
        }
      });
      child.once("error", reject);
      child.once("exit", (code, signal) => {
        if (!this.url) reject(new Error(`Native helper exited before ready: ${code ?? signal}`));
        if (this.child === child) {
          this.child = null;
          this.url = null;
          this.readyInfo = null;
        }
      });
    });
    const timeout = new Promise((_, reject) => {
      setTimeout(() => reject(new Error("Native helper startup timed out")), timeoutMs).unref();
    });
    try {
      const value = await Promise.race([ready, timeout]);
      this.url = value.url;
      this.readyInfo = value;
      return value;
    } catch (error) {
      await this.stop();
      throw error;
    }
  }

  async health() {
    return this.#json("/health");
  }

  async prepare({ timeoutMs = 600_000 } = {}) {
    if (this.child) throw new Error("Stop the native helper before preparing Core ML assets");
    if (!this.binary || !this.modelPackage) {
      throw new Error("binary and modelPackage are required");
    }
    const args = ["prepare-coreml", "--package", this.modelPackage];
    if (this.decoderSegment) args.push("--decoder-segment", this.decoderSegment);
    for (const bundle of this.decoderBundles) args.push("--decoder-bundle", bundle);
    if (this.coreMLCache) args.push("--coreml-cache", this.coreMLCache);
    const child = spawn(this.binary, args, {
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
      cwd: dirname(this.binary),
      env: process.env,
    });
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", this.onStderr);
    let stdout = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    const result = new Promise((resolve, reject) => {
      child.once("error", reject);
      child.once("exit", (code, signal) => {
        if (code !== 0) {
          reject(new Error(`Core ML preparation failed: ${code ?? signal}`));
          return;
        }
        try {
          const line = stdout.trim().split("\n").filter(Boolean).at(-1);
          const value = JSON.parse(line);
          if (value.status !== "prepared") throw new Error("Unexpected preparation response");
          resolve(value);
        } catch (error) {
          reject(error);
        }
      });
    });
    const timeout = new Promise((_, reject) => {
      setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error("Core ML preparation timed out"));
      }, timeoutMs).unref();
    });
    return Promise.race([result, timeout]);
  }

  async embed({ messages, input, model = "wemm-embedding-2b-apple-2048", requestId, signal }) {
    if (!this.url) throw new Error("Native helper is not started");
    if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
    const id = requestId ?? randomBytes(16).toString("hex");
    let aborted = false;
    const cancel = () => {
      aborted = true;
      void this.#cancelRequest(id);
    };
    signal?.addEventListener("abort", cancel, { once: true });
    try {
      const value = await this.#json("/v1/embeddings", {
        method: "POST",
        body: JSON.stringify({ model, messages, input, request_id: id }),
      });
      if (aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
      return value;
    } finally {
      signal?.removeEventListener("abort", cancel);
    }
  }

  async stop({ timeoutMs = 10_000 } = {}) {
    const child = this.child;
    this.child = null;
    this.url = null;
    this.readyInfo = null;
    this.startPromise = null;
    if (!child || child.exitCode !== null) return;
    child.kill("SIGTERM");
    const exited = once(child, "exit");
    const timeout = new Promise((resolve) => {
      setTimeout(resolve, timeoutMs, "timeout").unref();
    });
    if (await Promise.race([exited, timeout]) === "timeout" && child.exitCode === null) {
      child.kill("SIGKILL");
      await once(child, "exit");
    }
  }

  async #cancelRequest(id) {
    for (let attempt = 0; attempt < 5; attempt += 1) {
      try {
        await this.#json(`/v1/requests/${encodeURIComponent(id)}`, { method: "DELETE" });
        return;
      } catch (error) {
        if (error.status !== 404 || attempt === 4) return;
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
    }
  }

  async #json(path, options = {}) {
    if (!this.url) throw new Error("Native helper is not started");
    const response = await fetch(`${this.url}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.token}`,
        ...options.headers,
      },
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(body?.error?.message ?? `Native helper HTTP ${response.status}`);
      error.status = response.status;
      error.payload = body;
      throw error;
    }
    return body;
  }
}
