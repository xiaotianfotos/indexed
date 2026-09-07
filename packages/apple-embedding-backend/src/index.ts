import { requireAppleExecutionMode, type AppleExecutionMode } from "@indexed/contracts";
export type { AppleExecutionMode } from "@indexed/contracts";
import { randomBytes } from "node:crypto";
import { spawn, type ChildProcessByStdio } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import type { Readable } from "node:stream";

export const APPLE_EMBEDDING_PROVIDER = "apple-native";

export type AppleEmbeddingMode = "fast";
export type AppleVisionCompute = "ane" | "gpu";

export interface ApplePrivateANEKernelOptions {
  videoDownProjection?: "q8" | "fp16";
  videoPipeline?: 1 | 2;
  sequenceLength?: number;
  mlpFraction?: number;
  mlpVariant?: number;
  mlpMaxLayers?: number;
  recurrenceProfile?: string;
  recurrenceBlockSize?: number;
  recurrenceLayerSlots?: number[];
  recurrenceQueryScale?: number;
  recurrenceMaxTokens?: number;
  recurrenceIODtype?: "fp16" | "fp32";
  recurrenceVerifyReference?: boolean;
}

export interface AppleEmbeddingBackendOptions {
  binary?: string;
  modelPackage: string;
  coreMLCache?: string;
  mode?: AppleEmbeddingMode;
  visionCompute?: AppleVisionCompute;
  executionMode?: AppleExecutionMode;
  privateANE?: ApplePrivateANEKernelOptions;
  dimension?: number;
  maxQueuedRequests?: number;
  startupTimeoutMs?: number;
  maintenanceTimeoutMs?: number;
  autoRestart?: boolean;
  maxRestarts?: number;
  expectedModel?: string;
  expectedEmbeddingSpace?: string;
  extraArgs?: string[];
  onStateChange?: (status: AppleEmbeddingBackendStatus) => void;
  onStderr?: (line: string) => void;
}

export interface AppleEmbeddingRuntimeConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  dimension: number;
  embeddingSpace: string;
  inputStyle: "wemm";
}

export interface AppleEmbeddingBackendStatus {
  managed: true;
  state: "stopped" | "starting" | "ready" | "restarting" | "failed";
  mode: AppleEmbeddingMode;
  visionCompute: AppleVisionCompute;
  executionMode: AppleExecutionMode;
  pid: number | null;
  baseUrl: string;
  model: string;
  dimension: number;
  embeddingSpace: string;
  startedAt: string | null;
  restartCount: number;
  lastExit: { code: number | null; signal: NodeJS.Signals | null } | null;
  lastError: string;
  stderrTail: string[];
  ready: Record<string, unknown> | null;
}

const DEFAULT_STARTUP_TIMEOUT_MS = 300_000;
const DEFAULT_MAINTENANCE_TIMEOUT_MS = 600_000;
const STDERR_TAIL_LINES = 80;
const OFFICIAL_DIMENSIONS = new Set([64, 128, 256, 512, 1024, 2048]);

function text(value: unknown): string {
  return String(value ?? "").trim();
}

function resolveUserPath(value: string): string {
  const selected = text(value);
  const expanded = selected === "~"
    ? os.homedir()
    : selected.startsWith("~/") ? path.join(os.homedir(), selected.slice(2)) : selected;
  return path.resolve(expanded);
}

function indexedHomeDirectory(): string {
  const configured = text(process.env.INDEXED_HOME);
  return configured ? resolveUserPath(configured) : "";
}

function existingFile(value: string): string {
  if (!value) return "";
  const resolved = resolveUserPath(value);
  try {
    return fs.statSync(resolved).isFile() ? resolved : "";
  } catch {
    return "";
  }
}

/** Optional override; the helper includes the validated D profile by default. */
export function managedAppleEmbeddingAssets(indexedHome = indexedHomeDirectory()): {
  recurrenceProfile: string;
} {
  if (!indexedHome) {
    return { recurrenceProfile: "" };
  }
  return {
    recurrenceProfile: existingFile(path.join(
      indexedHome, "runtime", "apple-embedding",
      "profiles",
      "q8_seq2112_mlp24_recurrence_slot0_fp16_block8_verified.json",
    )),
  };
}

function boundedInteger(value: unknown, fallback: number, minimum: number, maximum: number): number {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(maximum, Math.max(minimum, Math.floor(number)));
}

function isExecutable(file: string): boolean {
  try {
    fs.accessSync(file, fs.constants.X_OK);
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
}

/** Resolve both a packaged Electron helper and the source-tree development helper. */
export function resolveAppleEmbeddingBinary(configured = ""): string {
  const resourcesPath = (process as NodeJS.Process & { resourcesPath?: string }).resourcesPath;
  const indexedHome = indexedHomeDirectory();
  const candidates = [
    text(configured),
    text(process.env.INDEXED_APPLE_EMBEDDING_BINARY),
    indexedHome ? path.join(indexedHome, "runtime", "apple-embedding", "indexed-apple-embedding") : "",
    typeof resourcesPath === "string"
      ? path.join(resourcesPath, "native", "apple-embedding", "indexed-apple-embedding")
      : "",
    path.resolve(process.cwd(), "native/apple-embedding/dist/apple-silicon/indexed-apple-embedding"),
  ].filter(Boolean).map((candidate) => resolveUserPath(candidate));
  const binary = candidates.find(isExecutable);
  if (binary) return binary;
  throw new Error(
    "找不到可执行的 indexed-apple-embedding；请配置 embedding.native.binary 或 INDEXED_APPLE_EMBEDDING_BINARY",
  );
}

function assertDirectory(label: string, value: string, required = true): string {
  const resolved = value ? resolveUserPath(value) : "";
  if (!resolved && !required) return "";
  if (!resolved || !fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory()) {
    throw new Error(`${label} 目录不存在：${resolved || "(空)"}`);
  }
  return resolved;
}

function assertFile(label: string, value: string, required = true): string {
  const resolved = value ? resolveUserPath(value) : "";
  if (!resolved && !required) return "";
  if (!resolved || !fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    throw new Error(`${label} 文件不存在：${resolved || "(空)"}`);
  }
  return resolved;
}

function executionDefaults(executionMode: AppleExecutionMode): {
  mode: AppleEmbeddingMode;
  visionCompute: AppleVisionCompute;
} {
  if (executionMode === "a") return { mode: "fast", visionCompute: "gpu" };
  return { mode: "fast", visionCompute: "ane" };
}

function numberList(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.map(Number).filter((item) => Number.isInteger(item)))];
}

async function reserveLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close((error) => error ? reject(error) : resolve(port));
    });
  });
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function parseLastJSON(stdout: string): Record<string, unknown> {
  const line = stdout.trim().split("\n").map((item) => item.trim()).filter(Boolean).at(-1);
  if (!line) throw new Error("原生 helper 没有返回 JSON");
  const value = JSON.parse(line) as unknown;
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("原生 helper 返回的不是 JSON 对象");
  }
  return value as Record<string, unknown>;
}

async function runJSONCommand(
  binary: string,
  args: string[],
  timeoutMs: number,
  onStderr?: (line: string) => void,
): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, args, {
      cwd: path.dirname(binary),
      env: appleEmbeddingChildEnvironment(),
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => { stdout += chunk; });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
      for (const line of chunk.split(/\r?\n/).map((item) => item.trim()).filter(Boolean)) {
        onStderr?.(line);
      }
    });
    let settled = false;
    const finish = (error?: unknown, value?: Record<string, unknown>) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolve(value ?? {});
    };
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish(new Error(`原生 helper 命令超时（${timeoutMs}ms）`));
    }, timeoutMs);
    timer.unref();
    child.once("error", (error) => finish(error));
    child.once("exit", (code, signal) => {
      if (code !== 0) {
        finish(new Error(`原生 helper 命令失败：${code ?? signal}; ${stderr.trim().slice(-1000)}`));
        return;
      }
      try { finish(undefined, parseLastJSON(stdout)); }
      catch (error) { finish(error); }
    });
  });
}

function appleEmbeddingChildEnvironment(extra: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv {
  const indexedHome = indexedHomeDirectory();
  const tempRoot = text(process.env.INDEXED_TEMP_ROOT)
    || (indexedHome ? path.join(indexedHome, "tmp", "apple-embedding") : "");
  if (tempRoot) fs.mkdirSync(resolveUserPath(tempRoot), { recursive: true, mode: 0o700 });
  return {
    ...process.env,
    ...(tempRoot ? {
      TMPDIR: resolveUserPath(tempRoot),
      TMP: resolveUserPath(tempRoot),
      TEMP: resolveUserPath(tempRoot),
    } : {}),
    ...extra,
  };
}

/** Install a validated, already-converted model package into app-managed storage. */
export async function installAppleEmbeddingModel(options: {
  binary?: string;
  source: string;
  modelsDirectory: string;
  timeoutMs?: number;
  onStderr?: (line: string) => void;
}): Promise<Record<string, unknown>> {
  const binary = resolveAppleEmbeddingBinary(options.binary);
  const source = assertDirectory("待安装模型包", text(options.source));
  const configuredModelsDirectory = text(options.modelsDirectory);
  if (!configuredModelsDirectory) throw new Error("modelsDirectory 不能为空");
  const modelsDirectory = resolveUserPath(configuredModelsDirectory);
  const result = await runJSONCommand(
    binary,
    ["install-model", "--source", source, "--models-dir", modelsDirectory],
    boundedInteger(options.timeoutMs, DEFAULT_MAINTENANCE_TIMEOUT_MS, 1_000, 7_200_000),
    options.onStderr,
  );
  if (result.status !== "installed") throw new Error("模型安装没有返回 installed");
  return result;
}

export function isAppleEmbeddingProfile(profile: Record<string, any> | undefined): boolean {
  return text(profile?.embedding?.provider).toLowerCase() === APPLE_EMBEDDING_PROVIDER;
}

export function appleEmbeddingOptionsFromProfile(
  profile: Record<string, any>,
  overrides: Partial<AppleEmbeddingBackendOptions> = {},
): AppleEmbeddingBackendOptions {
  const embedding = profile?.embedding ?? {};
  const native = embedding.native ?? {};
  const executionMode = requireAppleExecutionMode(native);
  const { mode, visionCompute } = executionDefaults(executionMode);
  const indexedHome = indexedHomeDirectory();
  const managedAssets = managedAppleEmbeddingAssets(indexedHome);
  const configuredRecurrenceSlots = numberList(native.privateANE?.recurrenceLayerSlots);
  return {
    binary: text(native.binary),
    modelPackage: text(
      native.modelPackage
        || process.env.INDEXED_APPLE_EMBEDDING_MODEL
        || (indexedHome ? path.join(indexedHome, "models", "WeMM-Embedding-2B-Apple-Q8-G64") : ""),
    ),
    coreMLCache: text(native.coreMLCache || (indexedHome ? path.join(indexedHome, "cache", "coreml") : "")),
    mode,
    visionCompute,
    executionMode,
    privateANE: {
      videoDownProjection: native.privateANE?.videoDownProjection === "fp16" ? "fp16" : "q8",
      videoPipeline: native.privateANE?.videoPipeline === 1 ? 1 : 2,
      sequenceLength: boundedInteger(native.privateANE?.sequenceLength, 2112, 64, 2112),
      mlpFraction: Number(native.privateANE?.mlpFraction ?? 0.75),
      mlpVariant: boundedInteger(native.privateANE?.mlpVariant, 8, 1, 64),
      mlpMaxLayers: boundedInteger(native.privateANE?.mlpMaxLayers, 24, 1, 24),
      recurrenceProfile: text(native.privateANE?.recurrenceProfile || (executionMode === "d" ? managedAssets.recurrenceProfile : "")),
      recurrenceBlockSize: boundedInteger(native.privateANE?.recurrenceBlockSize, 8, 2, 8),
      recurrenceLayerSlots: configuredRecurrenceSlots.length ? configuredRecurrenceSlots : [0],
      recurrenceQueryScale: Number(native.privateANE?.recurrenceQueryScale ?? 4096),
      recurrenceMaxTokens: boundedInteger(native.privateANE?.recurrenceMaxTokens, 8192, 64, 8192),
      recurrenceIODtype: native.privateANE?.recurrenceIODtype === "fp32" ? "fp32" : "fp16",
      recurrenceVerifyReference: native.privateANE?.recurrenceVerifyReference === true,
    },
    dimension: boundedInteger(embedding.dimension, 2048, 1, 4096),
    maxQueuedRequests: boundedInteger(native.maxQueuedRequests, 16, 0, 1024),
    startupTimeoutMs: boundedInteger(native.startupTimeoutSeconds, 300, 1, 3600) * 1000,
    maintenanceTimeoutMs: boundedInteger(native.maintenanceTimeoutSeconds, 600, 1, 7200) * 1000,
    autoRestart: native.autoRestart !== false,
    maxRestarts: boundedInteger(native.maxRestarts, 3, 0, 100),
    expectedModel: text(embedding.model),
    expectedEmbeddingSpace: text(profile.spaceId),
    ...overrides,
  };
}

export class AppleEmbeddingBackend {
  readonly options: Required<Omit<AppleEmbeddingBackendOptions,
    "binary" | "coreMLCache" | "onStateChange" | "onStderr" | "privateANE">> & {
      binary: string;
      coreMLCache: string;
      privateANE: Required<ApplePrivateANEKernelOptions>;
      onStateChange?: AppleEmbeddingBackendOptions["onStateChange"];
      onStderr?: AppleEmbeddingBackendOptions["onStderr"];
    };

  #child: ChildProcessByStdio<null, Readable, Readable> | null = null;
  #startPromise: Promise<AppleEmbeddingRuntimeConfig> | null = null;
  #stopping = false;
  #port = 0;
  #state: AppleEmbeddingBackendStatus["state"] = "stopped";
  #token = "";
  #runtime: AppleEmbeddingRuntimeConfig | null = null;
  #ready: Record<string, unknown> | null = null;
  #startedAt: string | null = null;
  #restartCount = 0;
  #lastExit: AppleEmbeddingBackendStatus["lastExit"] = null;
  #lastError = "";
  #stderrTail: string[] = [];
  #stderrBuffer = "";

  constructor(options: AppleEmbeddingBackendOptions) {
    const executionMode = requireAppleExecutionMode(options);
    const { mode, visionCompute } = executionDefaults(executionMode);
    const dimension = boundedInteger(options.dimension, 2048, 1, 4096);
    if (!OFFICIAL_DIMENSIONS.has(dimension)) {
      throw new Error(`WeMM 输出维度不受支持：${dimension}`);
    }
    const binary = resolveAppleEmbeddingBinary(options.binary);
    const modelPackage = assertDirectory("WeMM 模型包", text(options.modelPackage));
    const coreMLCache = text(options.coreMLCache)
      ? resolveUserPath(text(options.coreMLCache))
      : "";
    const privateANE = options.privateANE ?? {};
    const sequenceLength = boundedInteger(privateANE.sequenceLength, 2112, 64, 2112);
    const recurrenceMaxTokens = boundedInteger(privateANE.recurrenceMaxTokens, 8192, 64, 8192);
    const recurrenceBlockSize = boundedInteger(privateANE.recurrenceBlockSize, 8, 2, 8);
    const recurrenceLayerSlots = numberList(privateANE.recurrenceLayerSlots);
    if (sequenceLength % 64 || recurrenceMaxTokens % 64) {
      throw new Error("private ANE sequenceLength 和 recurrenceMaxTokens 必须是 64 的倍数");
    }
    if (![2, 4, 8].includes(recurrenceBlockSize)) {
      throw new Error("private ANE recurrenceBlockSize 必须是 2、4 或 8");
    }
    if (recurrenceLayerSlots.some((slot) => slot < 0 || slot >= 18)) {
      throw new Error("private ANE recurrenceLayerSlots 必须位于 0..17");
    }
    const mlpFraction = Number(privateANE.mlpFraction ?? 0.75);
    const mlpVariant = boundedInteger(privateANE.mlpVariant, 8, 1, 64);
    const mlpMaxLayers = boundedInteger(privateANE.mlpMaxLayers, 24, 1, 24);
    const selectedSlots = recurrenceLayerSlots.length ? recurrenceLayerSlots : [0];
    if (["c", "d"].includes(executionMode)) {
      if (mlpVariant !== 8 || privateANE.videoDownProjection === "fp16") {
        throw new Error("Swift C/D 当前支持 MLP variant 8 与 Q8 down projection，请显式选择受支持的参数");
      }
      if (!Number.isFinite(mlpFraction) || mlpFraction <= 0 || mlpFraction >= 1) throw new Error("MLP 分流比例必须位于 0..1 之间");
      if (executionMode === "d" && (sequenceLength !== 2112 || mlpFraction !== 0.75 || mlpMaxLayers !== 24
        || recurrenceBlockSize !== 8 || selectedSlots.length !== 1 || selectedSlots[0] !== 0
        || Number(privateANE.recurrenceQueryScale ?? 4096) !== 4096 || recurrenceMaxTokens !== 8192
        || privateANE.recurrenceIODtype === "fp32" || privateANE.recurrenceVerifyReference === true)) {
        throw new Error("Swift D 当前支持经过验证的 seq2112/MLP24/75%/block8/slot0/FP16/scale4096/max8192 配置；逐次参考验证请使用离线验证工具");
      }
    }
    const recurrenceProfile = text(privateANE.recurrenceProfile)
      ? assertFile("private ANE recurrence profile", text(privateANE.recurrenceProfile))
      : "";
    this.options = {
      binary,
      modelPackage,
      coreMLCache,
      mode,
      visionCompute,
      executionMode,
      privateANE: {
        videoDownProjection: privateANE.videoDownProjection === "fp16" ? "fp16" : "q8",
        videoPipeline: privateANE.videoPipeline === 1 ? 1 : 2,
        sequenceLength,
        mlpFraction,
        mlpVariant,
        mlpMaxLayers,
        recurrenceProfile,
        recurrenceBlockSize,
        recurrenceLayerSlots: selectedSlots,
        recurrenceQueryScale: Number(privateANE.recurrenceQueryScale ?? 4096),
        recurrenceMaxTokens,
        recurrenceIODtype: privateANE.recurrenceIODtype === "fp32" ? "fp32" : "fp16",
        recurrenceVerifyReference: privateANE.recurrenceVerifyReference === true,
      },
      dimension,
      maxQueuedRequests: boundedInteger(options.maxQueuedRequests, 16, 0, 1024),
      startupTimeoutMs: boundedInteger(options.startupTimeoutMs, DEFAULT_STARTUP_TIMEOUT_MS, 1_000, 3_600_000),
      maintenanceTimeoutMs: boundedInteger(options.maintenanceTimeoutMs, DEFAULT_MAINTENANCE_TIMEOUT_MS, 1_000, 7_200_000),
      autoRestart: options.autoRestart !== false,
      maxRestarts: boundedInteger(options.maxRestarts, 3, 0, 100),
      expectedModel: text(options.expectedModel),
      expectedEmbeddingSpace: text(options.expectedEmbeddingSpace),
      extraArgs: [...(options.extraArgs ?? [])],
      onStateChange: options.onStateChange,
      onStderr: options.onStderr,
    };
  }

  get runtime(): AppleEmbeddingRuntimeConfig | null {
    return this.#runtime ? { ...this.#runtime } : null;
  }

  status(): AppleEmbeddingBackendStatus {
    return {
      managed: true,
      state: this.#state,
      mode: this.options.mode,
      visionCompute: this.options.visionCompute,
      executionMode: this.options.executionMode,
      pid: this.#child?.pid ?? null,
      baseUrl: this.#runtime?.baseUrl ?? (this.#port ? `http://127.0.0.1:${this.#port}` : ""),
      model: this.#runtime?.model ?? "",
      dimension: this.options.dimension,
      embeddingSpace: this.#runtime?.embeddingSpace ?? "",
      startedAt: this.#startedAt,
      restartCount: this.#restartCount,
      lastExit: this.#lastExit,
      lastError: this.#lastError,
      stderrTail: [...this.#stderrTail],
      ready: this.#ready ? { ...this.#ready } : null,
    };
  }

  async start(): Promise<AppleEmbeddingRuntimeConfig> {
    if (this.#state === "ready" && this.#runtime) return { ...this.#runtime };
    if (this.#startPromise) return this.#startPromise;
    this.#stopping = false;
    this.#startPromise = this.#startOnce(false);
    try {
      return await this.#startPromise;
    } finally {
      this.#startPromise = null;
    }
  }

  async restart(): Promise<AppleEmbeddingRuntimeConfig> {
    this.#stopping = true;
    await this.#stopChild();
    this.#stopping = false;
    this.#restartCount += 1;
    this.#state = "restarting";
    this.#emitState();
    return this.start();
  }

  async stop(): Promise<void> {
    const inFlight = this.#startPromise;
    this.#stopping = true;
    this.#state = "stopped";
    this.#runtime = null;
    this.#ready = null;
    this.#emitState();
    await this.#stopChild();
    // A stop during port reservation or model load must not leave a late child
    // behind or let the in-flight start overwrite the stopped state.
    if (inFlight) await inFlight.catch(() => undefined);
  }

  async health(): Promise<Record<string, unknown>> {
    return this.#json("/health");
  }

  async models(): Promise<Record<string, unknown>> {
    return this.#json("/v1/models");
  }

  async validate(full = false): Promise<Record<string, unknown>> {
    const args = ["validate-model", "--package", this.options.modelPackage];
    if (full) args.push("--full");
    const result = await this.#maintenance(args);
    if (result.status !== "valid") throw new Error("模型校验没有返回 valid");
    return result;
  }

  async prepare(): Promise<Record<string, unknown>> {
    if (this.#child) throw new Error("prepare-coreml 前必须停止 Apple embedding 后端");
    const args = ["prepare-coreml", "--package", this.options.modelPackage];
    if (this.options.coreMLCache) args.push("--coreml-cache", this.options.coreMLCache);
    const result = await this.#maintenance(args);
    if (result.status !== "prepared") throw new Error("Core ML 准备没有返回 prepared");
    return result;
  }

  async #startOnce(supervised: boolean): Promise<AppleEmbeddingRuntimeConfig> {
    if (!this.#port) this.#port = await reserveLoopbackPort();
    if (this.#stopping) throw new Error("Apple embedding 后端启动已取消");
    this.#state = supervised ? "restarting" : "starting";
    this.#lastError = "";
    this.#token = randomBytes(32).toString("hex");
    this.#emitState();
    const args = this.#nativeServeArgs();
    args.push(...this.options.extraArgs);

    const child = spawn(this.options.binary, args, {
      cwd: path.dirname(this.options.binary),
      env: appleEmbeddingChildEnvironment({ INDEXED_APPLE_EMBEDDING_AUTH_TOKEN: this.#token }),
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    this.#child = child;
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => this.#captureStderr(chunk));

    let settled = false;
    const readyPromise = new Promise<Record<string, unknown>>((resolve, reject) => {
      let buffer = "";
      child.stdout.setEncoding("utf8");
      child.stdout.on("data", (chunk: string) => {
        buffer += chunk;
        for (;;) {
          const newline = buffer.indexOf("\n");
          if (newline < 0) break;
          const line = buffer.slice(0, newline).trim();
          buffer = buffer.slice(newline + 1);
          if (!line) continue;
          try {
            const value = JSON.parse(line) as Record<string, unknown>;
            if (value.status === "ready" && text(value.url)) {
              settled = true;
              resolve(value);
              return;
            }
          } catch (error) {
            settled = true;
            reject(new Error(`原生 helper ready JSON 无效：${errorMessage(error)}`));
            return;
          }
        }
      });
      child.once("error", (error) => {
        if (!settled) reject(error);
      });
      child.once("exit", (code, signal) => {
        this.#lastExit = { code, signal };
        if (this.#child === child) this.#child = null;
        if (!settled) reject(new Error(`原生 helper 就绪前退出：${code ?? signal ?? "unknown"}`));
        else if (!this.#stopping) void this.#scheduleSupervisedRestart();
      });
    });
    let timer: NodeJS.Timeout | undefined;
    try {
      const timeout = new Promise<never>((_resolve, reject) => {
        timer = setTimeout(
          () => reject(new Error(`原生 helper 启动超时（${this.options.startupTimeoutMs}ms）`)),
          this.options.startupTimeoutMs,
        );
        timer.unref();
      });
      const ready = await Promise.race([readyPromise, timeout]);
      if (this.#stopping) throw new Error("Apple embedding 后端启动已取消");
      const baseUrl = text(ready.url).replace(/\/+$/, "");
      const expectedBaseUrl = `http://127.0.0.1:${this.#port}`;
      if (baseUrl !== expectedBaseUrl) {
        throw new Error(`原生 helper 返回了非预期监听地址：${baseUrl || "(空)"}`);
      }
      const model = text(ready.default_model);
      const embeddingSpace = text(ready.default_embedding_space);
      if (!baseUrl || !model || !embeddingSpace) throw new Error("原生 helper ready 信息不完整");
      if ((ready.execution_mode !== undefined || ["c", "d"].includes(this.options.executionMode))
        && text(ready.execution_mode) !== this.options.executionMode) {
        throw new Error(`原生 helper 执行模式不匹配：期望 ${this.options.executionMode}，实际 ${text(ready.execution_mode) || "未声明"}`);
      }
      if (this.options.expectedModel && model !== this.options.expectedModel) {
        throw new Error(`原生 helper 模型不匹配：期望 ${this.options.expectedModel}，实际 ${model}`);
      }
      if (this.options.expectedEmbeddingSpace && embeddingSpace !== this.options.expectedEmbeddingSpace) {
        throw new Error(
          `原生 helper 向量空间不匹配：期望 ${this.options.expectedEmbeddingSpace}，实际 ${embeddingSpace}`,
        );
      }
      const runtime = {
        baseUrl,
        apiKey: this.#token,
        model,
        dimension: this.options.dimension,
        embeddingSpace,
        inputStyle: "wemm" as const,
      };
      this.#runtime = runtime;
      this.#ready = ready;
      await this.health();
      const models = await this.models();
      const advertised = Array.isArray(models.data)
        ? models.data.find((item) => item && typeof item === "object" && text(item.id) === model) as Record<string, unknown> | undefined
        : undefined;
      if (!advertised) throw new Error(`原生 helper /v1/models 未广告默认模型 ${model}`);
      if (Number(advertised.dimension) !== this.options.dimension) {
        throw new Error(
          `原生 helper 维度不匹配：期望 ${this.options.dimension}，实际 ${text(advertised.dimension) || "(空)"}`,
        );
      }
      if (text(advertised.embedding_space) !== embeddingSpace) {
        throw new Error("原生 helper ready 与 /v1/models 的向量空间不一致");
      }
      this.#startedAt = new Date().toISOString();
      this.#state = "ready";
      this.#emitState();
      return { ...runtime };
    } catch (error) {
      const cancelled = this.#stopping;
      this.#lastError = cancelled ? "" : errorMessage(error);
      this.#state = cancelled ? "stopped" : "failed";
      this.#runtime = null;
      this.#emitState();
      const priorStopping = this.#stopping;
      this.#stopping = true;
      await this.#stopChild();
      this.#stopping = priorStopping;
      throw error;
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  #nativeServeArgs(): string[] {
    const args = [
      "serve",
      "--package", this.options.modelPackage,
      "--port", String(this.#port),
      "--default-dimension", String(this.options.dimension),
      "--execution-mode", this.options.executionMode,
      "--vision-compute", this.options.visionCompute,
      "--language-compute", "gpu",
      "--max-queued-requests", String(this.options.maxQueuedRequests),
    ];
    if (this.options.coreMLCache) args.push("--coreml-cache", this.options.coreMLCache);
    if (["c", "d"].includes(this.options.executionMode)) args.push(...this.#privateServeArgs());
    return args;
  }

  #privateServeArgs(): string[] {
    const kernel = this.options.privateANE;
    const args = [
      "--private-ane-sequence-length", String(kernel.sequenceLength),
      "--video-down-projection", kernel.videoDownProjection,
      "--video-pipeline", String(kernel.videoPipeline),
      "--private-ane-mlp-fraction", String(kernel.mlpFraction),
      "--private-ane-mlp-variant", String(kernel.mlpVariant),
      "--private-ane-mlp-max-layers", String(kernel.mlpMaxLayers),
      "--private-ane-recurrence-block-size", String(kernel.recurrenceBlockSize),
      "--private-ane-recurrence-query-scale", String(kernel.recurrenceQueryScale),
      "--private-ane-recurrence-max-tokens", String(kernel.recurrenceMaxTokens),
      "--private-ane-recurrence-io-dtype", kernel.recurrenceIODtype,
    ];
    if (kernel.recurrenceProfile) args.push("--private-ane-recurrence-profile", kernel.recurrenceProfile);
    if (kernel.recurrenceLayerSlots.length) {
      args.push("--private-ane-recurrence-layer-slots", kernel.recurrenceLayerSlots.join(","));
    }
    if (kernel.recurrenceVerifyReference) args.push("--private-ane-recurrence-verify-reference");
    return args;
  }

  async #scheduleSupervisedRestart(): Promise<void> {
    this.#runtime = null;
    this.#ready = null;
    if (!this.options.autoRestart || this.#restartCount >= this.options.maxRestarts) {
      this.#state = "failed";
      this.#lastError = this.options.autoRestart
        ? `原生 helper 意外退出，已达到自动重启上限 ${this.options.maxRestarts}`
        : "原生 helper 意外退出";
      this.#emitState();
      return;
    }
    this.#restartCount += 1;
    this.#state = "restarting";
    this.#emitState();
    await delay(Math.min(5_000, 250 * (2 ** (this.#restartCount - 1))));
    if (this.#stopping) return;
    try {
      await this.#startOnce(true);
    } catch (error) {
      this.#lastError = errorMessage(error);
      if (!this.#stopping) void this.#scheduleSupervisedRestart();
    }
  }

  async #stopChild(timeoutMs = 10_000): Promise<void> {
    const child = this.#child;
    this.#child = null;
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    await new Promise<void>((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        resolve();
      };
      child.once("exit", finish);
      child.kill("SIGTERM");
      const timer = setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
        finish();
      }, timeoutMs);
      timer.unref();
    });
  }

  async #maintenance(args: string[]): Promise<Record<string, unknown>> {
    return runJSONCommand(
      this.options.binary,
      args,
      this.options.maintenanceTimeoutMs,
      (line) => this.#captureStderr(`${line}\n`),
    );
  }

  async #json(endpoint: string, init: RequestInit = {}): Promise<Record<string, unknown>> {
    const baseUrl = this.#runtime?.baseUrl;
    if (!baseUrl) throw new Error("Apple embedding 后端尚未就绪");
    const response = await fetch(`${baseUrl}${endpoint}`, {
      ...init,
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${this.#token}`,
        ...init.headers,
      },
    });
    const body = await response.json().catch(() => ({})) as Record<string, unknown>;
    if (!response.ok) {
      const detail = body.error && typeof body.error === "object"
        ? text((body.error as Record<string, unknown>).message)
        : "";
      const error = new Error(detail || `Apple embedding helper HTTP ${response.status}`) as Error & { status?: number };
      error.status = response.status;
      throw error;
    }
    return body;
  }

  #captureStderr(chunk: string): void {
    this.#stderrBuffer += chunk;
    const lines = this.#stderrBuffer.split(/\r?\n/);
    this.#stderrBuffer = lines.pop() ?? "";
    for (const raw of lines) {
      const line = raw.trim();
      if (!line) continue;
      this.#stderrTail.push(line.slice(0, 2_000));
      if (this.#stderrTail.length > STDERR_TAIL_LINES) this.#stderrTail.shift();
      this.options.onStderr?.(line);
    }
  }

  #emitState(): void {
    this.options.onStateChange?.(this.status());
  }
}
