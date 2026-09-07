import { createHash, randomBytes } from "node:crypto";
import { spawn, execFileSync, type ChildProcess } from "node:child_process";
import { once } from "node:events";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { parseArgs } from "node:util";

type JSONValue = string | number | boolean | null | JSONValue[] | { [key: string]: JSONValue };
type JSONObject = { [key: string]: JSONValue };
interface Case { id: string; text?: string; repeatText?: number; image?: string; videoFrames?: number; framesPerSecond?: number; images?: string[] }
interface Spec { version: number; name: string; dimension: number; repeats: number; warmupPerCase: number; kernel: Record<string, JSONValue>; gates: Record<string, JSONValue>; cases: Case[] }
export interface Sample { vector: number[]; promptTokens: number; embeddingSpace: string; wallMS: number; timings: JSONObject }
export interface CaseResult { id: string; inputSHA256: string; samples: Sample[]; before: JSONObject; after: JSONObject }
export interface Reference {
  version: 1; runtime: string; mode: string; suiteSHA256: string; modelIdentity: string;
  packageFingerprint: string; specification: Spec; sourceHashes: Record<string, string>;
  createdAt: string; environment: JSONObject; startupMS: number; peakRSSBytes: number;
  runtimeInfo: JSONObject; cases: CaseResult[]; concurrentVideo: { wallMS: number; samples: Sample[]; before: JSONObject; after: JSONObject };
}

const here = path.dirname(fileURLToPath(import.meta.url));
const native = path.resolve(here, "..");
const hash = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
const readObject = (filename: string): JSONObject => JSON.parse(fs.readFileSync(filename, "utf8")) as JSONObject;
function object(value: JSONValue | undefined): JSONObject { return value && typeof value === "object" && !Array.isArray(value) ? value : {}; }

export function vectorAgreement(expected: number[], actual: number[]) {
  if (!expected.length || expected.length !== actual.length || [...expected, ...actual].some(value => !Number.isFinite(value))) {
    throw new Error("Embedding vectors must have the same nonzero dimension and finite values");
  }
  let dot = 0, left = 0, right = 0, squared = 0, absolute = 0;
  for (let index = 0; index < expected.length; index++) {
    const a = expected[index]!, b = actual[index]!;
    dot += a * b; left += a * a; right += b * b;
    squared += (a - b) ** 2; absolute = Math.max(absolute, Math.abs(a - b));
  }
  if (!left || !right) throw new Error("Zero embeddings are invalid");
  return { cosine: dot / Math.sqrt(left * right), relativeL2: Math.sqrt(squared / left), maximumAbsoluteError: absolute };
}

function messages(fixture: Case): JSONObject[] {
  const imageURI = (name: string) => {
    if (path.basename(name) !== name) throw new Error("Fixture assets must be local names");
    return `data:image/png;base64,${fs.readFileSync(path.join(native, "mixed-query-set/assets", name)).toString("base64")}`;
  };
  const content: JSONValue[] = [];
  if (fixture.image) content.push({ type: "image_url", image_url: { url: imageURI(fixture.image) } });
  if (fixture.videoFrames) {
    const images = fixture.images || [];
    if (!images.length || !fixture.framesPerSecond) throw new Error("Video fixture needs images and timestamps");
    content.push({ type: "video_frames", frames: Array.from({ length: fixture.videoFrames }, (_, index) => ({
      image_url: { url: imageURI(images[index % images.length]!) }, timestamp: index / fixture.framesPerSecond!,
    })) });
  }
  if (fixture.text) content.push({ type: "text", text: fixture.text.repeat(fixture.repeatText || 1) });
  return [{ role: "user", content }];
}

async function hashFile(filename: string): Promise<string> {
  const digest = createHash("sha256");
  for await (const chunk of fs.createReadStream(filename)) digest.update(chunk as Buffer);
  return digest.digest("hex");
}

async function stop(child: ChildProcess) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const ended = once(child, "exit");
  const timer = setTimeout(() => child.kill("SIGKILL"), 15_000);
  child.kill("SIGTERM");
  await ended;
  clearTimeout(timer);
}

function sanitize(value: JSONObject, roots: string[]): JSONObject {
  let serialized = JSON.stringify(value);
  for (const root of roots.filter(Boolean).sort((a, b) => b.length - a.length)) serialized = serialized.replaceAll(root, "<local-asset>");
  return JSON.parse(serialized) as JSONObject;
}

export async function capture(args: { runtime: "python" | "swift"; mode: string; packagePath: string; binary: string; candidateBinary?: string; recurrenceProfile?: string; python?: string; researchRoot?: string; output: string }): Promise<void> {
  if (fs.existsSync(args.output)) throw new Error("Refusing to overwrite a frozen reference; choose a new output file");
  if (!/^[abcd]$/.test(args.mode)) throw new Error("Mode must be A/B/C/D");
  const privateCandidate = args.runtime === "swift" && ["c", "d"].includes(args.mode) && !!args.candidateBinary;
  if (args.candidateBinary && !privateCandidate) throw new Error("--candidate-binary is only for isolated Swift C/D maintainer candidates");
  if (privateCandidate && args.mode === "d" && !args.recurrenceProfile) throw new Error("Swift D candidate requires an explicit --recurrence-profile");
  if (args.runtime === "python" && !args.python) throw new Error("Reference Python executable is required");
  if (args.runtime === "python" && ["c", "d"].includes(args.mode) && !args.researchRoot) throw new Error("Python C/D require the existing isolated research environment");
  const specBytes = fs.readFileSync(path.join(here, "cd-reference-spec.json"));
  const specification = JSON.parse(specBytes.toString()) as Spec;
  const model = readObject(path.join(args.packagePath, "manifest.json"));
  // Actual model validation precedes every reference; a declared fingerprint alone is not evidence.
  execFileSync(args.binary, ["validate-model", "--package", args.packagePath, "--full"], { timeout: 180_000, stdio: "pipe" });
  const sourceHashes: Record<string, string> = {};
  for (const file of ["service.py", "ane_lane.py", "video_projection.py", ...fs.readdirSync(path.join(native, "private_ane")).filter(value => value.endsWith(".py")).map(value => `private_ane/${value}`)]) {
    sourceHashes[file] = await hashFile(path.join(native, file));
  }
  const identity: Record<string, string> = { manifest: hash(fs.readFileSync(path.join(args.packagePath, "manifest.json"))) };
  for (const file of ["tokenizer.json", "tokenizer_config.json", "embedding_chat_template.jinja", "processor_config.json", "language/config.json"]) {
    identity[file] = await hashFile(path.join(args.packagePath, file));
  }
  const kernel = specification.kernel;
  const profile = args.recurrenceProfile || (args.researchRoot ? path.join(args.researchRoot, "profiles/q8_seq2112_mlp24_recurrence_slot0_fp16_block8_verified.json") : "");
  if (args.recurrenceProfile) sourceHashes.recurrenceProfile = await hashFile(args.recurrenceProfile);
  const experiment = args.researchRoot ? path.join(args.researchRoot, "private-ane-experiment") : "";
  if (args.researchRoot) {
    for (const file of ["profiles/q8_seq2112_mlp24_recurrence_slot0_fp16_block8_verified.json", "private-ane-experiment/results/real_g_safe_c64_specialized.mil", "private-ane-experiment/.cache/ane-private-runtime/bridge/libane_bridge.dylib"]) {
      sourceHashes[`research/${file}`] = await hashFile(path.join(args.researchRoot, file));
    }
  }
  let dependencies: JSONObject = {};
  if (args.runtime === "python") {
    const info = execFileSync(args.python!, ["-c", `import importlib.metadata as m, importlib.util, json, hashlib
names=['service','omlx.patches.qwen35_ane_prefill','omlx.custom_kernels.qwen35_prefill.fast','omlx.custom_kernels.qwen35_prefill._ext','mlx_vlm.models.qwen3_5.gated_delta','mlx_vlm.models.qwen3_5.qwen3_5']
out={'versions':{n:m.version(n) for n in ['mlx','mlx-metal','mlx-vlm','mlx-lm','omlx','numpy','pillow','coremltools']},'modules':{}}
for n in names:
 s=importlib.util.find_spec(n)
 if s and s.origin:
  out['modules'][n]=hashlib.sha256(open(s.origin,'rb').read()).hexdigest()
print(json.dumps(out))`], { cwd: native, encoding: "utf8", timeout: 60_000 });
    dependencies = JSON.parse(info.trim().split("\n").at(-1)!) as JSONObject;
  } else {
    sourceHashes.swiftBinary = await hashFile(args.binary);
    sourceHashes.swiftResolved = await hashFile(path.join(native, "swift/Package.resolved"));
    if (args.candidateBinary) sourceHashes.swiftCandidateBinary = await hashFile(args.candidateBinary);
  }
  const token = randomBytes(32).toString("hex");
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-native-reference-"));
  const common = ["--package", args.packagePath, "--port", "0", "--default-dimension", String(specification.dimension)];
  const pythonArgs = [path.join(native, "service.py"), "--backend", "mlx", ...common, "--execution-mode", args.mode];
  if (["c", "d"].includes(args.mode)) {
    pythonArgs.push("--private-ane-sequence-length", String(kernel.sequenceLength), "--private-ane-mlp-fraction", String(kernel.mlpFraction),
      "--private-ane-mlp-variant", String(kernel.mlpVariant), "--private-ane-mlp-max-layers", String(kernel.mlpMaxLayers),
      "--video-down-projection", String(kernel.videoDownProjection), "--video-pipeline", String(kernel.videoPipeline));
  }
  if (args.mode === "d") pythonArgs.push("--private-ane-experiment-root", experiment, "--private-ane-recurrence-profile", profile,
    "--private-ane-recurrence-block-size", String(kernel.recurrenceBlockSize), "--private-ane-recurrence-layer-slots", "0",
    "--private-ane-recurrence-query-scale", String(kernel.recurrenceQueryScale), "--private-ane-recurrence-max-tokens", String(kernel.recurrenceMaxTokens),
    "--private-ane-recurrence-io-dtype", String(kernel.recurrenceIODtype));
  const command = args.runtime === "python" ? args.python! : args.candidateBinary || args.binary;
  const argv = args.runtime === "python" ? pythonArgs : privateCandidate
    ? ["--reference-server", "--reference-mode", args.mode, ...common, "--coreml-cache", path.join(scratch, "coreml"), ...(args.mode === "d" ? ["--recurrence-profile", profile] : [])]
    : ["serve", ...common, "--execution-mode", args.mode, "--language-compute", "gpu", "--coreml-cache", path.join(scratch, "coreml"),
      ...(["c", "d"].includes(args.mode) ? ["--private-ane-sequence-length", String(kernel.sequenceLength),
        "--private-ane-mlp-fraction", String(kernel.mlpFraction), "--private-ane-mlp-variant", String(kernel.mlpVariant),
        "--private-ane-mlp-max-layers", String(kernel.mlpMaxLayers), "--video-down-projection", String(kernel.videoDownProjection),
        "--video-pipeline", String(kernel.videoPipeline)] : []),
      ...(args.mode === "d" ? ["--private-ane-recurrence-block-size", String(kernel.recurrenceBlockSize),
        "--private-ane-recurrence-layer-slots", "0", "--private-ane-recurrence-query-scale", String(kernel.recurrenceQueryScale),
        "--private-ane-recurrence-max-tokens", String(kernel.recurrenceMaxTokens), "--private-ane-recurrence-io-dtype", String(kernel.recurrenceIODtype),
        ...(args.recurrenceProfile ? ["--private-ane-recurrence-profile", profile] : [])] : [])];
  const started = performance.now();
  const child = spawn(command, argv, { cwd: native, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, INDEXED_APPLE_EMBEDDING_AUTH_TOKEN: token, PYTHONDONTWRITEBYTECODE: "1" } });
  let stderr = "", peakRSSBytes = 0;
  child.stderr?.on("data", value => { stderr = (stderr + String(value)).slice(-16_000); });
  const sampler = setInterval(() => {
    if (!child.pid || child.exitCode !== null || child.signalCode !== null) return;
    try { peakRSSBytes = Math.max(peakRSSBytes, Number(execFileSync("ps", ["-o", "rss=", "-p", String(child.pid)], { encoding: "utf8", timeout: 1000 }).trim()) * 1024); } catch { /* An exit can race sampling. */ }
  }, 200);
  const roots = [args.packagePath, args.researchRoot || "", native, scratch];
  try {
    const ready = await new Promise<JSONObject>((resolve, reject) => {
      const lines = createInterface({ input: child.stdout! });
      const timer = setTimeout(() => { lines.close(); reject(new Error(`Reference startup timed out: ${stderr}`)); }, 300_000);
      lines.on("line", line => {
        try { const value = JSON.parse(line) as JSONObject; if (value.status === "ready") { clearTimeout(timer); lines.close(); resolve(value); } } catch { /* Third-party runtime startup diagnostics. */ }
      });
      child.once("error", error => { clearTimeout(timer); lines.close(); reject(error); });
      child.once("exit", code => { clearTimeout(timer); lines.close(); reject(new Error(`Reference process exited (${code}): ${stderr}`)); });
    });
    const startupMS = performance.now() - started;
    if (privateCandidate && ready.candidate !== `swift-${args.mode}-pipeline-candidate`) throw new Error("Candidate binary did not identify its actual experimental implementation");
    const url = String(ready.url || "").replace(/\/$/, "");
    if (!/^http:\/\/127\.0\.0\.1:\d+$/.test(url)) throw new Error("Reference helper must bind to an ephemeral loopback port");
    async function request(endpoint: string, body?: JSONObject): Promise<JSONObject> {
      const response = await fetch(`${url}${endpoint}`, { method: body ? "POST" : "GET", headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        ...(body ? { body: JSON.stringify(body) } : {}), signal: AbortSignal.timeout(180_000) });
      const result = await response.json() as JSONObject;
      if (!response.ok) throw new Error(`Reference HTTP ${response.status}: ${JSON.stringify(result)}`);
      return result;
    }
    const health = async () => sanitize(await request("/health"), roots);
    async function embed(input: JSONObject[]): Promise<Sample> {
      const before = performance.now();
      const result = await request("/v1/embeddings", { model: `wemm-embedding-2b-apple-${specification.dimension}`, messages: input });
      const row = Array.isArray(result.data) ? object(result.data[0]) : {};
      const vector = row.embedding as number[];
      vectorAgreement(vector, vector);
      if (vector.length !== specification.dimension) throw new Error("Unexpected embedding dimension");
      const indexed = object(result.indexed);
      return { vector, promptTokens: Number(object(result.usage).prompt_tokens), embeddingSpace: String(indexed.embedding_space), wallMS: performance.now() - before, timings: object(indexed.timings_ms) };
    }
    console.log(`${args.runtime} ${args.mode.toUpperCase()} ready (${Math.round(startupMS)} ms)`);
    const cases: CaseResult[] = [];
    for (const fixture of specification.cases) {
      const input = messages(fixture);
      for (let index = 0; index < specification.warmupPerCase; index++) await embed(input);
      const before = await health();
      const samples: Sample[] = [];
      for (let index = 0; index < specification.repeats; index++) samples.push(await embed(input));
      cases.push({ id: fixture.id, inputSHA256: hash(JSON.stringify(input)), samples, before, after: await health() });
      console.log(`${fixture.id}: ${samples[0]!.promptTokens} tokens, ${Math.round(samples.reduce((sum, sample) => sum + sample.wallMS, 0) / samples.length)} ms mean`);
    }
    const video = specification.cases.find(value => value.id === "video-10s")!;
    const before = await health(), batchStarted = performance.now();
    const samples = await Promise.all([embed(messages(video)), embed(messages(video))]);
    const concurrentVideo = { wallMS: performance.now() - batchStarted, samples, before, after: await health() };
    const reference: Reference = { version: 1, runtime: args.runtime, mode: args.mode, suiteSHA256: hash(specBytes), modelIdentity: hash(JSON.stringify(identity)),
      packageFingerprint: String(model.package_fingerprint), specification, sourceHashes, createdAt: new Date().toISOString(),
      environment: { platform: process.platform, arch: process.arch, release: os.release(), cpu: os.cpus()[0]?.model || "unknown", totalMemoryBytes: os.totalmem(), dependencies },
      startupMS, peakRSSBytes, runtimeInfo: sanitize(ready, roots), cases, concurrentVideo };
    fs.mkdirSync(path.dirname(args.output), { recursive: true });
    fs.writeFileSync(args.output, `${JSON.stringify(reference)}\n`, { flag: "wx" });
    console.log(`Saved ${args.runtime} ${args.mode.toUpperCase()} reference (${cases.length} cases)`);
  } finally {
    clearInterval(sampler);
    await stop(child);
    fs.rmSync(scratch, { recursive: true, force: true });
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: Object.fromEntries(["runtime", "mode", "package", "binary", "candidate-binary", "recurrence-profile", "python", "research-root", "output"].map(key => [key, { type: "string" as const }])) });
  for (const name of ["runtime", "mode", "package", "binary", "output"]) if (!values[name]) throw new Error(`Missing --${name}`);
  if (!["python", "swift"].includes(values.runtime!)) throw new Error("--runtime must be python or swift");
  await capture({ runtime: values.runtime as "python" | "swift", mode: values.mode!.toLowerCase(), packagePath: path.resolve(values.package!), binary: path.resolve(values.binary!), output: path.resolve(values.output!),
    ...(values["candidate-binary"] ? { candidateBinary: path.resolve(values["candidate-binary"]) } : {}),
    ...(values["recurrence-profile"] ? { recurrenceProfile: path.resolve(values["recurrence-profile"]) } : {}),
    ...(values.python ? { python: path.resolve(values.python) } : {}), ...(values["research-root"] ? { researchRoot: path.resolve(values["research-root"]) } : {}) });
}
