#!/usr/bin/env node
// @ts-nocheck -- argument parsing is preserved while typed commands are introduced.
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import {
  AppleEmbeddingBackend,
  appleEmbeddingOptionsFromProfile,
  installAppleEmbeddingModel,
  isAppleEmbeddingProfile,
} from "@indexed/apple-embedding-backend";
import {
  activeProfile,
  clearRuntimeEmbeddingOverride,
  configPath,
  indexedHomePath,
  loadConfig,
  publicConfig,
  setPath,
  setRuntimeEmbeddingOverride,
  writeConfig,
} from "@indexed/config";
import { addLibrary, assetModelInfo, deleteVideo, librariesStatus, libraryCoverage, listAssets, listLocalFiles, listVideos, pruneMissingAssets, readAssetDocument, removeLibrary, scanLibrary, scanLocalLibrary, search, searchAllAssets, searchLocalFiles, status, updateVideo, writeAssetDocument } from "@indexed/core";
import { appleEmbeddingModel, requireAppleExecutionMode } from "@indexed/contracts";
import { startServer } from "@indexed/server";
import { prepareVideoDeletion } from "@indexed/core";

const args = process.argv.slice(2);
const command = args.shift() || "help";

function value(flag, fallback = "") {
  const index = args.indexOf(flag);
  return index >= 0 && args[index + 1] !== undefined ? args[index + 1] : fallback;
}

function values(flag) {
  return args.flatMap((item, index) => item === flag && args[index + 1] !== undefined ? [String(args[index + 1])] : []);
}

const BOOLEAN_FLAGS = new Set(["--activate", "--confirm", "--no-auto-restart", "--no-legacy", "--dry-run", "--full", "--include-missing", "--no-scan", "--open", "--private-ane-verify-reference", "--purge", "--raw", "--yes"]);

function positional() {
  const values = [];
  for (let index = 0; index < args.length; index += 1) {
    const token = String(args[index]);
    if (token.startsWith("--")) {
      // Value-taking flags carry their argument; switches carry nothing, so the next
      // token is still a positional path or query.
      if (!BOOLEAN_FLAGS.has(token)) index += 1;
      continue;
    }
    values.push(token);
  }
  return values;
}

function output(data) {
  process.stdout.write(`${JSON.stringify(data, null, 2)}\n`);
}

/** Scanning one root takes minutes; silence looks like a hang. */
function progressReporter(label) {
  let reported = 0;
  return (job) => {
    const line = `${label} 索引 ${job.done}/${job.total} 跳过 ${job.skipped} 失败 ${job.failed}${job.current ? ` ${job.current}` : ""}`;
    if (process.stderr.isTTY) {
      process.stderr.write(`\r${line.padEnd(110).slice(0, 110)}`);
      return;
    }
    if (job.done - reported >= 10 || job.done >= job.total) {
      reported = job.done;
      process.stderr.write(`${line}\n`);
    }
  };
}

function usage() {
  process.stdout.write(`Indexed — local control plane for human and agent use

Usage:
  indexed serve [--host 127.0.0.1] [--port 18767] [--open]
  indexed open
  indexed status
  indexed search "query" [--limit 30] [--video VIDEO_ID]
  indexed videos list [--limit 100]
  indexed videos update SITE VIDEO_ID --title "..." [--note "..."] [--tags "a,b"]
  indexed videos delete SITE VIDEO_ID --yes
  indexed files list [--limit 200]
  indexed files search "query" [--limit 30]
  indexed files scan [--limit 500]
  indexed assets libraries
  indexed assets list [--limit 200] [--kind video|image|document] [--library PATH] [--include-missing]
  indexed assets search "query" [--limit 30] [--kind video|image|document] [--include-missing] [--no-legacy]
  indexed assets add DIRECTORY [--no-scan] [--limit 400]
  indexed assets remove DIRECTORY [--purge --confirm]
  indexed assets scan [DIRECTORY] [--limit 400]
  indexed assets estimate
  indexed assets prune [--library PATH] [--dry-run]
  indexed assets read FILE [--raw]
  indexed assets write FILE --from PATH|[-]
  indexed assets model-info
  indexed embedding status
  indexed embedding configure --model-package PATH --execution-mode A|B|C|D [--binary PATH]
    C/D: Swift native helper [--private-ane-sequence-length 2112]
    D: [--private-ane-recurrence-profile PATH] (defaults to the built-in validated profile)
    Kernel: [--private-ane-block-size 2|4|8] [--private-ane-layer-slots 0,1,...,17] [--private-ane-query-scale 4096] [--private-ane-max-tokens 8192] [--private-ane-io-dtype fp16|fp32] [--private-ane-verify-reference]
  indexed embedding install --source PACKAGE --models-dir DIRECTORY [--binary PATH]
  indexed embedding validate [--full]
  indexed embedding prepare
  indexed config path|show
  indexed config use PROFILE
  indexed config set dotted.path value
  indexed config profiles list
  indexed config profiles create PROFILE [--from EXISTING_PROFILE]
  indexed config profiles delete PROFILE --yes

All data commands print JSON and work without starting the dashboard.
`);
}

function openUrl(url) {
  const platform = process.platform;
  const executable = platform === "darwin" ? "open" : platform === "win32" ? "cmd" : "xdg-open";
  const openArgs = platform === "win32" ? ["/c", "start", "", url] : [url];
  spawn(executable, openArgs, { detached: true, stdio: "ignore" }).unref();
}

async function main() {
  if (command === "serve") {
    const started = await startServer({ host: value("--host"), port: value("--port") });
    let closing = false;
    const close = async () => {
      if (closing) return;
      closing = true;
      try { await started.close(); }
      catch (error) { console.error(error); process.exitCode = 1; }
    };
    process.once("SIGINT", () => void close());
    process.once("SIGTERM", () => void close());
    process.stdout.write(`Indexed dashboard: ${started.url}\nConfig: ${configPath()}\nProfile: ${started.profile}\n`);
    if (args.includes("--open")) openUrl(started.url);
    return;
  }
  if (command === "open") {
    const config = loadConfig();
    const url = `http://${config.server.host}:${config.server.port}/`;
    openUrl(url);
    output({ ok: true, url });
    return;
  }
  if (command === "embedding") {
    const action = args.shift() || "status";
    const config = loadConfig();
    const profile = activeProfile(config);
    if (action === "configure") {
      const indexedHome = indexedHomePath();
      const modelPackage = value(
        "--model-package",
        profile.embedding.native?.modelPackage
          || (indexedHome ? path.join(indexedHome, "models", "WeMM-Embedding-2B-Apple-Q8-G64") : ""),
      );
      if (!modelPackage) throw new Error("configure 需要 --model-package，或先设置 INDEXED_HOME");
      const dimension = Number(value("--dimension", 2048));
      const executionMode = requireAppleExecutionMode({ executionMode: value("--execution-mode", profile.embedding.native?.executionMode || "b") });
      if (args.some((arg) => /^--decoder-/.test(arg))) throw new Error("E 模式的 decoder 参数已移除，请使用 A/B/C/D");
      if (![64, 128, 256, 512, 1024, 2048].includes(dimension)) {
        throw new Error("Apple WeMM 2B dimension 必须是 64、128、256、512、1024 或 2048");
      }
      const mode = "fast";
      const visionCompute = executionMode === "a" ? "gpu" : "ane";
      const previousKernel = profile.embedding.native?.privateANE || {};
      const recommendedPrivateSequence = ["c", "d"].includes(executionMode) ? 2112 : (previousKernel.sequenceLength || 2112);
      const recommendedRecurrenceTokens = executionMode === "d" ? 8192 : (previousKernel.recurrenceMaxTokens || 8192);
      const resolveOptional = (selected) => selected ? path.resolve(selected) : "";
      let layerSlots = String(value(
        "--private-ane-layer-slots",
        Array.isArray(previousKernel.recurrenceLayerSlots) ? previousKernel.recurrenceLayerSlots.join(",") : "0",
      )).split(",").map((item) => item.trim()).filter(Boolean).map(Number);
      if (executionMode === "d" && !layerSlots.length) layerSlots = [0];
      if (layerSlots.some((slot) => !Number.isInteger(slot) || slot < 0 || slot >= 18)) {
        throw new Error("--private-ane-layer-slots 只能包含 0..17 的整数");
      }
      if (args.some((arg) => ["--research-python", "--research-service", "--private-ane-experiment-root", "--private-ane-bridge", "--private-ane-mil"].includes(arg))) {
        throw new Error("Python 服务与外部桥接参数已移除，A/B/C/D 统一使用 Swift helper");
      }
      const recurrenceProfile = value("--private-ane-recurrence-profile", previousKernel.recurrenceProfile || "");
      profile.embedding = {
        ...profile.embedding,
        provider: "apple-native",
        baseUrl: "",
        model: appleEmbeddingModel(dimension),
        dimension,
        inputStyle: "wemm",
        native: {
          ...profile.embedding.native,
          binary: value("--binary", profile.embedding.native?.binary || ""),
          modelPackage: path.resolve(modelPackage),
          coreMLCache: value("--coreml-cache", profile.embedding.native?.coreMLCache || ""),
          executionMode,
          mode,
          visionCompute,
          privateANE: {
            ...previousKernel,
            sequenceLength: Number(value("--private-ane-sequence-length", recommendedPrivateSequence)),
            mlpFraction: Number(value("--private-ane-mlp-fraction", previousKernel.mlpFraction || 0.75)),
            mlpVariant: Number(value("--private-ane-mlp-variant", previousKernel.mlpVariant || 8)),
            mlpMaxLayers: Number(value("--private-ane-mlp-max-layers", previousKernel.mlpMaxLayers || 24)),
            recurrenceProfile: resolveOptional(recurrenceProfile),
            recurrenceBlockSize: Number(value("--private-ane-block-size", previousKernel.recurrenceBlockSize || 8)),
            recurrenceLayerSlots: layerSlots,
            recurrenceQueryScale: Number(value("--private-ane-query-scale", previousKernel.recurrenceQueryScale || 4096)),
            recurrenceMaxTokens: Number(value("--private-ane-max-tokens", recommendedRecurrenceTokens)),
            recurrenceIODtype: value("--private-ane-io-dtype", previousKernel.recurrenceIODtype || "fp16") === "fp32" ? "fp32" : "fp16",
            recurrenceVerifyReference: args.includes("--private-ane-verify-reference"),
          },
          maxQueuedRequests: Number(value("--max-queued-requests", profile.embedding.native?.maxQueuedRequests || 16)),
          autoRestart: !args.includes("--no-auto-restart"),
        },
      };
      profile.spaceId = value("--space", "");
      const { id: profileId, ...persistedProfile } = profile;
      config.profiles[profileId] = persistedProfile;
      writeConfig(config);
      return output({ ok: true, profile: profileId, embedding: publicConfig().profiles[profileId].embedding });
    }
    if (action === "install") {
      const source = value("--source");
      const indexedHome = indexedHomePath();
      const modelsDirectory = value("--models-dir", indexedHome ? path.join(indexedHome, "models") : "");
      if (!source || !modelsDirectory) throw new Error("install 需要 --source 和 --models-dir，或先设置 INDEXED_HOME");
      return output(await installAppleEmbeddingModel({
        binary: value("--binary", profile.embedding.native?.binary || ""),
        source,
        modelsDirectory,
        onStderr: (line) => process.stderr.write(`[indexed/apple-embedding] ${line}\n`),
      }));
    }
    if (!isAppleEmbeddingProfile(profile)) throw new Error("当前档案不是 apple-native embedding provider");
    const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile, {
      onStderr: (line) => process.stderr.write(`[indexed/apple-embedding] ${line}\n`),
    }));
    if (action === "validate") return output(await backend.validate(args.includes("--full")));
    if (action === "prepare") return output(await backend.prepare());
    if (action === "status") {
      try {
        const runtime = await backend.start();
        return output({ ...backend.status(), runtime: { ...runtime, apiKey: undefined }, health: await backend.health() });
      } finally {
        await backend.stop();
      }
    }
    throw new Error(`未知 embedding 命令：${action}`);
  }
  if (command === "status") return output(await status());
  if (command === "search") return output(await search(positional().join(" "), { limit: Number(value("--limit", 30)), videoId: value("--video") }));
  if (command === "videos") {
    const action = args.shift() || "list";
    if (action === "list") return output(await listVideos({ limit: Number(value("--limit", 100)) }));
    const [site, id] = positional();
    if (!site || !id) throw new Error("需要 SITE 和 VIDEO_ID");
    if (action === "delete") {
      if (!args.includes("--yes")) throw new Error("删除向量不可恢复；确认后添加 --yes");
      const config = loadConfig();
      const plan = await prepareVideoDeletion(site, id, config);
      return output(await deleteVideo(site, id, config, { confirmation: { schema: plan.schema, token: plan.token, confirmed: true } }));
    }
    if (action === "update") {
      const patch = {};
      for (const [flag, key] of [["--title", "title"], ["--note", "note"], ["--tags", "tags"]]) {
        const field = value(flag);
        if (field) patch[key] = key === "tags" ? field.split(",").map((item) => item.trim()).filter(Boolean) : field;
      }
      return output(await updateVideo(site, id, patch));
    }
  }
  if (command === "files") {
    const action = args.shift() || "list";
    if (action === "list") return output(await listLocalFiles({ limit: Number(value("--limit", 200)) }));
    if (action === "search") return output(await searchLocalFiles(positional().join(" "), { limit: Number(value("--limit", 30)) }));
    if (action === "scan") return output(await scanLocalLibrary({ limit: Number(value("--limit", 500)) }));
  }
  if (command === "assets") {
    const action = args.shift() || "libraries";
    if (action === "libraries") return output(await librariesStatus());
    if (action === "model-info") return output(await assetModelInfo());
    if (action === "estimate") return output((await libraryCoverage()).ingestEstimate);
    if (action === "list") {
      return output(await listAssets({
        limit: Number(value("--limit", 200)),
        kind: value("--kind"),
        library: value("--library"),
        includeMissing: args.includes("--include-missing"),
      }));
    }
    if (action === "search") {
      return output(await searchAllAssets(positional().join(" "), {
        limit: Number(value("--limit", 30)),
        kind: value("--kind"),
        includeMissing: args.includes("--include-missing"),
        includeLegacy: !args.includes("--no-legacy"),
      }));
    }
    if (action === "scan") {
      // A one-shot process cannot leave a queue behind, so the CLI scans inline and
      // reports progress on stderr; the resident server keeps the queued behaviour.
      const config = loadConfig();
      const target = positional()[0] || "";
      const targets = target ? [path.resolve(target)] : (await librariesStatus(config)).libraries.map((entry) => entry.path);
      if (!targets.length) throw new Error("尚未注册素材库目录，先执行 indexed assets add DIRECTORY");
      const limit = Number(value("--limit", 400));
      const results = [];
      for (const root of targets) {
        results.push(await scanLibrary(root, { config, limit, recordPerformance: true, onProgress: progressReporter(path.basename(root)) }));
        if (process.stderr.isTTY) process.stderr.write("\n");
      }
      return output(target ? results[0] : { ok: results.every((item) => item.ok), roots: results });
    }
    if (action === "prune") {
      return output(await pruneMissingAssets({
        library: value("--library", ""),
        dryRun: args.includes("--dry-run"),
      }));
    }
    if (action === "read" || action === "write") {
      const [file] = positional();
      if (!file) throw new Error("需要素材文件路径");
      if (action === "read") {
        const document = await readAssetDocument(file);
        // --raw is the pipe-friendly form: the text alone, no JSON envelope.
        if (args.includes("--raw")) return process.stdout.write(document.text);
        return output(document);
      }
      const source = value("--from", "");
      if (!source) throw new Error("需要 --from 文件路径，或 --from - 从标准输入读取内容");
      const text = source === "-" ? fs.readFileSync(0, "utf8") : fs.readFileSync(source, "utf8");
      return output(await writeAssetDocument(file, text));
    }
    if (action === "add" || action === "remove") {
      const [directory] = positional();
      if (!directory) throw new Error("需要素材库目录");
      if (action === "add") {
        const config = loadConfig();
        const registered = await addLibrary(directory, { config, scan: false });
        if (args.includes("--no-scan")) return output(registered);
        const scan = await scanLibrary(registered.library.path, {
          config,
          limit: Number(value("--limit", 400)),
          recordPerformance: true,
          onProgress: progressReporter(registered.library.name || registered.library.path),
        });
        if (process.stderr.isTTY) process.stderr.write("\n");
        return output({ ...registered, scan });
      }
      const purge = args.includes("--purge");
      // Purging deletes vectors and cannot be undone, so --purge alone is not enough.
      if (purge && !args.includes("--confirm")) throw new Error("清除素材向量不可恢复；确认后添加 --confirm");
      return output(await removeLibrary(directory, { purge }));
    }
  }
  if (command === "config") {
    const action = args.shift() || "show";
    if (action === "path") return output({ path: configPath() });
    if (action === "show") return output(publicConfig());
    const config = loadConfig();
    if (action === "use") {
      const id = args[0];
      if (!config.profiles[id]) throw new Error(`配置档案不存在：${id}`);
      config.activeProfile = id;
      writeConfig(config);
      return output({ ok: true, activeProfile: id });
    }
    if (action === "set") {
      const [key, raw] = args;
      if (!key || raw === undefined) throw new Error("需要 dotted.path 和 value");
      setPath(config, key, raw);
      writeConfig(config);
      return output({ ok: true, activeProfile: activeProfile(config).id, path: key });
    }
    if (action === "profiles") {
      const profileAction = args.shift() || "list";
      if (profileAction === "list") {
        return output({
          activeProfile: config.activeProfile,
          profiles: Object.entries(config.profiles).map(([id, profile]) => ({
            id,
            label: profile.label || id,
            model: profile.embedding?.model || "",
            dimension: Number(profile.embedding?.dimension || 0),
          })),
        });
      }
      const id = args.shift();
      if (!id || !/^[a-z0-9][a-z0-9-]{1,48}$/.test(id)) throw new Error("配置档案 ID 只能使用小写英文、数字和短横线");
      if (profileAction === "create") {
        if (config.profiles[id]) throw new Error(`配置档案已存在：${id}`);
        const sourceId = value("--from", config.activeProfile);
        if (!config.profiles[sourceId]) throw new Error(`源配置档案不存在：${sourceId}`);
        const created = JSON.parse(JSON.stringify(config.profiles[sourceId]));
        created.label = id;
        created.spaceId = "";
        created.storage.accessKeyId = "";
        created.storage.accessKeySecret = "";
        created.storage.securityToken = "";
        config.profiles[id] = created;
        config.activeProfile = id;
        writeConfig(config);
        return output({ ok: true, created: id, activeProfile: id, copiedFrom: sourceId });
      }
      if (profileAction === "delete") {
        if (!args.includes("--yes")) throw new Error("删除配置档案前添加 --yes；向量数据不会被删除");
        if (!config.profiles[id]) throw new Error(`配置档案不存在：${id}`);
        if (Object.keys(config.profiles).length === 1) throw new Error("至少保留一个配置档案");
        delete config.profiles[id];
        if (config.activeProfile === id) config.activeProfile = Object.keys(config.profiles)[0];
        writeConfig(config);
        return output({ ok: true, deleted: id, activeProfile: config.activeProfile });
      }
    }
  }
  usage();
}

function commandNeedsEmbeddingBackend() {
  if (command === "status" || command === "search") return true;
  const action = String(args[0] || "");
  if (command === "files") return new Set(["search", "scan"]).has(action);
  if (command === "assets") return new Set(["model-info", "search", "scan", "add", "estimate"]).has(action);
  return false;
}

async function run() {
  if (!commandNeedsEmbeddingBackend()) return main();
  const config = loadConfig();
  const profile = activeProfile(config);
  if (!isAppleEmbeddingProfile(profile)) return main();
  const backend = new AppleEmbeddingBackend(appleEmbeddingOptionsFromProfile(profile, {
    onStderr: (line) => process.stderr.write(`[indexed/apple-embedding] ${line}\n`),
  }));
  try {
    const runtime = await backend.start();
    setRuntimeEmbeddingOverride(profile.id, runtime);
    return await main();
  } finally {
    clearRuntimeEmbeddingOverride(profile.id);
    await backend.stop();
  }
}

run().catch((error) => {
  process.stderr.write(`${String(error?.message || error)}\n`);
  process.exitCode = 1;
});
