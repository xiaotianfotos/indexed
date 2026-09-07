import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { activeProfile, configPath, directConfig, indexedHomePath, ingestPerformanceHistoryPath, loadConfig, mergeProfile, normalizeConfig, publicConfig, resolveLocalStoragePath, setPath, spaceId, writeConfig } from "@indexed/config";

test("INDEXED_HOME keeps local deployment state under one root", () => {
  const previous = {
    home: process.env.INDEXED_HOME,
    config: process.env.INDEXED_CONFIG,
    data: process.env.INDEXED_DATA_DIR,
    state: process.env.INDEXED_STATE_DIR,
  };
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-home-"));
  try {
    process.env.INDEXED_HOME = directory;
    delete process.env.INDEXED_CONFIG;
    delete process.env.INDEXED_DATA_DIR;
    delete process.env.INDEXED_STATE_DIR;
    assert.equal(indexedHomePath(), directory);
    assert.equal(configPath(), path.join(directory, "config", "config.json"));
    assert.equal(resolveLocalStoragePath(), path.join(directory, "data", "zvec"));
    assert.equal(ingestPerformanceHistoryPath(), path.join(directory, "state", "ingest-performance.json"));
  } finally {
    for (const [key, value] of Object.entries({
      INDEXED_HOME: previous.home,
      INDEXED_CONFIG: previous.config,
      INDEXED_DATA_DIR: previous.data,
      INDEXED_STATE_DIR: previous.state,
    })) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("configuration is file-backed, normalized, redacted and model-scoped", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-config-"));
  const target = path.join(directory, "config.json");
  process.env.INDEXED_CONFIG = target;
  process.env.INDEXED_STATE_DIR = path.join(directory, "state");
  const config = loadConfig();
  setPath(config, "profiles.default.embedding.model", "wemm-embedding-9b");
  setPath(config, "profiles.default.embedding.inputStyle", "wemm");
  setPath(config, "profiles.default.storage.accessKeySecret", "secret-value");
  setPath(config, "profiles.default.storage.path", "~/.indexed-test");
  writeConfig(config);
  const loaded = loadConfig();
  const profile = activeProfile(loaded);
  assert.equal(profile.embedding.model, "wemm-embedding-9b");
  assert.match(spaceId(profile), /^wemm-embedding-9b-4096-wemm-indexed-v1$/);
  assert.equal(directConfig(profile).ossAccessKeySecret, "secret-value");
  assert.equal(publicConfig(loaded).profiles.default.storage.accessKeySecret, "");
  assert.equal(publicConfig(loaded).profiles.default.resolvedStoragePath, path.join(os.homedir(), ".indexed-test"));
  assert.equal(publicConfig(loaded).performanceHistoryPath, path.join(directory, "state", "ingest-performance.json"));
  assert.equal(ingestPerformanceHistoryPath(), path.join(directory, "state", "ingest-performance.json"));
  assert.equal(resolveLocalStoragePath("~/.indexed"), path.join(os.homedir(), ".indexed"));
  assert.equal(fs.statSync(target).mode & 0o777, 0o600);
});

test("legacy E stays visible with an error instead of silently becoming B", () => {
  const legacyANE = normalizeConfig({
    activeProfile: "apple",
    profiles: {
      apple: { embedding: { native: { mode: "ane", visionCompute: "ane" } } },
    },
  });
  assert.equal(legacyANE.profiles.apple.embedding.native.executionMode, "e");
  assert.equal(legacyANE.profiles.apple.embedding.native.mode, "ane");
  assert.match(legacyANE.profiles.apple.embedding.native.executionModeIssue, /E 模式已退出/);

  const legacyGPU = normalizeConfig({
    activeProfile: "apple",
    profiles: {
      apple: { embedding: { native: { mode: "fast", visionCompute: "gpu" } } },
    },
  });
  assert.equal(legacyGPU.profiles.apple.embedding.native.executionMode, "a");
  assert.equal(legacyGPU.profiles.apple.embedding.native.visionCompute, "gpu");
});

test("saving an unsupported Apple mode preserves the file; explicit B retires obsolete controls", context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-retired-mode-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const filename = path.join(directory, "config.json");
  const legacy = { activeProfile: "apple", profiles: { apple: { spaceId: "old-space", embedding: {
    provider: "apple-native", native: { executionMode: "e", mode: "ane", decoderBundles: ["old-bundle"] },
  }, storage: { provider: "local", path: "old-index" } } } };
  const original = JSON.stringify(legacy);
  fs.writeFileSync(filename, original);
  assert.throws(() => writeConfig(legacy, filename), /E 模式已退出/);
  assert.equal(fs.readFileSync(filename, "utf8"), original);
  legacy.profiles.apple.embedding.native.executionMode = "invalid";
  assert.throws(() => writeConfig(legacy, filename), /重新选择模式/);
  legacy.profiles.apple.embedding.native.executionMode = "b";
  const saved = writeConfig(legacy, filename);
  assert.equal(saved.profiles.apple.embedding.native.executionMode, "b");
  assert.equal(saved.profiles.apple.embedding.native.mode, "fast");
  assert.equal(saved.profiles.apple.embedding.native.decoderBundles, undefined);
  assert.equal(saved.profiles.apple.spaceId, "old-space");
  assert.equal(saved.profiles.apple.storage.path, "old-index");
});

test("obsolete benchmark flags do not turn ordinary directories into forced reindex jobs", () => {
  assert.equal(normalizeConfig({ library: { roots: ["/demo"], benchmarkRoots: ["/demo"] } }).library.benchmarkRoots, undefined);
});

test("Apple-native profiles use the fixed model ID derived from dimension", () => {
  const normalized = normalizeConfig({
    activeProfile: "apple",
    profiles: {
      apple: {
        spaceId: "manually-mixed-space",
        embedding: {
          provider: "apple-native",
          baseUrl: "http://example.invalid",
          model: "custom-model-name",
          dimension: 512,
          inputStyle: "qwen",
        },
      },
    },
  });
  assert.equal(normalized.profiles.apple.embedding.model, "wemm-embedding-2b-apple-512");
  assert.equal(normalized.profiles.apple.embedding.baseUrl, "");
  assert.equal(normalized.profiles.apple.embedding.inputStyle, "wemm");

  const merged = mergeProfile(normalized.profiles.apple, {
    spaceId: "another-space",
    embedding: { model: "another-custom-name", dimension: 1024 },
  });
  assert.equal(merged.embedding.model, "wemm-embedding-2b-apple-1024");
  assert.equal(merged.spaceId, "");
});
