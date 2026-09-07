import assert from 'node:assert/strict';
import { spawn, execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import http from "node:http";
import os from 'node:os';
import path from 'node:path';
import test, { type TestContext } from 'node:test';
import { promisify } from 'node:util';

const binary = path.resolve(process.env.INDEXED_NATIVE_TEST_BINARY || 'native/apple-embedding/dist/apple-silicon/indexed-apple-embedding');
const run = promisify(execFile);
const hash = (bytes: string) => createHash('sha256').update(bytes).digest('hex');

test('Swift helper rejects unknown modes and unsupported private kernels before inference', async t => {
  await assert.rejects(run(binary, ['serve', '--execution-mode', 'unknown'], { timeout: 10_000 }), /必须是 a、b、c 或 d/);
  const { root } = modelPackage(t);
  for (const args of [ ['--execution-mode', 'c', '--private-ane-mlp-variant', '9'],
    ['--execution-mode', 'c', '--video-down-projection', 'fp16'],
    ['--execution-mode', 'd', '--private-ane-recurrence-block-size', '4'],
    ['--execution-mode', 'd', '--private-ane-recurrence-layer-slots', '0,2'],
    ['--execution-mode', 'b', '--video-pipeline', '2'],
    ['--execution-mode', 'c', '--research-python', '/absent/python'] ]) {
    await assert.rejects(run(binary, ['serve', '--package', root, ...args], { timeout: 10_000 }), /Native C\/D support|Native D requires|Private ANE options require|Unsupported native C\/D option/);
  }
});

test('real Swift executable refuses retired language-tower flags before model loading or installation', async () => {
  for (const args of [ ['serve', '--language-compute', 'ane'], ['serve', '--language-compute', 'auto'],
    ['serve', '--decoder-bundle', '/missing'], ['serve', '--execution-mode', 'E'],
    ['prepare-coreml', '--decoder-segment', '/missing'], ['install-model', '--decoder-loading', 'resident'] ]) {
    await assert.rejects(run(binary, args, { timeout: 10_000 }), /E 模式.*退出产品/);
  }
});

function modelPackage(t: TestContext) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'indexed-native-contract-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const language = 'synthetic-contract-test-only';
  fs.mkdirSync(path.join(root, 'vision.mlpackage'));
  fs.writeFileSync(path.join(root, 'model.safetensors'), language);
  fs.writeFileSync(path.join(root, 'tokenizer.json'), '{}');
  const manifest = {
    schema_version: 1, runtime_semantics: 'wemm-apple-embedding-v1', model: 'synthetic-contract-fixture',
    package_fingerprint: 'synthetic-contract-fixture', embedding_space_template: 'synthetic-{dimension}',
    embedding_token_id: 0, image_token_id: 1, video_token_id: 2,
    matryoshka_dimensions: [64, 128, 256, 512, 1024, 2048], tokenizer_files: ['tokenizer.json'],
    language: { backend: 'mlx', path: 'model.safetensors', precision: 'float16', sha256: hash(language),
      size_bytes: Buffer.byteLength(language), tensor_count: 0 },
    vision: { backend: 'coreml', compiled: false, image_size: 384, path: 'vision.mlpackage',
      packaged_sha256_tree: hash(''), size_bytes: 0 },
  };
  fs.writeFileSync(path.join(root, 'manifest.json'), JSON.stringify(manifest));
  return { root, manifest };
}

test('real Swift binary validates the package hash and rejects corruption and escaping paths', async t => {
  const { root, manifest } = modelPackage(t);
  const args = ['validate-model', '--package', root, '--full'];
  const result = await run(binary, args, { timeout: 20_000 });
  assert.equal(JSON.parse(result.stdout).status, 'valid');
  fs.writeFileSync(path.join(root, 'model.safetensors'), 'x'.repeat(manifest.language.size_bytes));
  await assert.rejects(run(binary, args, { timeout: 20_000 }), /SHA-256/);
  const outside = modelPackage(t);
  fs.symlinkSync(path.join(outside.root, 'model.safetensors'), path.join(root, 'escape.safetensors'));
  manifest.language.path = 'escape.safetensors';
  fs.writeFileSync(path.join(root, 'manifest.json'), JSON.stringify(manifest));
  await assert.rejects(run(binary, args, { timeout: 20_000 }), /越出模型目录/);
});

test('real Swift HTTP transport authenticates, isolates test semantics and exits gracefully', { timeout: 40_000 }, async t => {
  const { root } = modelPackage(t);
  const token = 'synthetic-native-contract-token';
  const child = spawn(binary, ['serve', '--package', root, '--port', '0', '--auth-token', token,
    '--development-deterministic-engine', '--skip-warmup'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  child.stderr.on('data', (chunk: Buffer) => { stderr = (stderr + chunk.toString()).slice(-32_768); });
  const exited = new Promise<number | null>((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', resolve);
  });
  // Register a handler immediately, including when startup fails before ready.
  void exited.catch(() => {});
  t.after(async () => { if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL'); await exited.catch(() => {}); });
  const ready = await new Promise<{ url: string; backend: string; default_embedding_space: string }>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Native ready timed out: ${stderr}`)), 20_000);
    let stdout = '';
    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString();
      if (!stdout.includes('\n')) return;
      clearTimeout(timer);
      try { resolve(JSON.parse(stdout.split('\n')[0]!)); } catch (error) { reject(error); }
    });
    void exited.then(code => { clearTimeout(timer); reject(new Error(`Native exited before ready (${code}): ${stderr}`)); }, reject);
  });
  assert.equal(ready.backend, 'deterministic-contract-test');
  assert.equal(ready.default_embedding_space, 'deterministic-test-2048');
  const request = async (route: string, method = 'GET', body?: unknown, authorized = true) => {
    const response = await fetch(`${ready.url}${route}`, { method,
      headers: { 'Content-Type': 'application/json', ...(authorized ? { Authorization: `Bearer ${token}` } : {}) },
      ...(body === undefined ? {} : { body: typeof body === 'string' ? body : JSON.stringify(body) }),
      signal: AbortSignal.timeout(10_000),
    });
    return { status: response.status, body: response.status === 204 ? {} : await response.json() };
  };
  assert.equal((await request('/health', 'GET', undefined, false)).status, 401);
  const health = await request('/health');
  assert.equal(health.body.status, 'ready');
  assert.equal(health.body.running_requests, 0);
  assert.equal(health.body.max_queued_requests, 16);
  const models = await request('/v1/models');
  assert.equal(models.body.data.length, 6);
  for (const model of models.body.data as Array<{ id: string; dimension: number }>) {
    const result = await request('/v1/embeddings', 'POST', { model: model.id, input: '窗边的猫', request_id: `native-${model.dimension}` });
    assert.equal(result.status, 200);
    assert.deepEqual(result.body.data[0].embedding, [1, ...Array(model.dimension - 1).fill(0)]);
    assert.equal(result.body.indexed.embedding_space, `deterministic-test-${model.dimension}`);
    assert.equal(result.body.indexed.request_id, `native-${model.dimension}`);
  }
  assert.equal((await request('/v1/embeddings', 'POST', { model: 'wemm-embedding-2b-apple-12', input: 'bad' })).status, 400);
  assert.equal((await request('/v1/embeddings', 'POST', '{not-json')).status, 400);
  assert.equal((await request('/missing')).status, 404);
  assert.equal((await request('/v1/requests/unknown', 'DELETE')).status, 404);
  const metrics = (await request('/health')).body;
  assert.equal(metrics.requests_succeeded, 6);
  assert.equal(metrics.requests_failed, 2);
  assert.equal(metrics.running_requests, 0);
  assert.equal(metrics.queued_requests, 0);
  child.kill('SIGTERM');
  assert.equal(await exited, 0);
});

async function delayedNative(t: TestContext) {
  const { root } = modelPackage(t);
  const token = 'isolated-native-cancellation-fixture';
  const child = spawn(binary, ['serve', '--package', root, '--port', '0', '--default-dimension', '64',
    '--development-deterministic-engine', '--development-delay-ms', '1000', '--skip-warmup'],
    { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, INDEXED_APPLE_EMBEDDING_AUTH_TOKEN: token } });
  let stderr = ''; child.stderr.on('data', chunk => { stderr += String(chunk); });
  const exited = new Promise<void>((resolve, reject) => { child.once('exit', () => resolve()); child.once('error', reject); });
  void exited.catch(() => {});
  t.after(async () => {
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
    const timer = setTimeout(() => child.kill('SIGKILL'), 3000);
    await exited.catch(() => {}); clearTimeout(timer);
  });
  const ready = await new Promise<{ url: string }>((resolve, reject) => {
    let stdout = ''; const timer = setTimeout(() => reject(new Error(stderr || 'Native startup timed out')), 10_000);
    child.stdout.on('data', chunk => {
      stdout += String(chunk); if (!stdout.includes('\n')) return;
      clearTimeout(timer); try { resolve(JSON.parse(stdout.split('\n')[0]!)); } catch (error) { reject(error); }
    });
    void exited.then(() => { clearTimeout(timer); reject(new Error(stderr || 'Native exited')); }, reject);
  });
  const headers = { authorization: `Bearer ${token}`, 'content-type': 'application/json' };
  const health = () => fetch(`${ready.url}/health`, { headers }).then(res => res.json());
  async function until(predicate: (value: Record<string, number>) => boolean) {
    const deadline = Date.now() + 3000;
    while (!predicate(await health())) { assert(Date.now() < deadline, 'Native work did not reach expected state'); await new Promise(resolve => setTimeout(resolve, 10)); }
  }
  return { url: ready.url, token, headers, health, until };
}

test('real Swift socket and router release running/queued work on disconnect and deadline, then accept another request', { timeout: 20_000 }, async t => {
  const helper = await delayedNative(t);
  const embed = (id: string, extra: Record<string, string> = {}, signal?: AbortSignal) => fetch(`${helper.url}/v1/embeddings`, {
    method: 'POST', headers: { ...helper.headers, ...extra }, body: JSON.stringify({ input: 'synthetic fixture', request_id: id }),
    ...(signal ? { signal } : {}),
  });
  const controller = new AbortController();
  const busy = embed('busy', {}, controller.signal);
  const rejected = assert.rejects(busy, { name: 'AbortError' });
  await helper.until(value => value.running_requests === 1);
  const queued = await embed('queued', { 'x-indexed-timeout-ms': '20' });
  assert.equal(queued.status, 504); assert.equal((await queued.json()).error.code, 'request_timeout');
  controller.abort(); await rejected;
  await helper.until(value => value.running_requests === 0 && value.queued_requests === 0);
  const runningTimeout = await embed('deadline', { 'x-indexed-timeout-ms': '40' });
  assert.equal(runningTimeout.status, 504); await runningTimeout.text();
  const invalid = await embed('invalid', { 'x-indexed-timeout-ms': 'NaN' });
  assert.equal(invalid.status, 400); await invalid.text();
  const slowUpload = await new Promise<{ status: number; body: string }>((resolve, reject) => {
    const request = http.request(`${helper.url}/v1/embeddings`, { method: 'POST', headers: {
      ...helper.headers, 'content-length': '1000', 'x-indexed-timeout-ms': '40',
    } }, response => {
      let body = ''; response.on('data', chunk => { body += String(chunk); });
      response.on('end', () => { resolve({ status: response.statusCode!, body }); request.destroy(); });
    });
    request.on('error', reject); request.write('{');
  });
  assert.equal(slowUpload.status, 504); assert.equal(JSON.parse(slowUpload.body).error.code, 'request_timeout');
  const recovered = await embed('recovered'); assert.equal(recovered.status, 200);
  const data = await recovered.json(); assert.equal(data.data[0].embedding.length, 64); assert.equal(data.data[0].embedding[0], 1);
  await helper.until(value => value.running_requests === 0 && value.queued_requests === 0);
});

test('actual Server → Core/client → Swift helper propagates search cancellation and timeout', { timeout: 20_000 }, async t => {
  const { startServer } = await import('@indexed/server');
  const { loadConfig, writeConfig } = await import('@indexed/config');
  const helper = await delayedNative(t);
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'indexed-native-cancel-stack-'));
  const oldConfig = process.env.INDEXED_CONFIG, oldState = process.env.INDEXED_STATE_DIR;
  process.env.INDEXED_CONFIG = path.join(root, 'config.json'); process.env.INDEXED_STATE_DIR = path.join(root, 'state');
  t.after(() => {
    if (oldConfig === undefined) delete process.env.INDEXED_CONFIG; else process.env.INDEXED_CONFIG = oldConfig;
    if (oldState === undefined) delete process.env.INDEXED_STATE_DIR; else process.env.INDEXED_STATE_DIR = oldState;
    fs.rmSync(root, { recursive: true, force: true });
  });
  const config = loadConfig();
  config.library = { ...config.library, roots: [], autoScan: false };
  config.profiles.default.embedding = { ...config.profiles.default.embedding, provider: 'http', baseUrl: helper.url,
    apiKey: helper.token, model: 'wemm-embedding-2b-apple-64', dimension: 64 };
  config.profiles.default.storage = { ...config.profiles.default.storage, provider: 'local', path: path.join(root, 'vectors') };
  writeConfig(config);
  const server = await startServer({ host: '127.0.0.1', port: 0 }); t.after(() => server.close());
  for (const route of ['api/assets/search', 'api/embedding/v1/embeddings']) {
    const controller = new AbortController();
    const promise = fetch(`${server.url}${route}`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ query: 'fixed fixture', kind: 'document', input: 'fixed fixture' }), signal: controller.signal });
    const rejected = assert.rejects(promise, { name: 'AbortError' });
    await helper.until(value => value.running_requests === 1); controller.abort(); await rejected;
    await helper.until(value => value.running_requests === 0 && value.queued_requests === 0);
  }
  const timed = await fetch(`${server.url}api/assets/search`, { method: 'POST',
    headers: { 'content-type': 'application/json', 'x-indexed-timeout-ms': '80' },
    body: JSON.stringify({ query: 'deadline fixture', kind: 'document' }) });
  assert.equal(timed.status, 504); await timed.text();
  await helper.until(value => value.running_requests === 0 && value.queued_requests === 0);
  const recovery = await fetch(`${server.url}api/assets/search`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ query: 'recovery fixture', kind: 'document' }) });
  assert.equal(recovery.status, 200); assert.deepEqual((await recovery.json()).hits, []);
});
