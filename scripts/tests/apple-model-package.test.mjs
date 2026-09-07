import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { preparePackage, verifyPackage, repository } from '../prepare-apple-model-package.mjs';

function fixture(context) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'indexed-model-package-'));
  context.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const source = path.join(root, 'source'); fs.mkdirSync(source);
  for (const f of ['LICENSE', 'tokenizer.json', 'tokenizer_config.json', 'processor_config.json', 'chat_template.jinja', 'embedding_chat_template.jinja', 'language/model.safetensors', 'vision/WeMMVision448.mlmodelc/model.bin']) {
    const p = path.join(source, f); fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, 'fixture');
  }
  fs.writeFileSync(path.join(source, 'language/config.json'), JSON.stringify({ model_type: 'qwen3_5', auto_map: { AutoModel: 'old.Python' } }));
  fs.mkdirSync(path.join(source, 'decoder-bundles')); fs.writeFileSync(path.join(source, 'decoder-bundles/old.bin'), 'retired');
  const name = Buffer.from('model.bin'), length = Buffer.alloc(8); length.writeBigUInt64BE(BigInt(name.length));
  const manifest = { schema_version: 1, runtime_semantics: 'wemm-apple-embedding-v1', package_fingerprint: 'a'.repeat(64), quantization: { bits: 8, group_size: 64 }, language: { path: 'language/model.safetensors', sha256: createHash('sha256').update('fixture').digest('hex') }, vision: { compiled: true, path: 'vision/WeMMVision448.mlmodelc', packaged_sha256_tree: createHash('sha256').update(length).update(name).update('fixture').digest('hex') } };
  fs.writeFileSync(path.join(source, 'manifest.json'), JSON.stringify(manifest));
  return { root, source, output: path.join(root, 'output') };
}

test('distribution preparation preserves source and tensor identity, links Indexed and excludes E', async context => {
  const { source, output } = fixture(context);
  const configBefore = fs.readFileSync(path.join(source, 'language/config.json'));
  await preparePackage(source, output);
  assert(configBefore.equals(fs.readFileSync(path.join(source, 'language/config.json'))));
  assert.equal(JSON.parse(fs.readFileSync(path.join(output, 'language/config.json'))).auto_map, undefined);
  assert(fs.readFileSync(path.join(output, 'README.md'), 'utf8').includes(repository));
  assert(fs.readFileSync(path.join(output, 'NOTICE'), 'utf8').includes(repository));
  assert.equal(fs.existsSync(path.join(output, 'decoder-bundles')), false);
  assert(fs.readFileSync(path.join(source, 'manifest.json')).equals(fs.readFileSync(path.join(output, 'manifest.json'))));
  await verifyPackage(output);
  await assert.rejects(preparePackage(source, output), /new directory/);
  await assert.rejects(preparePackage(source, path.join(source, 'nested')), /new directory/);
});

test('verification rejects tokenizer edits, missing files, extra files and path traversal', async context => {
  const { source, output } = fixture(context); await preparePackage(source, output);
  const file = path.join(output, 'tokenizer.json'); fs.writeFileSync(file, 'tampered');
  await assert.rejects(verifyPackage(output), /Checksum mismatch/);
  fs.writeFileSync(file, 'fixture'); fs.writeFileSync(path.join(output, 'extra.json'), '{}');
  await assert.rejects(verifyPackage(output), /inventory/); fs.unlinkSync(path.join(output, 'extra.json'));
  fs.unlinkSync(file); await assert.rejects(verifyPackage(output), /inventory/); fs.writeFileSync(file, 'fixture');
  fs.appendFileSync(path.join(output, 'checksums.sha256'), 'a'.repeat(64) + '  ../escape\n');
  await assert.rejects(verifyPackage(output), /Unsafe/);
});

test('corrupt source weights fail before creating distribution output', async context => {
  const { source, output } = fixture(context); fs.writeFileSync(path.join(source, 'language/model.safetensors'), 'corrupt');
  await assert.rejects(preparePackage(source, output), /digest mismatch/);
  assert.equal(fs.existsSync(output), false);
});
