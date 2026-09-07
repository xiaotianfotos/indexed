import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { manifest, prepareMetallib, selectArtifact } from '../prepare-metallib.mjs';

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const license = 'Copyright © 2023-2025 Apple Inc.\n';

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'indexed-metal-tool-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const source = path.join(root, 'archive');
  fs.mkdirSync(path.join(source, 'mlx/lib'), { recursive: true });
  fs.mkdirSync(path.join(source, path.dirname(manifest.licenseMember)), { recursive: true });
  fs.writeFileSync(path.join(source, manifest.member), 'synthetic-metal-test');
  // The test changes only the license hash expected by the injected fixture.
  fs.writeFileSync(path.join(source, manifest.licenseMember), license);
  fs.writeFileSync(path.join(source, 'should-not-ship.py'), 'raise RuntimeError("must not execute")');
  const archive = path.join(root, 'fixture.whl');
  execFileSync('zip', ['-qr', archive, '.'], { cwd: source });
  const bytes = fs.readFileSync(archive);
  return { root, archive, bytes, artifact: { minimumMacOS: 26, url: 'https://example.invalid/fixture.whl',
    size: bytes.length, archiveSha256: digest(bytes), metallibSha256: digest('synthetic-metal-test') } };
}

test('Metal build selects a pinned host-compatible archive and rejects unsupported hosts', () => {
  assert.equal(selectArtifact(14, 'darwin', 'arm64').minimumMacOS, 14);
  assert.equal(selectArtifact(15, 'darwin', 'arm64').minimumMacOS, 15);
  assert.equal(selectArtifact(26, 'darwin', 'arm64').metallibSha256, '198488eb61359e953580a9c4530400feee1a06dd2f28a930a6ffa58aec66a597');
  assert.throws(() => selectArtifact(13, 'darwin', 'arm64'), /requires/);
  assert.throws(() => selectArtifact(26, 'darwin', 'x64'), /requires/);
});

test('artifact hash, size and payload verification fail before publishing a library', async t => {
  const { root, archive, artifact, bytes } = fixture(t);
  const outputDirectory = path.join(root, 'output');
  for (const changed of [ { ...artifact, size: 1 }, { ...artifact, archiveSha256: '0'.repeat(64) },
    { ...artifact, metallibSha256: '0'.repeat(64) } ]) {
    await assert.rejects(prepareMetallib({ outputDirectory, artifact: changed, archivePath: archive }), /size|SHA-256/);
    assert.deepEqual(fs.readdirSync(outputDirectory), []);
  }
  await assert.rejects(prepareMetallib({ outputDirectory, artifact: { ...artifact, size: 1 },
    fetchArchive: async () => new Response(bytes) }), /exceeds pinned size/);
  assert.deepEqual(fs.readdirSync(outputDirectory), []);
});

test('only verified Metal and license files are published; cache tampering is rejected', async t => {
  const { root, archive, artifact, bytes } = fixture(t);
  const outputDirectory = path.join(root, 'output');
  const original = manifest.licenseSha256;
  manifest.licenseSha256 = digest(license);
  t.after(() => { manifest.licenseSha256 = original; });
  await prepareMetallib({ outputDirectory, artifact, fetchArchive: async () => new Response(bytes) });
  assert.deepEqual(fs.readdirSync(outputDirectory).sort(), ['MLX_METAL_ARTIFACT.json', 'mlx-metal-LICENSE.txt', 'mlx.metallib']);
  assert.equal(JSON.parse(fs.readFileSync(path.join(outputDirectory, 'MLX_METAL_ARTIFACT.json'), 'utf8')).archiveSha256, artifact.archiveSha256);
  await prepareMetallib({ outputDirectory, artifact, fetchArchive: async () => { throw new Error('cache must not fetch'); } });
  fs.writeFileSync(path.join(outputDirectory, 'mlx.metallib'), 'tampered');
  await assert.rejects(prepareMetallib({ outputDirectory, artifact, archivePath: archive }), /SHA-256 mismatch/);
  assert.equal(fs.readFileSync(path.join(outputDirectory, 'mlx.metallib'), 'utf8'), 'tampered');
});
