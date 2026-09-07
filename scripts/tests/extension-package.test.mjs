import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import test from 'node:test';
import { extensionTag, packageExtension } from '../package-extension.mjs';

test('extension tags use the browser manifest version', () => {
  assert.equal(extensionTag('0.14.6'), 'extension-v0.14.6');
  for (const version of ['0.0.0', '01.2', '1.2-beta', '65536.1', '1.2.3.4.5', '../1']) assert.throws(() => extensionTag(version));
});

test('extension ZIP is installable in layout and excludes local files and source maps', async context => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'indexed-extension-fixture-'));
  context.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const put = (file, value) => { fs.mkdirSync(path.dirname(path.join(root, file)), { recursive: true }); fs.writeFileSync(path.join(root, file), value); };
  put('LICENSE', 'Fixture license');
  put('apps/dashboard/indexed.svg', '<svg/>');
  for (const file of ['popup.html', 'search.html']) put(`apps/extension/static/${file}`, '<html></html>');
  for (const file of ['popup.css', 'search.css']) put(`apps/extension/static/${file}`, 'body{}');
  put('apps/extension/static/THIRD_PARTY_NOTICES.md', 'Fixture third-party attribution');
  put('apps/extension/static/icons/indexed.svg', '<svg/>');
  put('apps/extension/static/manifest.json', JSON.stringify({ manifest_version: 3, name: 'Fixture', version: '0.14.6', key: 'public-fixture-key', background: { service_worker: 'service-worker.js', type: 'module' }, action: { default_popup: 'popup.html' }, content_scripts: [{ js: ['content-script.js'], matches: ['https://example.invalid/*'] }] }));
  for (const file of ['service-worker', 'content-script', 'popup', 'search']) put(`apps/extension/src/${file}.ts`, 'export const fixture: number = 1;');
  const git = (...args) => execFileSync('git', args, { cwd: root, stdio: 'pipe' });
  git('init'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid'); git('add', '.'); git('commit', '-m', 'fixture');
  put('apps/extension/static/README.md', 'private local document');
  put('apps/extension/static/config.local.json', '{"local":true}');
  put('.indexed-extension-key', 'local-key-must-not-ship');
  put('dist/extension/stale.py', '# stale build');
  const result = await packageExtension(root, { tag: 'extension-v0.14.6' });
  const entries = execFileSync('unzip', ['-Z1', result.archive], { encoding: 'utf8' }).trim().split('\n');
  for (const required of ['manifest.json', 'service-worker.js', 'content-script.js', 'popup.js', 'search.js', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'SOURCE.json']) assert(entries.includes(required), required);
  assert(!entries.some(x => /README|config\.local|stale|\.map$|node_modules|\.indexed/.test(x)));
  const manifest = JSON.parse(execFileSync('unzip', ['-p', result.archive, 'manifest.json'], { encoding: 'utf8' }));
  assert.equal(manifest.key, 'public-fixture-key');
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(result.archive)).digest('hex'), result.sha256);
  assert(fs.readFileSync(path.join(root, 'dist/releases/SHA256SUMS'), 'utf8').includes(result.sha256));
  await assert.rejects(packageExtension(root, { tag: 'extension-v9.9.9' }), /must match manifest/);
  const source = 'apps/extension/static/manifest.json';
  const broken = JSON.parse(fs.readFileSync(path.join(root, source), 'utf8'));
  broken.background.service_worker = 'missing.js'; put(source, JSON.stringify(broken));
  await assert.rejects(packageExtension(root), /Missing extension entry/);
});
