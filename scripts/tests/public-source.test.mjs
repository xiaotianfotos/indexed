import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import test from 'node:test';
import { auditRepository, inspectContent, privatePath } from '../audit-public-source.mjs';

test('public surface excludes private documents and artifacts while retaining legal and functional files', () => {
  for (const file of ['docs/plan.json', 'AGENTS.md', 'README.md', '.env.production', '.data/config.json', 'native/apple-embedding/results/run.json', 'model.safetensors', 'vision/foo.mlmodelc/weights/a.bin']) assert.equal(privatePath(file), true, file);
  for (const file of ['LICENSE', 'NOTICE', 'native/THIRD_PARTY_NOTICES.md', 'skills/indexed/SKILL.md', '.github/workflows/ci.yml', '.env.example', 'src/main.ts']) assert.equal(privatePath(file), false, file);
});

test('scanner checks binary metadata and returns locations without exposing credential contents', () => {
  const token = ['ghp', '_', 'A'.repeat(36)].join('');
  const findings = inspectContent('asset.png', Buffer.from('\0metadata\n' + token));
  assert.equal(findings[0].rule, 'GitHub token');
  assert.equal(JSON.stringify(findings).includes(token), false);
});

test('staged and published-history secrets cannot be hidden by editing or deleting the working file', context => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'indexed-public-scan-'));
  context.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const git = (...args) => execFileSync('git', args, { cwd: root, stdio: 'pipe' });
  git('init'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid');
  const token = ['ghp', '_', 'B'.repeat(36)].join('');
  fs.writeFileSync(path.join(root, 'source.js'), token); git('add', 'source.js');
  fs.writeFileSync(path.join(root, 'source.js'), '// clean\n');
  let result = auditRepository(root);
  assert(result.findings.some(x => x.source === 'index' && x.rule === 'GitHub token'));
  git('commit', '-m', 'fixture'); git('add', 'source.js'); git('commit', '-m', 'clean');
  result = auditRepository(root, { historyRefs: ['HEAD'] });
  assert(result.findings.some(x => x.source === 'history:HEAD' && x.rule === 'GitHub token'));
  assert(!result.findings.some(x => x.source === 'worktree'));
  fs.writeFileSync(path.join(root, '.gitignore'), 'docs/\n'); fs.mkdirSync(path.join(root, 'docs'));
  fs.writeFileSync(path.join(root, 'docs', 'local.md'), token);
  assert(!auditRepository(root).findings.some(x => x.file.startsWith('docs/')));
});

test('literal credential checks include prefixed constants and compound cloud key fields', () => {
  for (const field of ['SERVICE_API_KEY', 'accessKeySecret', 'aws_secret_access_key', 'securityToken']) {
    const value = 'Q'.repeat(24);
    const findings = inspectContent('src/config.ts', Buffer.from(`const ${field} = "${value}";`));
    assert(findings.some(x => x.rule === 'literal credential; review required'), field);
    assert(!JSON.stringify(findings).includes(value));
  }
});

test('only the two exact credential-redaction test sentinels are exempted', () => {
  const file = 'tests/ingest-performance.test.ts';
  const sentinel = ['secret', 'must', 'not', 'be', 'recorded'].join('-');
  assert.deepEqual(inspectContent(file, Buffer.from(`accessKeySecret: "${sentinel}"`)), []);
  const value = 'R'.repeat(24);
  assert(inspectContent(file, Buffer.from(`accessKeySecret: "${value}"`)).length > 0);
});
