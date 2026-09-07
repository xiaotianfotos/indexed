import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

const legal = /^(?:LICENSE|LICENCE|COPYING|NOTICE)(?:[.-].*)?$|^(?:THIRD_PARTY_NOTICES|ACKNOWLEDGMENTS)\.md$/i;
// Exact synthetic sentinels used to assert that credentials never enter ingest history.
const fixtureCredentials = new Map([['tests/ingest-performance.test.ts', new Set(['must-not-be-recorded', 'secret-must-not-be-recorded'])]]);
const rules = [
  ['private key', /-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----/g],
  ['cloud access key', /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bLTAI[A-Za-z0-9]{16,}\b/g],
  ['GitHub token', /\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b/g],
  ['provider token', /\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|ms-[A-Za-z0-9]{25,}|xox[baprs]-[A-Za-z0-9-]{20,})\b/g],
  ['credential URL', /https?:\/\/[^\s/"'<>:]+:[^\s/@"'<>]+@/g],
];

export function privatePath(file) {
  if (/[\x00-\x1f\x7f\\]/.test(file)) return true;
  const parts = file.split('/');
  const name = parts.at(-1);
  if (parts.some(x => ['.data', '.indexed', 'node_modules', 'dist', '.build', '__pycache__', '.venv'].includes(x))) return true;
  if (file.startsWith('docs/') || file.startsWith('native/apple-embedding/results/')) return true;
  if (/\.(?:md|markdown|mdx|rst|adoc)$/i.test(file) && !legal.test(name) && file !== 'README.md' && file !== 'skills/indexed/SKILL.md') return true;
  if (/^(?:config\.local\.json|secrets\.json|\.indexed-extension-key)$/.test(name)) return true;
  if (/^\.env(?:\.|$)/.test(name) && !/^\.env\.(?:example|sample|template)$/.test(name)) return true;
  return /\.(?:safetensors|sqlite3?|db|npz|pem|p12|pfx)$/i.test(name) || parts.some(x => /\.(?:mlmodelc|mlpackage)$/.test(x));
}

export function inspectContent(file, bytes) {
  const findings = [];
  // Match known token formats even in binary metadata. Never print matched bytes.
  const text = bytes.toString('utf8');
  for (const [rule, pattern] of rules) {
    pattern.lastIndex = 0;
    for (const match of text.matchAll(pattern)) {
      findings.push({ file, line: text.slice(0, match.index).split('\n').length, rule });
    }
  }
  if (!bytes.includes(0)) {
    const lines = text.split('\n');
    lines.forEach((line, i) => {
      // Only literal, quoted assignments; identifiers and template references are not credentials.
      const credential = /\b(?:[a-z0-9_]*(?:api[_-]?key|access[_-]?key(?:[_-]?(?:id|secret))?|secret(?:[_-]?access)?[_-]?key|secret|password|security[_-]?token|access[_-]?token|auth[_-]?token)[a-z0-9_]*)\b["']?\s*[:=]\s*["']([^"'\s]{16,})["']/i.exec(line);
      if (credential && !fixtureCredentials.get(file)?.has(credential[1]) && !/^(?:example|placeholder|your|dummy|fake|test|redacted|\$|<)/i.test(credential[1])) findings.push({ file, line: i + 1, rule: 'literal credential; review required' });
      if (/\/(?:Users|Volumes)\/[A-Za-z0-9_.-]+\//.test(line) && !(file === 'tests/local-vectors.test.ts' && line.includes(['', 'Volumes', 'example', ''].join('/')))) findings.push({ file, line: i + 1, rule: 'personal filesystem path' });
    });
  }
  return findings;
}

export function auditRepository(cwd, { historyRefs = [] } = {}) {
  const git = (...args) => execFileSync('git', args, { cwd, maxBuffer: 128 * 1024 * 1024 });
  const split = buffer => buffer.toString('utf8').split('\0').filter(Boolean);
  const findings = [];
  let worktreeFiles = 0;
  let indexFiles = 0;
  let historyBlobs = 0;
  const inspect = (file, bytes, source) => {
    if (privatePath(file)) findings.push({ source, file, rule: 'private/distribution-excluded path' });
    findings.push(...inspectContent(file, bytes).map(x => ({ source, ...x })));
  };
  const paths = new Set(split(git('ls-files', '--cached', '--others', '--exclude-standard', '-z')));
  for (const file of paths) {
    const absolute = path.join(cwd, file);
    if (!fs.existsSync(absolute)) continue;
    const stat = fs.lstatSync(absolute);
    if (!stat.isFile()) { findings.push({ source: 'worktree', file, rule: 'non-regular publication input' }); continue; }
    inspect(file, fs.readFileSync(absolute), 'worktree'); worktreeFiles++;
  }
  // Inspect the index separately: staged secrets must not be hidden by a clean working copy.
  for (const entry of split(git('ls-files', '--stage', '-z'))) {
    const match = /^(\d+) ([0-9a-f]+) (\d)\t([\s\S]+)$/.exec(entry);
    if (!match) throw new Error('Unrecognized Git index entry');
    const [, mode, oid, stage, file] = match;
    if (stage !== '0' || !['100644', '100755'].includes(mode)) findings.push({ source: 'index', file, rule: 'unmerged/non-regular index entry' });
    if (mode !== '160000') inspect(file, git('cat-file', 'blob', oid), 'index');
    indexFiles++;
  }
  const seen = new Set();
  for (const ref of historyRefs) {
    // Validate before using refs, and never walk --all (private backup refs are intentionally retained).
    const sha = git('rev-parse', '--verify', `${ref}^{commit}`).toString().trim();
    // Tree paths must be checked independently: rev-list can emit a reused blob
    // only once even when a historical commit stored it under a private path.
    for (const commit of git('rev-list', sha).toString().trim().split('\n')) {
      for (const file of split(git('ls-tree', '-r', '--name-only', '-z', commit))) {
        if (privatePath(file)) findings.push({ source: `history:${ref}`, file, rule: 'private historical tree path' });
      }
    }
    const objects = git('rev-list', '--objects', sha).toString().trim().split('\n');
    for (const object of objects) {
      const pos = object.indexOf(' '); if (pos < 0) continue;
      const oid = object.slice(0, pos), file = object.slice(pos + 1);
      if (seen.has(oid + ':' + file)) continue;
      seen.add(oid + ':' + file);
      if (git('cat-file', '-t', oid).toString().trim() !== 'blob') continue;
      inspect(file, git('cat-file', 'blob', oid), `history:${ref}`); historyBlobs++;
    }
  }
  return { worktreeFiles, indexFiles, historyBlobs, historyRefs, findings };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const args = process.argv.slice(2), historyRefs = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] !== '--history-ref' || !args[i + 1]) throw new Error('Usage: node scripts/audit-public-source.mjs [--history-ref REF]');
    historyRefs.push(args[++i]);
  }
  const result = auditRepository(process.cwd(), { historyRefs });
  console.log(JSON.stringify(result, null, 2));
  if (result.findings.length) process.exitCode = 1;
}
