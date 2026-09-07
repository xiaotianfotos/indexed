import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { inspectContent } from './audit-public-source.mjs';

export function extensionTag(version) {
  const parts = version.split('.');
  if (parts.length < 1 || parts.length > 4 || !parts.every(x => /^(0|[1-9]\d*)$/.test(x) && Number(x) <= 65535) || parts.every(x => Number(x) === 0)) throw new Error('Invalid Chrome extension version');
  return `extension-v${version}`;
}

export async function packageExtension(root, { tag, output = path.join(root, 'dist/releases') } = {}) {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'apps/extension/static/manifest.json'), 'utf8'));
  const expectedTag = extensionTag(manifest.version);
  if (tag && tag !== expectedTag) throw new Error(`Release tag must match manifest: ${expectedTag}`);
  if (manifest.manifest_version !== 3) throw new Error('Expected Chrome Manifest V3');
  const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
  const commit = git('rev-parse', 'HEAD');
  const tracked = new Set(git('ls-files').split('\n'));
  const stage = fs.mkdtempSync(path.join(os.tmpdir(), 'indexed-extension-release-'));
  const archiveName = `indexed-extension-${manifest.version}.zip`;
  try {
    const staticFiles = ['manifest.json', 'popup.html', 'popup.css', 'search.html', 'search.css', 'THIRD_PARTY_NOTICES.md', ...[...tracked].filter(x => /^apps\/extension\/static\/icons\/[\w.-]+\.(png|svg)$/.test(x)).map(x => x.slice('apps/extension/static/'.length))];
    for (const file of staticFiles) {
      const source = `apps/extension/static/${file}`;
      if (!tracked.has(source) || !fs.lstatSync(path.join(root, source)).isFile()) throw new Error(`Missing regular tracked extension asset: ${file}`);
      fs.mkdirSync(path.dirname(path.join(stage, file)), { recursive: true });
      fs.copyFileSync(path.join(root, source), path.join(stage, file));
    }
    // Always use the public manifest key, never developer key overrides.
    fs.copyFileSync(path.join(root, 'apps/dashboard/indexed.svg'), path.join(stage, 'icons/indexed.svg'));
    fs.copyFileSync(path.join(root, 'LICENSE'), path.join(stage, 'LICENSE'));
    fs.writeFileSync(path.join(stage, 'NOTICE'), 'Indexed browser extension\n\nSource: https://github.com/xiaotianfotos/indexed\nLicensed under Apache-2.0; see LICENSE.\nThird-party attribution is included in THIRD_PARTY_NOTICES.md.\n');
    for (const [format, entries] of [['esm', ['service-worker', 'popup']], ['iife', ['content-script', 'search']]]) {
      await build({ absWorkingDir: root, entryPoints: Object.fromEntries(entries.map(x => [x, `apps/extension/src/${x}.ts`])), outdir: stage, bundle: true, platform: 'browser', format, target: 'chrome120', sourcemap: false, legalComments: 'inline' });
    }
    fs.writeFileSync(path.join(stage, 'SOURCE.json'), JSON.stringify({ repository: 'https://github.com/xiaotianfotos/indexed', commit, tag: expectedTag, version: manifest.version }, null, 2) + '\n');
    const files = fs.readdirSync(stage, { recursive: true }).filter(x => fs.statSync(path.join(stage, x)).isFile()).sort();
    const refs = [manifest.background?.service_worker, manifest.action?.default_popup, ...Object.values(manifest.icons || {}), ...Object.values(manifest.action?.default_icon || {}), ...(manifest.content_scripts || []).flatMap(x => [...(x.js || []), ...(x.css || [])])].filter(Boolean);
    for (const ref of refs) if (!files.includes(ref)) throw new Error(`Missing extension entry: ${ref}`);
    for (const file of files) {
      const findings = inspectContent(file, fs.readFileSync(path.join(stage, file)));
      if (findings.length) throw new Error(`Release audit failed: ${JSON.stringify(findings)}`);
      if (file.endsWith('.js')) execFileSync(process.execPath, ['--check', path.join(stage, file)]);
    }
    // Stable metadata and an explicit archive list prevent OS files or stale output from leaking.
    for (const file of files) { fs.chmodSync(path.join(stage, file), 0o644); fs.utimesSync(path.join(stage, file), new Date('2020-01-01T00:00:00Z'), new Date('2020-01-01T00:00:00Z')); }
    execFileSync('zip', ['-X', '-q', path.join(stage, archiveName), ...files], { cwd: stage });
    execFileSync('unzip', ['-tq', path.join(stage, archiveName)]);
    fs.mkdirSync(output, { recursive: true });
    const archive = path.join(output, archiveName);
    fs.copyFileSync(path.join(stage, archiveName), archive);
    const sha256 = crypto.createHash('sha256').update(fs.readFileSync(archive)).digest('hex');
    fs.writeFileSync(path.join(output, 'SHA256SUMS'), `${sha256}  ${archiveName}\n`);
    fs.writeFileSync(path.join(output, 'release-notes.txt'), `Indexed 浏览器插件 ${manifest.version}（预发布）\n\n下载 ${archiveName} 并解压到固定目录。打开 chrome://extensions，开启开发者模式，选择“加载已解压的扩展程序”，选中解压后包含 manifest.json 的目录。\n\n插件安装无需 Node.js 或编译。使用本机模式需另行启动 Indexed 服务；本压缩包不包含服务器、Apple 原生后端或模型。\n\n更新时将新版解压到原安装目录，再在扩展管理页点击重新加载；请保留该目录，不要删除扩展的浏览器存储。\n\n源码与配置说明：https://github.com/xiaotianfotos/indexed#readme\n模型：https://modelscope.cn/models/xiaotianfotos/WeMM-Embedding-2B-Apple-Q8-G64\n\nSource commit: ${commit}\nTag: ${expectedTag}\nSHA256SUMS 可用于校验下载的 ZIP。该版本尚未上架 Chrome 商店。\n`);
    return { version: manifest.version, tag: expectedTag, archive, sha256, files };
  } finally {
    fs.rmSync(stage, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  if (args.length && (args.length !== 2 || args[0] !== '--tag')) throw new Error('Usage: node scripts/package-extension.mjs [--tag extension-vVERSION]');
  console.log(JSON.stringify(await packageExtension(process.cwd(), { tag: args[1] }), null, 2));
}
