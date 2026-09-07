import { spawnSync, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const manifest = JSON.parse(fs.readFileSync(new URL('./mlx-metal-artifacts.json', import.meta.url), 'utf8'));

export function selectArtifact(major, platform = process.platform, arch = process.arch) {
  if (platform !== 'darwin' || arch !== 'arm64' || !Number.isInteger(major) || major < 14) {
    throw new Error('The Apple helper build requires macOS 14+ on arm64.');
  }
  return manifest.artifacts.filter(item => item.minimumMacOS <= major).at(-1);
}

async function sha256(filename) {
  const hash = createHash('sha256');
  for await (const chunk of fs.createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

async function verify(filename, expected) {
  if (await sha256(filename) !== expected) throw new Error(`SHA-256 mismatch: ${path.basename(filename)}`);
}

function extract(archive, member, target) {
  // Only these named members are extracted. No Python modules, dylibs or wheel
  // installer execute, and archive paths never become filesystem destinations.
  const fd = fs.openSync(target, 'wx', 0o644);
  try {
    const result = spawnSync('/usr/bin/unzip', ['-p', archive, member], {
      stdio: ['ignore', fd, 'pipe'], timeout: 120_000,
    });
    if (result.error || result.status !== 0) throw new Error(`Cannot extract ${member}: ${result.error || result.stderr}`);
  } finally { fs.closeSync(fd); }
}

export async function prepareMetallib({ outputDirectory, artifact, archivePath, fetchArchive = fetch }) {
  fs.mkdirSync(outputDirectory, { recursive: true });
  const output = path.join(outputDirectory, 'mlx.metallib');
  const license = path.join(outputDirectory, 'mlx-metal-LICENSE.txt');
  const provenance = path.join(outputDirectory, 'MLX_METAL_ARTIFACT.json');
  // A stale/corrupt cached artifact must not silently enter a release.
  if (fs.existsSync(output)) await verify(output, artifact.metallibSha256);
  if (fs.existsSync(license)) await verify(license, manifest.licenseSha256);
  if (!fs.existsSync(output) || !fs.existsSync(license)) {
    const scratch = fs.mkdtempSync(path.join(outputDirectory, '.metallib-'));
    try {
      const archive = path.join(scratch, 'source.whl');
      if (archivePath) fs.copyFileSync(archivePath, archive, fs.constants.COPYFILE_EXCL);
      else {
        const response = await fetchArchive(artifact.url, { signal: AbortSignal.timeout(180_000), redirect: 'error' });
        if (!response.ok || !response.body) throw new Error(`Metal artifact download failed: ${response.status}`);
        let received = 0;
        await pipeline(Readable.fromWeb(response.body), new Transform({
          transform(chunk, _encoding, callback) {
            received += chunk.length;
            callback(received > artifact.size ? new Error('Metal artifact exceeds pinned size') : null, chunk);
          },
        }), fs.createWriteStream(archive, { flags: 'wx', mode: 0o600 }));
      }
      if (fs.statSync(archive).size !== artifact.size) throw new Error('Metal artifact size differs from pinned size');
      await verify(archive, artifact.archiveSha256);
      const stagedMetal = path.join(scratch, 'mlx.metallib');
      const stagedLicense = path.join(scratch, 'LICENSE');
      extract(archive, manifest.member, stagedMetal);
      extract(archive, manifest.licenseMember, stagedLicense);
      await verify(stagedMetal, artifact.metallibSha256);
      await verify(stagedLicense, manifest.licenseSha256);
      fs.renameSync(stagedLicense, license);
      fs.renameSync(stagedMetal, output);
    } finally { fs.rmSync(scratch, { recursive: true, force: true }); }
  }
  fs.writeFileSync(provenance, `${JSON.stringify({ version: manifest.version, ...artifact,
    source: manifest.source, member: manifest.member, licenseSha256: manifest.licenseSha256 }, null, 2)}\n`);
  return output;
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  const major = Number(execFileSync('/usr/bin/sw_vers', ['-productVersion'], { encoding: 'utf8' }).trim().split('.')[0]);
  const outputDirectory = path.resolve(process.argv[2] || fileURLToPath(new URL('./swift/.build/release', import.meta.url)));
  console.log(await prepareMetallib({ outputDirectory, artifact: selectArtifact(major),
    archivePath: process.env.INDEXED_MLX_METAL_ARCHIVE }));
}
