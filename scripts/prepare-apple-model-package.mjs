import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

export const repository = 'https://github.com/xiaotianfotos/indexed';
const required = ['LICENSE', 'manifest.json', 'language/config.json', 'language/model.safetensors',
  'tokenizer.json', 'tokenizer_config.json', 'processor_config.json', 'chat_template.jinja', 'embedding_chat_template.jinja'];

function files(root, prefix = '') {
  return fs.readdirSync(path.join(root, prefix), { withFileTypes: true }).flatMap(entry => {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isSymbolicLink()) throw new Error(`Symlinks are not allowed in model packages: ${relative}`);
    if (entry.isDirectory()) return files(root, relative);
    if (!entry.isFile()) throw new Error(`Not a regular model file: ${relative}`);
    if (/\s/.test(relative)) throw new Error('Model package paths must not contain whitespace');
    return [relative];
  }).sort();
}
async function digest(file) {
  const hash = createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}
async function treeDigest(root) {
  const hash = createHash('sha256');
  for (const relative of files(root)) {
    const name = Buffer.from(relative), length = Buffer.alloc(8);
    length.writeBigUInt64BE(BigInt(name.length)); hash.update(length); hash.update(name);
    for await (const chunk of fs.createReadStream(path.join(root, relative))) hash.update(chunk);
  }
  return hash.digest('hex');
}

export async function verifyPackage(root) {
  const text = fs.readFileSync(path.join(root, 'checksums.sha256'), 'utf8');
  const entries = new Map();
  for (const line of text.trim().split('\n')) {
    const match = /^([a-f0-9]{64})  ([^\s]+)$/.exec(line);
    if (!match) throw new Error('Invalid checksum entry');
    const [, hash, file] = match;
    if (file.startsWith('/') || file.includes('\\') || file.split('/').some(x => x === '..' || x === '.') || entries.has(file)) throw new Error('Unsafe or duplicate checksum path');
    entries.set(file, hash);
  }
  const actual = files(root).filter(x => x !== 'checksums.sha256');
  if (JSON.stringify(actual) !== JSON.stringify([...entries.keys()].sort())) throw new Error('Checksum inventory does not match package files');
  for (const file of actual) if (await digest(path.join(root, file)) !== entries.get(file)) throw new Error(`Checksum mismatch: ${file}`);
  return { files: actual.length, bytes: actual.reduce((sum, file) => sum + fs.statSync(path.join(root, file)).size, 0) };
}

export async function preparePackage(source, output) {
  source = fs.realpathSync(source);
  output = path.resolve(output);
  // Resolve the existing parent too, so a parent symlink cannot point into the live model.
  const parent = fs.realpathSync(path.dirname(output));
  output = path.join(parent, path.basename(output));
  if (output === source || output.startsWith(source + path.sep) || fs.existsSync(output)) throw new Error('Output must be a new directory outside the source model');
  const manifest = JSON.parse(fs.readFileSync(path.join(source, 'manifest.json'), 'utf8'));
  if (manifest.schema_version !== 1 || manifest.runtime_semantics !== 'wemm-apple-embedding-v1'
    || manifest.language?.path !== 'language/model.safetensors'
    || manifest.vision?.path !== 'vision/WeMMVision448.mlmodelc' || manifest.vision.compiled !== true
    || manifest.quantization?.bits !== 8 || manifest.quantization?.group_size !== 64) throw new Error('Expected Indexed Apple Q8/G64 package v1');
  const inputs = files(source).filter(x => required.includes(x) || x.startsWith('vision/WeMMVision448.mlmodelc/'));
  for (const file of required) if (!inputs.includes(file)) throw new Error(`Missing model file: ${file}`);
  if (await digest(path.join(source, manifest.language.path)) !== manifest.language.sha256) throw new Error('Source language weight digest mismatch');
  if (await treeDigest(path.join(source, manifest.vision.path)) !== manifest.vision.packaged_sha256_tree) throw new Error('Source vision digest mismatch');
  fs.mkdirSync(output);
  for (const relative of inputs) {
    const destination = path.join(output, relative); fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.copyFileSync(path.join(source, relative), destination, fs.constants.COPYFILE_FICLONE);
  }
  const configPath = path.join(output, 'language/config.json');
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  // Indexed constructs its Swift architecture explicitly; the upstream Python
  // auto_map references code that is intentionally absent from this distribution.
  delete config.auto_map;
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2) + '\n');
  fs.writeFileSync(path.join(output, 'README.md'), `---
license: apache-2.0
pipeline_tag: feature-extraction
tags:
  - indexed
  - mlx
  - coreml
  - multimodal-embedding
  - quantized
base_model: tencent/WeMM-Embedding-2B
---

# WeMM-Embedding-2B Apple Q8/G64 for Indexed

This is a dedicated model package for [Indexed](${repository}), an open-source,
Apache-2.0 local multimodal search application. Download the application source
and use its native Swift Apple embedding backend to load this package.

This conversion derives from [Tencent WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B).
It contains MLX affine Q8 weights (group size 64), a compiled Core ML vision
encoder for 448-pixel inputs, the tokenizer, preprocessing configuration,
embedding templates, and the Indexed package manifest. Language weights omit
the generation head and duplicate vision weights. The config's obsolete
Python auto_map has been removed; model tensors and package identity are preserved.

## Runtime and compatibility

The whole repository is an Indexed-specific hybrid package. It is not a
standalone Transformers, mlx-lm, or generic MLX model download. Its supported
product runtime is Swift and requires no Python, pip, or model conversion at
installation. Core ML and MLX execution scheduling belongs to Indexed code.

A/B/C/D share this package. B is the default: Core ML vision with CPU/Neural
Engine routing and MLX GPU language. C/D are experimental and use private ANE
APIs in Indexed; model files alone do not provide those modes. See the
application's diagnostics for actual routing and fallback.

Real-model validation currently covers base Apple M4 on macOS 26. Other Apple
chips and OS versions require separate validation, especially for the compiled
Core ML asset and private ANE paths. No cross-device performance guarantee is made.

## Integrity and installation

Preserve the directory layout and all tokenizer/template files. Install the
local package through Indexed's Apple model setup. Versioned automatic download
integration is not yet provided by the application.

To verify every distributed file from an Indexed source checkout:

\`node scripts/prepare-apple-model-package.mjs --verify /path/to/model-package\`

The checksums cover configuration, tokenizer, templates, notices and both model
components. Use checksums from a trusted, pinned repository revision; a checksum
file by itself is not a signature. The current native installer separately checks
language and vision hashes from manifest.json; the complete distribution check
above is an additional verification step.

Package fingerprint: \`${manifest.package_fingerprint}\`.
The manifest defines the embedding-space identity and supported output dimensions.
Do not mix vectors from another package or embedding space in an existing index.

## License and attribution

Retain the complete LICENSE and NOTICE files. Tencent's original model and
third-party components retain their respective license terms. This is a
community conversion for Indexed, not an official Tencent or Apple release.
`);
  fs.writeFileSync(path.join(output, 'NOTICE'), `WeMM-Embedding-2B Apple Q8/G64 for Indexed\n\nApplication: ${repository}\nSource model: https://huggingface.co/tencent/WeMM-Embedding-2B\n\nThe original model is copyright Tencent and is distributed under its supplied\nLICENSE, including the third-party notices and terms contained in that file.\n\nIndexed conversion changes: split the language and vision components; quantize\nlanguage weights to MLX affine 8-bit with group size 64; omit the language\ngeneration head and duplicate vision weights; convert the vision encoder to\nCore ML; add Indexed package metadata and integrity information. Distribution\nconfiguration removes the upstream Python auto_map, without changing tensors.\n\nThe Indexed application is Apache-2.0 open-source software. Model artifacts keep\ntheir own supplied licenses. No endorsement by Tencent or Apple is implied.\n`);
  const lines = [];
  for (const relative of files(output)) lines.push(`${await digest(path.join(output, relative))}  ${relative}`);
  fs.writeFileSync(path.join(output, 'checksums.sha256'), lines.join('\n') + '\n');
  return { output, fingerprint: manifest.package_fingerprint, ...await verifyPackage(output) };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  if (args.length === 2 && args[0] === '--verify') console.log(JSON.stringify(await verifyPackage(path.resolve(args[1])), null, 2));
  else if (args.length === 4 && args[0] === '--source' && args[2] === '--output') console.log(JSON.stringify(await preparePackage(args[1], args[3]), null, 2));
  else throw new Error('Usage: --source MODEL_DIR --output NEW_DIR | --verify MODEL_DIR');
}
