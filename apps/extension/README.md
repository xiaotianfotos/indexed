# Extension

Target: Chrome Manifest V3 built with TypeScript and esbuild. Browser-specific code belongs here: player adapters, content scripts, service worker, IndexedDB queue, Chrome storage, popup, and extension search.

Reusable Embedding and OSS Vectors code belongs in `packages/clients`, not in the service worker.

Build the repository and load `dist/extension` as an unpacked Chrome extension. See the bilingual installation guides:

- [简体中文](../../docs/extension-installation.zh-CN.md)
- [English](../../docs/extension-installation.md)
