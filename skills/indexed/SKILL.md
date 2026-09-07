---
name: indexed
description: Search and manage Indexed video memory and local file libraries through its TypeScript CLI, including local zvec storage, model profiles, server status, metadata, deletion, and opening the Dashboard. Use for previously indexed YouTube or Bilibili content and user-configured local folders; do not use it as a downloader.
---

# Indexed

Use the Indexed CLI as the stable Agent interface. The Skill is not the web backend and should not automate Dashboard clicks when a structured command exists.

Resolve the executable before the first operation: prefer `indexed` from `PATH`; in a source checkout where it has not been linked yet, run `./bin/indexed` from the repository root. If neither resolves, report that the CLI is unavailable instead of substituting Dashboard calls.

## Operation routing

- Use `indexed status` when model or storage health is unknown.
- Use `indexed search "QUERY" --limit N` for retrieval. Keep visual and transcript rankings separate.
- Use `indexed videos list --limit N` to resolve exact video identifiers.
- Use `indexed files list`, `indexed files search`, and `indexed files scan` for configured local library roots.
- Use `indexed assets libraries`, `indexed assets list`, `indexed assets search "QUERY"`, and `indexed assets scan [DIRECTORY]` for the unified asset library. It includes video clips, images, and documents inside directories the user registered, plus web-video memories captured by the Chrome extension. Local hits carry `path`; web hits carry `openUrl` and should be opened in the browser. Scanning needs a local zvec profile; report the limitation instead of scanning elsewhere.
- Document search includes UTF-8 text, PDF files with a text layer, and `.doc`/`.docx`. Long documents are indexed as paragraph-aware chunks but returned as one best-passage result per file. A scanned PDF without a text layer is not searchable until an OCR workflow exists; do not report it as indexed merely because the file was discovered.
- For local-file results, `indexed assets search` never returns a result whose file is missing from disk; the response counts them in `hiddenMissing`. Pass `--include-missing` to see dead rows and `--no-legacy` to search only the registered libraries. When results point at files the user deleted by hand, run `indexed assets prune --dry-run` first, report the file and row counts, then `indexed assets prune` to drop those index rows — it removes index records only, never files.
- `indexed assets add DIRECTORY` registers a directory and scans it in the foreground, printing progress on stderr; pass `--no-scan` to register it and leave the work to the running server. `indexed assets scan [DIRECTORY]` (no directory: every registered root) rescans inline the same way.
- Use `indexed assets add DIRECTORY` only for a directory the user named. `indexed assets remove DIRECTORY` drops records only; adding `--purge --confirm` deletes their vectors, which is irreversible and requires the user's explicit confirmation.
- Use `indexed assets read FILE [--raw]` to read an editable text asset (`.md .mdx .txt .json .jsonl .csv .srt .vtt`) that already lives in a registered library, and `indexed assets write FILE --from PATH` (or `--from -` for standard input) to save it back. A save rewrites that one file in place, refreshes all of its document chunks and the excerpt its neighbouring clip quotes, and reports `vectors`, `removed`, and `previews`. PDF, Word, video and image bytes are always read-only.
- Use `indexed assets model-info` before promising a scan: it reports the embedding model, its advertised context length, and the vector dimension the profile writes.
- Use `indexed videos update SITE VIDEO_ID` for metadata changes requested by the user.
- Use `indexed serve --open` when the user asks to see or configure the Dashboard.
- Use `indexed config profiles` commands for model-space configuration.
- Legacy database migration tools are not provided during this test stage. If an old index is rejected, explain how to select a fresh storage directory and rescan the registered roots; never delete the old directory automatically.

Read operations and ordinary metadata updates may proceed within the user's request. Before deletion, resolve the exact `sourceSite` and `videoId`, explain that vector deletion is irreversible, obtain explicit authorization, and pass `--yes` only after that authorization.

Never print credentials or infer that equal vector dimensions make two Embedding models compatible. New video capture enters through the Chrome extension after the user chooses a video. Local files are scanned only from roots already configured by the user. Do not download or silently capture unrelated content.

If the TypeScript CLI command needed for a request has not yet landed during migration, report that limitation instead of fabricating a successful operation.
