import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  chunkDocumentText,
  documentFormatCapabilities,
  extractDocumentText,
} from "@indexed/core";

test("document format groups stay simple while naming the real parsers", () => {
  const formats = documentFormatCapabilities();
  assert.deepEqual(formats.map((item) => item.id), ["plain", "word", "pdf"]);
  assert.deepEqual(formats.find((item) => item.id === "word")?.extensions, ["DOC", "DOCX"]);
  assert.match(formats.find((item) => item.id === "pdf")?.note || "", /PDF|pdftotext|OCR/);
});

test("plain documents are extracted as UTF-8 and split through their final paragraph", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-document-chunks-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const file = path.join(directory, "long-note.md");
  const paragraphs = Array.from({ length: 90 }, (_, index) => `## 第 ${index + 1} 节\n${`这是用于验证长文档召回的第 ${index + 1} 段。`.repeat(8)}`);
  paragraphs.push("## 最后一节\n只有文档尾部才有的关键词：深海青铜小提琴。\n");
  fs.writeFileSync(file, paragraphs.join("\n\n"));

  const extracted = await extractDocumentText(file);
  const result = chunkDocumentText(extracted);
  assert.ok(result.chunks.length > 1);
  assert.equal(result.truncated, false);
  assert.match(result.chunks.at(-1)?.text || "", /深海青铜小提琴/);
  assert.ok(result.chunks.every((chunk) => chunk.text.length <= 3_600));
  assert.ok(result.chunks.slice(1).every((chunk, index) => chunk.startCharacter < result.chunks[index]!.endCharacter));
});

test("binary-looking text is rejected before it reaches the embedding service", async (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "indexed-document-binary-"));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const file = path.join(directory, "not-text.txt");
  fs.writeFileSync(file, Buffer.from([0, 1, 2, 3]));
  await assert.rejects(() => extractDocumentText(file), /空字节/);
});
