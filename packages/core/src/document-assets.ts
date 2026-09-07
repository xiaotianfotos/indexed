import { execFile, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import WordExtractor from "word-extractor";

const run = promisify(execFile);

export const DOCUMENT_CHUNK_VERSION = "paragraph-window-v1";
export const DOCUMENT_CHUNK_CHARACTERS = 3_600;
export const DOCUMENT_CHUNK_OVERLAP = 320;
export const MAX_DOCUMENT_CHUNKS = 64;
export const MAX_STRUCTURED_DOCUMENT_BYTES = 64 * 1024 * 1024;

export const PLAIN_DOCUMENT_EXTENSIONS = new Set([
  ".md", ".mdx", ".txt", ".json", ".jsonl", ".csv", ".srt", ".vtt",
  ".ts", ".tsx", ".js", ".jsx", ".py", ".html", ".css",
]);
export const WORD_DOCUMENT_EXTENSIONS = new Set([".doc", ".docx"]);
export const PDF_DOCUMENT_EXTENSIONS = new Set([".pdf"]);

export interface DocumentChunk {
  index: number;
  text: string;
  startCharacter: number;
  endCharacter: number;
}

export interface DocumentChunks {
  chunks: DocumentChunk[];
  truncated: boolean;
  characters: number;
}

export interface DocumentFormatCapability {
  id: "plain" | "word" | "pdf";
  label: string;
  extensions: string[];
  available: boolean;
  note: string;
}

let pdfToolAvailable: boolean | undefined;

function commandAvailable(command: string): boolean {
  try {
    const result = spawnSync(command, ["-v"], { encoding: "utf8", timeout: 5_000 });
    return !result.error && result.status === 0;
  } catch {
    return false;
  }
}

export function documentFormatCapabilities(): DocumentFormatCapability[] {
  if (pdfToolAvailable === undefined) pdfToolAvailable = commandAvailable("pdftotext");
  return [
    {
      id: "plain",
      label: "文本",
      extensions: ["MD", "TXT", "CSV", "SRT", "JSON"],
      available: true,
      note: "按段落分块；笔记类文本可在 Indexed 内编辑",
    },
    {
      id: "word",
      label: "Word",
      extensions: ["DOC", "DOCX"],
      available: true,
      note: "提取正文并按段落分块；源文件只读",
    },
    {
      id: "pdf",
      label: "PDF",
      extensions: ["PDF"],
      available: pdfToolAvailable,
      note: pdfToolAvailable
        ? "支持带文本层的 PDF；扫描版暂不自动 OCR"
        : "需要安装 poppler-utils（pdftotext）；扫描版暂不自动 OCR",
    },
  ];
}

function normalizedText(value: string): string {
  return String(value || "")
    .replace(/\r\n?/g, "\n")
    .replace(/\0/g, " ")
    .replace(/[\t\f\v]+/g, " ")
    .replace(/[ \u00a0]+\n/g, "\n")
    .replace(/\n{4,}/g, "\n\n\n")
    .trim();
}

function safeTextBuffer(assetPath: string): string {
  const stat = fs.statSync(assetPath);
  if (stat.size > MAX_STRUCTURED_DOCUMENT_BYTES) {
    throw new Error(`文档过大（${stat.size} 字节），上限 ${MAX_STRUCTURED_DOCUMENT_BYTES} 字节`);
  }
  const buffer = fs.readFileSync(assetPath);
  if (buffer.includes(0)) throw new Error("文本文件包含空字节");
  const text = buffer.toString("utf8");
  if (text.includes("�")) throw new Error("文本文件不是 UTF-8 编码");
  return text;
}

async function wordText(assetPath: string): Promise<string> {
  const extracted = await new WordExtractor().extract(assetPath);
  return [
    extracted.getBody(),
    extracted.getFootnotes(),
    extracted.getEndnotes(),
    extracted.getTextboxes({ includeHeadersAndFooters: false, includeBody: true }),
  ].filter(Boolean).join("\n\n");
}

async function pdfText(assetPath: string): Promise<string> {
  if (!documentFormatCapabilities().find((item) => item.id === "pdf")?.available) {
    throw new Error("PDF 解析器不可用，请安装 poppler-utils（pdftotext）");
  }
  const { stdout } = await run(
    "pdftotext",
    ["-enc", "UTF-8", "-layout", assetPath, "-"],
    { encoding: "utf8", timeout: 120_000, maxBuffer: 32 * 1024 * 1024 },
  );
  return stdout;
}

/** Extract searchable text without ever modifying the source document. */
export async function extractDocumentText(assetPath: string): Promise<string> {
  const target = path.resolve(assetPath);
  const stat = fs.statSync(target);
  if (!stat.isFile()) throw new Error(`文档不存在：${target}`);
  if (stat.size > MAX_STRUCTURED_DOCUMENT_BYTES) {
    throw new Error(`文档过大（${stat.size} 字节），上限 ${MAX_STRUCTURED_DOCUMENT_BYTES} 字节`);
  }
  const extension = path.extname(target).toLowerCase();
  let raw = "";
  if (PLAIN_DOCUMENT_EXTENSIONS.has(extension)) raw = safeTextBuffer(target);
  else if (WORD_DOCUMENT_EXTENSIONS.has(extension)) raw = await wordText(target);
  else if (PDF_DOCUMENT_EXTENSIONS.has(extension)) raw = await pdfText(target);
  else throw new Error(`不支持的文档格式：${extension || "无扩展名"}`);
  const text = normalizedText(raw);
  if (!text) {
    const hint = extension === ".pdf" ? "（扫描版 PDF 需要 OCR）" : "";
    throw new Error(`没有提取到可读文本${hint}`);
  }
  return text;
}

function boundaryBefore(text: string, start: number, idealEnd: number): number {
  if (idealEnd >= text.length) return text.length;
  const floor = Math.min(idealEnd, start + Math.floor(DOCUMENT_CHUNK_CHARACTERS * 0.58));
  const window = text.slice(floor, idealEnd);
  const paragraph = window.lastIndexOf("\n\n");
  if (paragraph >= 0) return floor + paragraph + 2;
  const sentence = Math.max(
    window.lastIndexOf("。"), window.lastIndexOf("！"), window.lastIndexOf("？"),
    window.lastIndexOf(". "), window.lastIndexOf("! "), window.lastIndexOf("? "),
    window.lastIndexOf("\n"),
  );
  return sentence >= 0 ? floor + sentence + 1 : idealEnd;
}

/** Paragraph-aware bounded windows; every part of a long document remains recallable. */
export function chunkDocumentText(value: string): DocumentChunks {
  const text = normalizedText(value);
  if (!text) return { chunks: [], truncated: false, characters: 0 };
  const chunks: DocumentChunk[] = [];
  let cursor = 0;
  while (cursor < text.length && chunks.length < MAX_DOCUMENT_CHUNKS) {
    const idealEnd = Math.min(text.length, cursor + DOCUMENT_CHUNK_CHARACTERS);
    const end = Math.max(cursor + 1, boundaryBefore(text, cursor, idealEnd));
    const leading = text.slice(cursor, end).search(/\S/);
    const start = leading < 0 ? cursor : cursor + leading;
    const body = text.slice(start, end).trimEnd();
    if (body) {
      chunks.push({ index: chunks.length, text: body, startCharacter: start, endCharacter: start + body.length });
    }
    if (end >= text.length) break;
    cursor = Math.max(cursor + 1, end - DOCUMENT_CHUNK_OVERLAP);
    while (cursor < end && /\s/.test(text[cursor] || "")) cursor += 1;
  }
  const last = chunks.at(-1);
  return {
    chunks,
    truncated: Boolean(last && last.endCharacter < text.length),
    characters: text.length,
  };
}
