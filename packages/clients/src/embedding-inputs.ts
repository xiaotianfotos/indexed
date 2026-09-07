export type EmbeddingTextContent = {
  type: "text";
  text: string;
};

export type EmbeddingTextMessage = {
  role: "system" | "user";
  content: EmbeddingTextContent[];
};

export type EmbeddingCorpusModality = "text" | "image" | "video";

export const WEMM_TEXT_CORPUS_INPUT_VERSION = "wemm-text-user-v1";
export const WEMM_MEDIA_CORPUS_INPUT_VERSION = "wemm-media-user-v1";
export const QWEN_TEXT_CORPUS_INPUT_VERSION = "qwen-text-system-user-v1";
export const QWEN_MEDIA_CORPUS_INPUT_VERSION = "qwen-media-system-user-v1";

function clean(value: unknown): string {
  return String(value ?? "").trim();
}

export function isWemmInputStyle(inputStyle: unknown): boolean {
  return clean(inputStyle).toLowerCase() === "wemm";
}

export function userTextMessages(text: unknown): EmbeddingTextMessage[] {
  return [{ role: "user", content: [{ type: "text", text: clean(text) }] }];
}

export function systemInstructionMessages(text: unknown, instruction: unknown): EmbeddingTextMessage[] {
  return [
    { role: "system", content: [{ type: "text", text: clean(instruction) }] },
    { role: "user", content: [{ type: "text", text: clean(text) }] },
  ];
}

/**
 * WeMM's released examples encode corpus text as an ordinary user input. Other
 * input styles keep Indexed's established system-instruction representation.
 */
export function corpusTextMessages(
  text: unknown,
  instruction: unknown,
  inputStyle: unknown,
): EmbeddingTextMessage[] {
  return isWemmInputStyle(inputStyle)
    ? userTextMessages(text)
    : systemInstructionMessages(text, instruction);
}

/**
 * Text retrieval follows the asymmetric query format used by the instruction
 * retrieval pipeline: the task and query are one user message; documents remain
 * instruction-free. A system role would produce a different representation.
 */
export function retrievalQueryMessages(
  text: unknown,
  instruction: unknown,
  inputStyle: unknown,
): EmbeddingTextMessage[] {
  if (!isWemmInputStyle(inputStyle)) return systemInstructionMessages(text, instruction);
  return userTextMessages(`Instruct: ${clean(instruction)}\nQuery: ${clean(text)}`);
}

/** Multimodal task instructions in the official evaluator directly prefix the query. */
export function multimodalQueryMessages(
  text: unknown,
  instruction: unknown,
  inputStyle: unknown,
): EmbeddingTextMessage[] {
  if (!isWemmInputStyle(inputStyle)) return systemInstructionMessages(text, instruction);
  return userTextMessages(`${clean(instruction)} ${clean(text)}`);
}

export function corpusInputVersion(
  inputStyle: unknown,
  modality: EmbeddingCorpusModality,
): string {
  if (isWemmInputStyle(inputStyle)) {
    return modality === "text" ? WEMM_TEXT_CORPUS_INPUT_VERSION : WEMM_MEDIA_CORPUS_INPUT_VERSION;
  }
  return modality === "text" ? QWEN_TEXT_CORPUS_INPUT_VERSION : QWEN_MEDIA_CORPUS_INPUT_VERSION;
}
