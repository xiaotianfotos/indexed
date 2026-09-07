import assert from "node:assert/strict";
import test from "node:test";
import {
  corpusInputVersion,
  corpusTextMessages,
  multimodalQueryMessages,
  retrievalQueryMessages,
  WEMM_MEDIA_CORPUS_INPUT_VERSION,
  WEMM_TEXT_CORPUS_INPUT_VERSION,
} from "../packages/clients/src/embedding-inputs.js";

test("WeMM keeps corpus text instruction-free", () => {
  assert.deepEqual(corpusTextMessages("  文档正文  ", "不要进入索引", "wemm"), [
    { role: "user", content: [{ type: "text", text: "文档正文" }] },
  ]);
  assert.equal(corpusInputVersion("wemm", "text"), WEMM_TEXT_CORPUS_INPUT_VERSION);
  assert.equal(corpusInputVersion("wemm", "video"), WEMM_MEDIA_CORPUS_INPUT_VERSION);
});

test("WeMM puts retrieval instructions and queries in one user message", () => {
  assert.deepEqual(retrievalQueryMessages("  海岸线航拍  ", "Find the matching asset.", "wemm"), [
    {
      role: "user",
      content: [{ type: "text", text: "Instruct: Find the matching asset.\nQuery: 海岸线航拍" }],
    },
  ]);
  assert.deepEqual(multimodalQueryMessages("  海岸线航拍  ", "Find a video:", "wemm"), [
    { role: "user", content: [{ type: "text", text: "Find a video: 海岸线航拍" }] },
  ]);
});

test("non-WeMM styles preserve the system and user compatibility shape", () => {
  const expected = [
    { role: "system", content: [{ type: "text", text: "Represent this document." }] },
    { role: "user", content: [{ type: "text", text: "正文" }] },
  ];
  assert.deepEqual(corpusTextMessages("正文", "Represent this document.", "qwen"), expected);
  assert.deepEqual(retrievalQueryMessages("正文", "Represent this document.", "qwen"), expected);
  assert.deepEqual(multimodalQueryMessages("正文", "Represent this document.", "qwen"), expected);
});
