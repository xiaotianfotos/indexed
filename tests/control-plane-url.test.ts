import assert from "node:assert/strict";
import test from "node:test";
import {
  httpHostPermissionPattern,
  normalizeControlPlaneUrl,
} from "@indexed/clients/control-plane-url";

test("normalizes a bare IPv4 address to the Indexed default port", () => {
  assert.equal(normalizeControlPlaneUrl("192.0.2.20"), "http://192.0.2.20:18767");
  assert.equal(normalizeControlPlaneUrl(" indexed.local "), "http://indexed.local:18767");
});

test("preserves explicit schemes and ports while removing paths", () => {
  assert.equal(normalizeControlPlaneUrl("http://192.0.2.20:19000/"), "http://192.0.2.20:19000");
  assert.equal(normalizeControlPlaneUrl("https://indexed.example:443/settings"), "https://indexed.example");
});

test("rejects unsafe schemes and incomplete numeric addresses", () => {
  assert.throws(() => normalizeControlPlaneUrl("file:///tmp/indexed"), /http 或 https/);
  assert.throws(() => normalizeControlPlaneUrl("100.10"), /完整 IP 地址/);
  assert.throws(() => normalizeControlPlaneUrl(""), /请填写/);
});

test("creates Chrome host permission patterns without ports", () => {
  assert.equal(httpHostPermissionPattern("http://192.0.2.20:18767"), "http://192.0.2.20/*");
  assert.equal(httpHostPermissionPattern("https://indexed.example:8443"), "https://indexed.example/*");
});
