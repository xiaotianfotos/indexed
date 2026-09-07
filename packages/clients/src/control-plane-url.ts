import { PRODUCT_DEFAULTS } from "@indexed/contracts";

export function normalizeControlPlaneUrl(
  input: string,
  defaultPort = PRODUCT_DEFAULTS.server.port,
): string {
  const raw = String(input || "").trim();
  if (!raw) throw new Error("请填写本地 Indexed 服务地址");

  const hasScheme = /^[a-z][a-z\d+.-]*:\/\//i.test(raw);
  const candidate = hasScheme ? raw : `http://${raw}`;
  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new Error("本地服务地址格式不正确");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("本地服务地址只支持 http 或 https");
  }
  if (parsed.username || parsed.password) {
    throw new Error("本地服务地址不能包含用户名或密码");
  }

  const suppliedAuthority = candidate.slice(candidate.indexOf("://") + 3).split(/[/?#]/, 1)[0] || "";
  const authorityWithoutCredentials = suppliedAuthority.slice(suppliedAuthority.lastIndexOf("@") + 1);
  const suppliedHost = authorityWithoutCredentials.startsWith("[")
    ? authorityWithoutCredentials.slice(1, authorityWithoutCredentials.indexOf("]"))
    : authorityWithoutCredentials.replace(/:\d+$/, "");
  if (/^\d+(?:\.\d+)+$/.test(suppliedHost)) {
    const octets = suppliedHost.split(".");
    if (octets.length !== 4 || octets.some((part) => Number(part) > 255)) {
      throw new Error("请输入完整 IP 地址，例如 10.0.0.20");
    }
  }

  const hasExplicitPort = authorityWithoutCredentials.startsWith("[")
    ? /\]:\d+$/.test(authorityWithoutCredentials)
    : /:\d+$/.test(authorityWithoutCredentials);
  if (!hasExplicitPort) parsed.port = String(defaultPort);

  return parsed.origin;
}

export function httpHostPermissionPattern(input: string): string {
  let parsed: URL;
  try {
    parsed = new URL(input);
  } catch {
    throw new Error("服务地址格式不正确");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("服务地址只支持 http 或 https");
  }
  return `${parsed.protocol}//${parsed.hostname}/*`;
}
