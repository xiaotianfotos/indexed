export const APPLE_EXECUTION_MODES = ["a", "b", "c", "d"] as const;
export type AppleExecutionMode = typeof APPLE_EXECUTION_MODES[number];

/** Preserve unsupported selections on read so configuration stays repairable. */
export function readAppleExecutionMode(value: unknown): string {
  const native = value && typeof value === "object" ? value as Record<string, unknown> : {};
  const explicit = String(native.executionMode ?? "").trim().toLowerCase();
  if (explicit) return explicit;
  if (native.mode === "ane" || native.mode === "auto"
    && (Boolean(native.decoderSegment) || Array.isArray(native.decoderBundles) && native.decoderBundles.length > 0)) return "e";
  return native.visionCompute === "gpu" ? "a" : "b";
}

export function appleExecutionModeIssue(value: unknown): string {
  const mode = readAppleExecutionMode(value);
  if (mode === "e") return "E 模式已退出产品；请明确选择 B（稳定）或 A/C/D（实验）。已有索引不会自动修改，旧 E 测试索引请在新的存储目录重新建立。";
  return APPLE_EXECUTION_MODES.includes(mode as AppleExecutionMode)
    ? "" : "Apple 计算模式必须是 A、B、C 或 D；请重新选择模式。";
}

export function requireAppleExecutionMode(value: unknown): AppleExecutionMode {
  const issue = appleExecutionModeIssue(value);
  if (issue) throw Object.assign(new Error(issue), { status: 400, code: "unsupported_apple_execution_mode" });
  return readAppleExecutionMode(value) as AppleExecutionMode;
}
