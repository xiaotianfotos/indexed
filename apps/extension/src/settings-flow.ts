export type ExtensionBackendMode = "cloud" | "local";

export interface BackendModePresentation {
  mode: ExtensionBackendMode;
  help: string;
  selected: string;
}

export function backendModePresentation(mode: unknown): BackendModePresentation {
  if (mode === "local") {
    return {
      mode: "local",
      help: "仅使用本地 Indexed 服务；不需要填写阿里云或 Embedding 模型配置。",
      selected: "已选择本地服务；保存后插件只连接这个地址。",
    };
  }
  return {
    mode: "cloud",
    help: "仅使用阿里云直连；需要 Embedding 与 OSS Vector Bucket 配置，不需要本地 Indexed 服务。",
    selected: "已选择阿里云直连；保存后插件不会使用本地服务地址。",
  };
}

export async function saveBeforeConnectionTest(
  save: () => Promise<boolean>,
  testConnection: () => Promise<boolean>,
): Promise<{ saved: boolean; valid: boolean | null }> {
  const saved = await save();
  if (!saved) return { saved: false, valid: null };
  return { saved: true, valid: await testConnection() };
}
