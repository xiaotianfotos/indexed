export interface SetupState {
  phase: "checking" | "offline" | "configure" | "loading" | "error" | "ready";
  detail: string;
  directories: number;
  indexed: number;
  scanning: boolean;
}

export function setupPresentation(state: SetupState) {
  if (state.phase === "offline") return { title: "无法连接本机服务", detail: state.detail,
    action: "retry", label: "重新连接", searchable: false };
  if (state.phase === "configure") return { title: "先设置检索模型", detail: state.detail,
    action: "engine", label: "设置模型", searchable: false };
  if (state.phase === "loading") return { title: "模型正在准备", detail: state.detail,
    action: "engine", label: "查看模型状态", searchable: false };
  if (state.phase === "error") return { title: "连接检查未通过", detail: state.detail,
    action: "engine", label: "检查配置", searchable: false };
  if (state.phase === "checking") return { title: "正在检查模型与索引", detail: "本机配置、模型和素材状态将显示在这里。",
    action: "retry", label: "重新检查", searchable: false };
  if (state.scanning) return { title: "正在建立索引", detail: state.indexed > 0 ? "已完成的素材可以先检索，扫描会继续进行。" : "首批素材正在处理，完成后即可检索。",
    action: "assets", label: "查看进度", searchable: state.indexed > 0 };
  if (state.indexed > 0) return { title: "可以开始检索了", detail: "输入描述，查找已保存的素材。",
    action: "", label: "", searchable: true };
  return state.directories > 0
    ? { title: "开始第一轮索引", detail: "已添加素材目录。启动扫描后即可生成可检索内容。", action: "assets", label: "查看目录并扫描", searchable: false }
    : { title: "添加第一个素材目录", detail: "选择要检索的图片、视频或文档目录。只会处理你选择的目录。", action: "add", label: "添加目录", searchable: false };
}
