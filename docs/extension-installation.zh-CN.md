# Indexed Chrome 插件：本地安装

[English](./extension-installation.md)

Indexed 目前以 Chrome Manifest V3 插件提供。插件只会处理你在 YouTube 或 B站播放器中主动选择的视频。

## 环境要求

- Chrome 120 或更高版本
- Node.js 20 或更高版本
- npm 10 或更高版本

## 1. 获取并构建

```bash
git clone https://github.com/xiaotianfotos/indexed.git
cd indexed
npm install
npm run build
```

构建完成后，Chrome 插件位于：

```text
dist/extension
```

不要直接加载 `apps/extension`，其中是尚未编译的 TypeScript 源码。

## 2. 加载到 Chrome

1. 在 Chrome 打开 `chrome://extensions/`。
2. 打开右上角的「开发者模式」。
3. 点击「加载已解压的扩展程序」。
4. 选择项目中的 `dist/extension` 目录。
5. 建议把 Indexed 固定到浏览器工具栏。

## 3. 选择运行方式

点击 Indexed 图标并展开设置。插件支持两种互斥模式。

### 云端直连

该模式不需要 Indexed 业务后端。插件直接调用你配置的多模态 Embedding 服务，并把向量写入阿里云 OSS Vectors。

需要填写：

- Embedding 服务地址，例如 `http://127.0.0.1:5007`
- 模型名和输出维度；必须与创建索引时的维度一致
- 阿里云 Region、账号 ID 和 Vector Bucket 名称
- 画面 Index 和字幕 Index
- 具有所需最小权限的 RAM AccessKey ID、Secret，以及可选的 STS Token

Embedding 地址填写服务根地址，不要重复添加 `/v1/embeddings`。首次连接时，Chrome 会请求访问模型服务和 OSS Vectors Endpoint 的权限。

AccessKey ID、Secret 和 STS Token 只保存在当前设备；模型地址、模型名和 OSS 资源名称等非密钥配置使用 Chrome Sync。

### 本地模式

插件界面已经预留本地模式，用于连接包含模型配置和向量数据库的 Indexed Local Server。首个公开版本优先发布云端直连插件，本地 Server、LanceDB 和 Dashboard 将在后续版本中补充。没有兼容服务时请选择云端直连。

## 4. 记录一个视频

1. 打开一个 YouTube 或 B站标准视频播放页。
2. 在播放器控制栏找到 Indexed 小图标。
3. 点击图标开始记忆当前视频；再次点击即可停止。
4. 插件只处理实际播放并观看过的十秒片段。
5. 点击插件中的「打开记忆库」进行文字搜索、图片搜索或删除视频索引。

画面、字幕和查询内容会发送到你配置的 Embedding 服务。云端直连模式还会把向量、来源、时间码、字幕文本和可选预览写入你的 OSS Vector Bucket。

## 5. 更新插件

```bash
git pull
npm install
npm run build
```

回到 `chrome://extensions/`，在 Indexed 卡片上点击「重新加载」。重新构建会覆盖 `dist/extension`，不会主动删除 Chrome 中已经保存的插件配置和本地队列。

## 常见问题

### 播放器里没有 Indexed 图标

确认当前是 `youtube.com/watch` 或 `bilibili.com/video` 标准播放页，然后刷新页面。插件重新加载后，已经打开的视频页面也需要刷新一次。

### 显示模型服务离线

确认模型服务可以从运行 Chrome 的电脑访问，并检查地址、模型名和输出维度。使用局域网 HTTP 服务时，Chrome 可能再次弹出访问该地址的授权请求。

### 向量写入失败

确认两个 Index 已存在、维度与模型输出一致，并检查 RAM 用户是否拥有对应 Bucket 和 Index 的查询、写入及删除权限。
