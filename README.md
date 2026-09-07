# Indexed

**看过一次，就应该能再次找到。**

Indexed 把主动添加的本机图片、视频、文档，以及在 YouTube / Bilibili 主动采集的视频片段，变成可搜索的多模态资料库。用自然语言找画面、片段和文字，结果保留文件位置、原始链接、时间码与文字摘录，方便人和 AI Agent 回到来源。

当前为测试阶段，主要面向本机使用。浏览器插件提供预发布 ZIP；完整应用尚未提供签名、公证和自动升级的正式安装包。

## 能做什么

- **本机资料检索**：登记素材目录，索引图片、视频、纯文本、带文本层的 PDF 和 Word 文档。扫描版 PDF 暂不提供 OCR。
- **网页视频记忆**：Chrome 扩展在用户主动启用后采集视频画面及同期字幕；不作为视频下载器。
- **Dashboard、CLI 与 AI Skill**：共用应用服务和索引，让人和 Agent 检索同一份资料。
- **本地向量存储**：使用 zvec，按模型与向量空间隔离索引。
- **Apple 原生 embedding**：Swift helper 结合 Core ML 与 MLX，产品构建、模型准备和推理无需 Python、pip 或虚拟环境。

## 从源码启动

获取源码并启动：

```bash
git clone https://github.com/xiaotianfotos/indexed.git
cd indexed
npm ci
npm run build
./bin/indexed serve --open
```

以上为源码运行方式，使用 Node.js 20 或更新版本。

Dashboard 默认地址为 `http://127.0.0.1:18767`。首次启动后，先在“模型与数据库”完成 embedding 配置，再添加需要索引的素材目录。添加目录会开始扫描，可在“素材与扫描”中停止或关闭自动扫描。

可以使用下方的 Apple 原生后端，也可以配置外部 embedding 服务；后者会把处理的内容发送给你选择的服务。参考配置见 [config/indexed.example.json](config/indexed.example.json)。真实密钥只保存在本机配置中。

使用 `INDEXED_HOME` 可将配置、模型、数据和缓存放入指定的独立目录；请在配置与启动命令中保持一致。避免多个进程同时扫描同一个 zvec 目录。

## Apple Silicon 本地模型

当前真实模型验证设备为基础款 Apple M4 / macOS 26。其他芯片与系统版本尚需单独验证，尤其是编译后的 Core ML 模型与私有 ANE 路径。

在 Indexed 源码目录中构建原生 helper：

```bash
zsh native/apple-embedding/build_native_service.sh
```

模型与源码分开分发。使用 Indexed 专用的 **WeMM-Embedding-2B-Apple-Q8-G64** 模型包，约 2.50 GiB，包含 MLX Q8/G64 语言权重、Core ML 视觉编码器、tokenizer、模板及校验清单。这是基于 Tencent WeMM-Embedding-2B 的混合格式包，不能直接当作通用 mlx-lm 或 Transformers 模型使用。

模型已托管在 [魔搭 / ModelScope](https://modelscope.cn/models/xiaotianfotos/WeMM-Embedding-2B-Apple-Q8-G64)。下载完整文件并保留 `language/` 和 `vision/` 层级；校验清单也覆盖魔搭仓库的 `.gitattributes` 与 `configuration.json`。如通过 Git 获取，需取得实际 LFS 权重，并将文件导出到不含 `.git/` 的模型目录再校验。应用尚未实现自动下载。将下面的 `/path/to/model-package` 替换成该目录的绝对路径：

```bash
node scripts/prepare-apple-model-package.mjs --verify /path/to/model-package
./bin/indexed embedding configure \
  --binary "$PWD/native/apple-embedding/dist/apple-silicon/indexed-apple-embedding" \
  --model-package /path/to/model-package \
  --execution-mode B \
  --dimension 2048
./bin/indexed embedding validate --full
./bin/indexed embedding prepare
./bin/indexed serve --open
```

完整分发文件校验是上面的独立步骤；原生安装器另外校验 manifest 中的语言与视觉模型哈希。

| 模式 | 执行方式 | 定位 |
| --- | --- | --- |
| A | Core ML GPU 视觉 + MLX GPU 语言 | 性能基线 |
| B | 公开 Core ML CPU/ANE 视觉 + MLX GPU 语言 | 默认模式 |
| C | 在 B 的基础上，将部分语言 MLP 交给私有 ANE 路径 | 显式实验模式 |
| D | 在 C 的基础上，使用固定质量档案接管部分 GDN 计算 | 显式实验模式 |

A/B/C/D 共用一个模型包，均使用 Swift helper。Dashboard 的 A/C/D 入口需要开启实验选项；C/D 使用私有 Apple API，不能视为公开 API 支持承诺。实际设备路由与回退以诊断信息为准。

## Chrome 扩展与 Agent

从 [Releases](https://github.com/xiaotianfotos/indexed/releases) 下载 `indexed-extension-版本号.zip`，解压到一个固定目录。插件安装无需 Node.js 或自行编译：

1. 打开 Chrome 扩展管理页 `chrome://extensions`，开启“开发者模式”。
2. 点击“加载已解压的扩展程序”，选择解压后包含 `manifest.json` 的目录。
3. 在扩展设置中连接本机 Indexed 服务，在需要的视频上主动开启采集。

插件包不包含本机服务器、Apple 原生后端或模型；本机模式仍需按上方说明启动 Indexed。插件尚未上架 Chrome 商店，当前没有自动更新：将新版解压到原安装目录后，在扩展管理页点击重新加载。保留该目录及扩展的浏览器存储。

如需从源码构建，执行 `npm ci` 和 `npm run build` 后加载 `dist/extension/`。发行附件中的 `SHA256SUMS` 用于校验 ZIP。

CLI 示例：

```bash
./bin/indexed status
./bin/indexed assets libraries
./bin/indexed assets search "展示时间线编辑器的片段"
```

AI Agent 接口见 [Indexed Skill](skills/indexed/SKILL.md)。Skill 通过 CLI 调用应用服务，不承担应用运行时职责。

## 开发与数据边界

<details>
<summary>源码环境与按需依赖</summary>

- 源码运行使用 Node.js 20+，当前 GitHub CI 使用 Node.js 24。
- 视频处理使用 FFmpeg；带文本层的 PDF 提取使用 Poppler 的 `pdftotext`。这两个工具目前需要在本机另行准备，仅在使用对应功能时需要，纯文本检索无需安装。
- Apple 原生 helper 的源码构建需要支持 Swift 6.3 的 Xcode / Command Line Tools。
- 当前源码预览尚未提供统一打包这些依赖的安装程序。

</details>

应用代码使用 TypeScript；macOS helper 使用 Swift 与少量原生桥接。核心业务位于 `packages/core`，共享协议位于 `packages/contracts`，界面与 CLI 位于 `apps/`。

```bash
npm run check
npx playwright install chromium
npm run test:browser
```

修改原生 helper 后另运行 `npm run test:native`。默认原生协议测试使用合成输入，并不代表真实模型跨设备验收。当前 GitHub CI 执行 Ubuntu / Node.js 24 应用和浏览器检查；正式发行与完整硬件支持矩阵仍待完善。

维护者可运行 `npm run release:extension` 生成插件 ZIP 与校验文件。发布时，标签 `extension-v版本号` 必须匹配插件 manifest 的版本，并指向 `main` 已包含的提交。推送该标签会在检查通过后发布预发布版本；手动运行 Release 工作流并指定已有标签则创建草稿。流程不会覆盖已有 Release。

服务仅供本机使用。项目不会自动采集无关浏览内容；本地模式的索引和素材留在本机，外部模型或云存储的数据流取决于你启用的配置。

测试阶段不提供旧数据库迁移工具。遇到不兼容索引，请选择新的空存储目录并重新扫描；程序不会自动删除或改写旧目录。不同模型的 `embedding_space` 不能混用。

内部设计文档、基准报告和本地配置不随源码发布。根目录 README、许可证、第三方声明与功能性 Skill 保持公开。

历史 Python 服务、模型转换及离线参考工具已退出当前公开源码；现有模型包可直接由 Swift 后端加载。

## English

Indexed is a local-first multimodal search application for explicitly added images, videos, documents, and user-selected YouTube / Bilibili video captures. Results retain source paths, URLs, timestamps, and text passages for both people and AI agents.

This is an early source preview for local-machine use. Build with Node.js 20+ using `npm ci && npm run build`, then run `./bin/indexed serve --open`. Configure an embedding backend before scanning. Local vector storage uses zvec; external embedding services receive the content you choose to process.

The browser extension is also available as a preview ZIP on [Releases](https://github.com/xiaotianfotos/indexed/releases). Extract it, enable Developer mode at `chrome://extensions`, and load the directory containing `manifest.json`. No build tools are needed to install the ZIP. The local server and Apple models are separate; extension updates currently require replacing the extracted files and reloading the extension.

The Apple Silicon backend uses Swift, Core ML, and MLX without a Python product dependency. B is the default; A is a GPU baseline and C/D are explicit private-ANE experiments. Real-model validation currently covers base M4 on macOS 26. The dedicated approximately 2.50 GiB WeMM model package is available separately on [ModelScope](https://modelscope.cn/models/xiaotianfotos/WeMM-Embedding-2B-Apple-Q8-G64). Preserve its directory layout, download actual LFS weights, and export files without Git metadata before verification. Signed releases, automatic model downloads, and broader hardware validation are not yet available.

## 许可证 / License

Indexed 源码采用 [Apache-2.0](LICENSE)。模型权重、第三方代码与原生依赖保留各自许可证；请同时阅读 [NOTICE](NOTICE) 和 [原生依赖声明](native/apple-embedding/THIRD_PARTY_NOTICES.md)。

This project is a community implementation, not an official Tencent or Apple release.
