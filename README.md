# Indexed

[中文](#中文) · [English](#english)

## 中文

**看过一次，就应该能再次找到。**

我们每天在 YouTube、B站和各种视频资料里看到大量有价值的画面与知识，但传统收藏只能记住链接，关键词搜索也只能找到标题或字幕。真正重要的界面演示、视觉案例和某个具体片段，往往很难再次找回，更无法直接成为 AI 的长期上下文。

Indexed 把用户主动选择的视频转化为可搜索、可管理、也可供 AI 调用的多模态记忆。画面与字幕分别建立向量索引，同时保留来源、时间码和视频关系，让人和 AI 都能用自然语言回到原视频的准确位置。

## 架构

```text
YouTube / B站
      │ 用户主动选择
      ▼
Chrome Extension
      ├── 视频片段 ──► Multimodal Embedding ──► Visual Index
      └── 同期字幕 ──► Text Embedding ─────────► Transcript Index
                                                   │
                              ┌────────────────────┴────────────────────┐
                              ▼                                         ▼
                       Local Dashboard                           CLI + AI Skill
                         面向人类                                  面向 Agent
```

- **Chrome Extension**：在播放器内采集用户选择的视频，建立画面和字幕索引。
- **Local Dashboard**：配置模型与存储，搜索、查看和管理视频记忆。
- **CLI + AI Skill**：让 AI Agent 通过结构化接口完成查询与增删改查。
- **Model Profiles**：隔离不同 Embedding 模型的向量空间，避免不兼容的索引被混合查询。

Indexed 采用本地优先、组件解耦的设计。插件、Dashboard 和 AI Skill 可以共享同一套索引，但彼此不形成强制依赖；模型计算与向量存储也可以独立替换。

## 当前状态

项目正在继续优化产品体验、配置流程与检索质量。架构和功能稳定后，会补充完整代码、Quick Start、部署说明和演示素材。

---

## English

**If you have watched it once, you should be able to find it again.**

Every day, valuable ideas, interfaces, demonstrations, and visual references disappear into YouTube playlists and browser history. Traditional bookmarks remember links, while keyword search usually sees only titles or transcripts. The exact visual moment we need remains difficult to recover—and unavailable as long-term context for AI.

Indexed turns videos explicitly selected by the user into searchable multimodal memory. Video segments and their corresponding transcripts are embedded into separate indexes while retaining source links, timestamps, and video relationships. Both people and AI agents can use natural language to return to the relevant moment in the original video.

### Architecture

```text
YouTube / Bilibili
      │ selected by the user
      ▼
Chrome Extension
      ├── video segments ──► Multimodal Embedding ──► Visual Index
      └── transcripts ─────► Text Embedding ─────────► Transcript Index
                                                         │
                                  ┌──────────────────────┴──────────────────────┐
                                  ▼                                             ▼
                           Local Dashboard                               CLI + AI Skill
                              for people                                   for agents
```

- **Chrome Extension:** captures user-selected videos and indexes their visual content and transcripts.
- **Local Dashboard:** configures models and storage, then searches and manages video memories.
- **CLI + AI Skill:** gives AI agents structured search and CRUD operations.
- **Model Profiles:** isolate embedding spaces so incompatible vectors are never queried together.

Indexed is local-first and loosely coupled. The extension, Dashboard, and AI Skill can share the same indexes without becoming hard dependencies, while embedding models and vector storage remain independently replaceable.

### Status

The product experience, configuration flow, and retrieval quality are still being refined. The full source code, Quick Start, deployment guide, and demos will be published once the architecture and core behavior are ready.
