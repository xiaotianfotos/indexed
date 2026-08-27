# Indexed Chrome extension

This Manifest V3 extension captures fixed ten-second video segments and aligned subtitles only after the user selects the current YouTube or Bilibili video from the player control bar.

It directly calls an OpenAI-compatible multimodal Embedding endpoint and Alibaba Cloud OSS Vectors. No Indexed business backend is required for capture or search.

## Load for development

1. Open `chrome://extensions/`.
2. Enable Developer mode.
3. Choose **Load unpacked** and select this directory.
4. Configure the Embedding model, vector dimension, OSS Vector Bucket and two Indexes in the popup.

Non-secret settings use Chrome Sync. AccessKey ID, AccessKey Secret, STS Token, queues and previews remain device-local.

Each record stores `embedding_model` and `embedding_space`. Queries filter the active model so equal-dimensional vectors from unrelated models are never compared.

See the repository root [README](../README.md) for architecture, RAM permissions, the Local Dashboard and Agent Skill.
