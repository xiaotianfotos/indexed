# Indexed Chrome Extension: Local Installation

[简体中文](./extension-installation.zh-CN.md)

Indexed is currently distributed as a Chrome Manifest V3 extension. It processes a YouTube or Bilibili video only after you explicitly select that video from the player controls.

## Requirements

- Chrome 120 or later
- Node.js 20 or later
- npm 10 or later

## 1. Clone and build

```bash
git clone https://github.com/xiaotianfotos/indexed.git
cd indexed
npm install
npm run build
```

The unpacked Chrome extension is generated at:

```text
dist/extension
```

Do not load `apps/extension` directly. It contains the uncompiled TypeScript source.

## 2. Load the extension in Chrome

1. Open `chrome://extensions/` in Chrome.
2. Enable **Developer mode**.
3. Select **Load unpacked**.
4. Choose the repository's `dist/extension` directory.
5. Pin Indexed to the Chrome toolbar for easier access.

## 3. Choose a runtime mode

Open the Indexed popup and expand Settings. The two modes are mutually exclusive.

### Direct cloud mode

This mode does not require an Indexed application backend. The extension calls your multimodal Embedding service directly and writes the resulting vectors to Alibaba Cloud OSS Vectors.

Enter the following values:

- Embedding service URL, for example `http://127.0.0.1:5007`
- Model name and output dimension; the dimension must match the indexes
- Alibaba Cloud Region, account ID, and Vector Bucket name
- Visual Index and Transcript Index names
- A least-privilege RAM AccessKey ID and Secret, plus an optional STS token

Enter the service root URL without appending `/v1/embeddings`. On the first connection, Chrome asks for permission to access the configured model origin and OSS Vectors endpoint.

The AccessKey ID, Secret, and STS token remain on the current device. Non-secret settings such as the model endpoint, model name, and OSS resource names use Chrome Sync.

### Local mode

The extension UI already reserves a local mode for an Indexed Local Server that owns model configuration and vector storage. The first public release focuses on the direct-cloud extension. Local Server, LanceDB, and Dashboard will be added in a later release; select direct cloud mode unless you already run a compatible service.

## 4. Remember a video

1. Open a standard YouTube or Bilibili video page.
2. Find the small Indexed icon in the player control bar.
3. Click it to remember the current video; click it again to stop.
4. Indexed processes only the ten-second segments that actually play while you watch.
5. Open the memory library from the popup to search with text or an image, or to delete a video's indexes.

Visual segments, transcripts, and search inputs are sent to the Embedding service you configure. In direct cloud mode, Indexed also writes vectors, source information, timestamps, transcript text, and optional previews to your OSS Vector Bucket.

## 5. Update the extension

```bash
git pull
npm install
npm run build
```

Return to `chrome://extensions/` and click **Reload** on the Indexed card. Rebuilding replaces `dist/extension`; it does not intentionally remove extension settings or the local processing queue stored by Chrome.

## Troubleshooting

### The Indexed icon is missing from the player

Make sure the current page is a standard `youtube.com/watch` or `bilibili.com/video` page, then refresh it. Pages that were already open when the extension was reloaded also need one refresh.

### The Embedding service appears offline

Confirm that the service is reachable from the computer running Chrome, then check the endpoint, model name, and output dimension. Chrome may request origin access again when you use a LAN HTTP endpoint.

### Vector writes fail

Confirm that both indexes already exist, their dimensions match the model output, and the RAM user has the required query, write, and delete permissions for the target Bucket and indexes.
