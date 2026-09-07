# Apple embedding backend third-party notices

The Indexed Apple embedding backend uses or derives from the following projects.
Their license texts remain authoritative and must accompany binary distributions.

| Component | Pinned source | License | Use |
|---|---|---|---|
| MLX Swift LM | `3.31.4` / `bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57` | MIT | Vendored runtime subset with the modifications listed below |
| MLX Swift | `0.31.4` / `dc43e62d7055353c7f99fa071a4e71d29dfddc44` | MIT | Swift tensor runtime |
| mlx-metal | `0.31.1`; platform-specific archive and payload hashes in `mlx-metal-artifacts.json` | MIT | Only `mlx.metallib` and the upstream MIT license are extracted; no Python wheel is installed |
| swift-transformers | `1.3.3` / `2fa33e1f5e7131a7fc64c28e6d161dcec0d24820` | Apache-2.0 | Tokenization and processor support |
| WeMM-Embedding-2B | Tencent public model distribution | Apache-2.0 with listed third-party terms | Separately downloaded model and converted Apple artifacts |
| oMLX | `0.6.4` / `1d7826185c5b5b69b38b27cbe57d7597b7551fd7` | Apache-2.0 | Private ANE bridge and hybrid MLP conventions adapted as described below. The Python C/D reference remains separate |

Indexed modifications to the vendored MLX Swift LM sources:

- `Libraries/MLXVLM/Models/Qwen35.swift`: embedding-only language execution,
  external image features and multimodal positions, decoder continuation and
  normalization, plus a layer-indexed optional embedding recurrence backend.
- `Libraries/MLXVLM/Models/Qwen3VL.swift`: video MRoPE positions for repeated
  timestamped frame blocks.
- `Libraries/MLXLMCommon/GatedDelta.swift`: an optional prefill backend after
  the original decay, beta and FP32 state preparation. An absent hook/result
  retains the existing GPU path; A/B do not install this hook.

The C/Objective-C bridge adapts oMLX's `qwen35_ane.mm` private runtime selectors,
weight blobs and IOSurface conventions through a narrow C ABI. Swift's hybrid
MLP adapts row-wise INT8 quantization, fixed-shape linear MIL, channel partitioning,
token tiling, tail padding and failure latching from `qwen35_ane_prefill.py`;
the FP32 SwiGLU merge follows `qwen35_ane.metal`. Indexed adds bounds checks,
shared-buffer lifetime management, cancellation, scheduling and diagnostics.
These experimental C/D paths require no Python interpreter. The classic variant
8 GPU tile uses the pinned MLX kernel implementation through MLXFast; the existing
MLX MIT license applies to that upstream implementation. Private Apple API use
remains experimental and is not part of default B inference.

`build_native_service.sh` copies the full resolved Swift dependency license texts
to `dist/apple-silicon/licenses/`. Converted WeMM packages copy the upstream
`LICENSE` and `README.md`, and the conversion pipeline adds `MODIFICATIONS.md`.

The Metal build uses Node.js and the system `unzip`, without Python or pip. The
archive is downloaded from the pinned PyPI file URL and checked against both its
size and SHA-256 before extraction; the library and license are checked again.
The macOS 26 library retains the previous payload SHA-256
`198488eb61359e953580a9c4530400feee1a06dd2f28a930a6ffa58aec66a597`.
Builds on macOS 14/15 select their respective upstream variants. Each binary
distribution includes `MLX_METAL_ARTIFACT.json` with the selected minimum macOS
version, source URL and hashes. This provenance is not a real-model performance
or cross-device compatibility certification.

Indexed is not affiliated with or endorsed by Apple, Tencent, Hugging Face, or
the MLX project. Product and project names are used only for attribution and
interoperability.
