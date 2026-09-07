#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
SWIFT_DIR="$SCRIPT_DIR/swift"
OUTPUT_DIR=${1:-"$SCRIPT_DIR/dist/apple-silicon"}

cd "$SWIFT_DIR"
swift build -c release --force-resolved-versions

METALLIB="$SWIFT_DIR/.build/release/mlx.metallib"
node "$SCRIPT_DIR/prepare-metallib.mjs" "$SWIFT_DIR/.build/release"

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=${OUTPUT_DIR:A}
install -m 0755 "$SWIFT_DIR/.build/release/indexed-apple-embedding" "$OUTPUT_DIR/"
install -m 0644 "$METALLIB" "$OUTPUT_DIR/"
install -m 0644 "$SCRIPT_DIR/THIRD_PARTY_NOTICES.md" "$OUTPUT_DIR/THIRD_PARTY_NOTICES.md"
mkdir -p "$OUTPUT_DIR/licenses"
install -m 0644 "$SWIFT_DIR/.build/release/mlx-metal-LICENSE.txt" "$OUTPUT_DIR/licenses/mlx-metal-MIT.txt"
install -m 0644 "$SWIFT_DIR/.build/release/MLX_METAL_ARTIFACT.json" "$OUTPUT_DIR/"
install -m 0644 "$SWIFT_DIR/Vendor/mlx-swift-lm/LICENSE" \
  "$OUTPUT_DIR/licenses/mlx-swift-lm-MIT.txt"
install -m 0644 "$SWIFT_DIR/Vendor/omlx-ane/LICENSE" \
  "$OUTPUT_DIR/licenses/omlx-ane-Apache-2.0.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/mlx-swift/LICENSE" \
  "$OUTPUT_DIR/licenses/mlx-swift-MIT.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/EventSource/LICENSE.md" \
  "$OUTPUT_DIR/licenses/EventSource.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/swift-asn1/LICENSE.txt" \
  "$OUTPUT_DIR/licenses/swift-asn1.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/swift-collections/LICENSE.txt" \
  "$OUTPUT_DIR/licenses/swift-collections.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/swift-crypto/LICENSE.txt" \
  "$OUTPUT_DIR/licenses/swift-crypto.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/swift-huggingface/LICENSE" \
  "$OUTPUT_DIR/licenses/swift-huggingface.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/swift-jinja/LICENSE" \
  "$OUTPUT_DIR/licenses/swift-jinja.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/swift-numerics/LICENSE.txt" \
  "$OUTPUT_DIR/licenses/swift-numerics.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/swift-syntax/LICENSE.txt" \
  "$OUTPUT_DIR/licenses/swift-syntax.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/swift-transformers/LICENSE" \
  "$OUTPUT_DIR/licenses/swift-transformers.txt"
install -m 0644 "$SWIFT_DIR/.build/checkouts/yyjson/LICENSE" \
  "$OUTPUT_DIR/licenses/yyjson.txt"

cd "$OUTPUT_DIR"
release_files=(
  indexed-apple-embedding
  mlx.metallib
  THIRD_PARTY_NOTICES.md
  MLX_METAL_ARTIFACT.json
  licenses/*(N.)
)
shasum -a 256 $release_files > SHA256SUMS
print "$OUTPUT_DIR"
