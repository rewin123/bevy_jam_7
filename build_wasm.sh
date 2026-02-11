#!/bin/bash
# Build Fever Dream for WASM
# Usage: ./build_wasm.sh [--release]
#
# Workaround: wasm-bindgen 0.2.108 segfaults due to deep recursion in walrus/rayon.
# Fix: RAYON_NUM_THREADS=1 + large stack.

set -euo pipefail

PROFILE="release"
CARGO_PROFILE_FLAG="--release"
OUT_DIR="web"

echo "=== Building Fever Dream for WASM ==="

# Step 1: cargo build
echo "[1/3] cargo build --target wasm32-unknown-unknown $CARGO_PROFILE_FLAG ..."
cargo build --target wasm32-unknown-unknown \
    --no-default-features --features burn-backend \
    $CARGO_PROFILE_FLAG

WASM_FILE="target/wasm32-unknown-unknown/$PROFILE/fever_dream.wasm"

# Step 2: wasm-bindgen (with rayon workaround)
echo "[2/3] wasm-bindgen ..."
mkdir -p "$OUT_DIR"
RAYON_NUM_THREADS=1 RUST_MIN_STACK=67108864 \
    wasm-bindgen --target web --out-dir "$OUT_DIR" "$WASM_FILE"

# Step 3: Copy assets
echo "[3/3] Copying assets ..."
rm -rf "$OUT_DIR/assets"
cp -r assets "$OUT_DIR/assets"

# Copy index.html if not present
if [ ! -f "$OUT_DIR/index.html" ]; then
    cp web_index.html "$OUT_DIR/index.html"
fi

WASM_SIZE=$(du -sh "$OUT_DIR/fever_dream_bg.wasm" | cut -f1)
echo ""
echo "=== Done! WASM size: $WASM_SIZE ==="
echo "Serve with: python3 -m http.server 8080 --directory $OUT_DIR"
