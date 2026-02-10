"""Verify Bevy ONNX inference matches Python inference.

Loads the raw input frame saved by Bevy (--test-inference mode),
runs it through the same ONNX model in Python, and compares the
result with Bevy's styled output pixel-by-pixel.

Usage:
    # 1. Run Bevy to capture test frames:
    cargo run --release -- --test-inference

    # 2. Verify:
    cd training && uv run python verify_bevy_inference.py

    # Or with custom paths:
    uv run python verify_bevy_inference.py \
        --input ../test_input.png \
        --bevy-output ../test_bevy_output.png \
        --model ../assets/models/styles/manga.onnx
"""

import argparse
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Verify Bevy vs Python ONNX inference")
    parser.add_argument(
        "--input", default="../test_input.png", help="Raw input frame from Bevy"
    )
    parser.add_argument(
        "--bevy-output",
        default="../test_bevy_output.png",
        help="Styled output frame from Bevy",
    )
    parser.add_argument(
        "--model",
        default="../assets/models/styles/manga.onnx",
        help="ONNX model file",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Max allowed pixel diff (0-255 scale)",
    )
    parser.add_argument(
        "--save-diff", action="store_true", help="Save difference visualization"
    )
    args = parser.parse_args()

    # Load input frame (RGBA PNG -> RGB float32 [0,1] -> NCHW)
    input_img = Image.open(args.input).convert("RGB")
    input_np = np.array(input_img).astype(np.float32) / 255.0
    h, w, _ = input_np.shape
    input_tensor = input_np.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, H, W]

    print(f"Input:  {args.input} ({w}x{h})")
    print(f"Model:  {args.model}")

    # Run ONNX inference in Python
    sess = ort.InferenceSession(args.model)
    python_output = sess.run(["output"], {"input": input_tensor})[0]

    # Convert Python output to uint8 image (same pipeline as Bevy's tensor_to_rgba)
    python_rgb = (
        python_output[0].transpose(1, 2, 0).clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)

    # Load Bevy output
    bevy_img = Image.open(args.bevy_output).convert("RGB")
    bevy_rgb = np.array(bevy_img)

    print(f"Bevy:   {args.bevy_output} ({bevy_img.width}x{bevy_img.height})")
    print(f"Python output shape: {python_rgb.shape}, Bevy output shape: {bevy_rgb.shape}")

    # Resize if shapes differ (shouldn't happen with same model)
    if python_rgb.shape != bevy_rgb.shape:
        print(f"WARNING: shape mismatch, resizing Python output to match Bevy")
        python_pil = Image.fromarray(python_rgb).resize(
            (bevy_rgb.shape[1], bevy_rgb.shape[0]), Image.LANCZOS
        )
        python_rgb = np.array(python_pil)

    # Compare
    diff = np.abs(python_rgb.astype(np.float32) - bevy_rgb.astype(np.float32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    p99_diff = np.percentile(diff, 99)

    print(f"\n--- Results ---")
    print(f"Max pixel diff:  {max_diff:.1f}")
    print(f"Mean pixel diff: {mean_diff:.4f}")
    print(f"P99 pixel diff:  {p99_diff:.1f}")
    print(f"Tolerance:       {args.tolerance}")

    if args.save_diff:
        # Amplify diff for visibility (scale to 0-255)
        if max_diff > 0:
            diff_vis = (diff / max_diff * 255).astype(np.uint8)
        else:
            diff_vis = np.zeros_like(python_rgb)
        Image.fromarray(diff_vis).save("test_diff.png")
        Image.fromarray(python_rgb).save("test_python_output.png")
        print(f"Saved: test_diff.png, test_python_output.png")

    if max_diff <= args.tolerance:
        print(f"\nPASSED")
        return 0
    else:
        print(f"\nFAILED (max_diff {max_diff:.1f} > tolerance {args.tolerance})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
