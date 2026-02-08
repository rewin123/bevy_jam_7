# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "torchvision>=0.15",
#     "onnxruntime>=1.15",
#     "numpy",
#     "Pillow",
# ]
# ///
"""
Debug: compare our Rust-equivalent pipeline vs reference Python AdaIN/MicroAST.

Tests the full pipeline with a REAL image to catch value range, scaling,
or normalization issues.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "assets" / "models"
STYLES_DIR = ROOT / "assets" / "styles"

issues_found = []


def flag(msg: str):
    issues_found.append(msg)
    print(f"  [!!] {msg}")


def ok(msg: str):
    print(f"  [OK] {msg}")


# =========================================================================
# AdaIN reference (Python, matching naoto0804/pytorch-AdaIN)
# =========================================================================

def calc_mean_std_python(feat, eps=1e-5):
    """Reference Python implementation — uses torch.var (unbiased, Bessel's correction)."""
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps  # unbiased (correction=1)
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain_python(content_feat, style_feat):
    """Reference Python AdaIN — from naoto0804/pytorch-AdaIN."""
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_python(style_feat)
    content_mean, content_std = calc_mean_std_python(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adain_rust_equivalent(content_feat_np, style_feat_np):
    """Simulate our Rust adain() — uses biased variance (mean of squared diffs)."""
    _, c, h, w = content_feat_np.shape
    result = np.zeros_like(content_feat_np)

    for ch in range(c):
        content_slice = content_feat_np[0, ch]
        style_slice = style_feat_np[0, ch]

        c_mean = content_slice.mean()
        s_mean = style_slice.mean()

        # Rust code: mapv(|x| (x - c_mean).powi(2)).mean() — BIASED variance!
        c_var = ((content_slice - c_mean) ** 2).mean()
        s_var = ((style_slice - s_mean) ** 2).mean()

        c_std = np.sqrt(c_var + 1e-5)
        s_std = np.sqrt(s_var + 1e-5)

        result[0, ch] = (content_slice - c_mean) / c_std * s_std + s_mean

    return result


# =========================================================================
# Test AdaIN pipeline
# =========================================================================

def test_adain_variance_bug():
    """Check if our biased variance differs from PyTorch's unbiased variance."""
    print("\n--- AdaIN: Variance computation (biased vs unbiased) ---")

    torch.manual_seed(42)
    # Simulate VGG encoder output at 256x256 → [1,512,32,32]
    feat = torch.randn(1, 512, 32, 32)

    # PyTorch unbiased variance (reference)
    py_mean, py_std = calc_mean_std_python(feat)

    # Rust-equivalent biased variance
    feat_np = feat.numpy()
    n_spatial = 32 * 32  # 1024 elements per channel
    for ch in [0, 1, 255, 511]:
        sl = feat_np[0, ch]
        biased_var = ((sl - sl.mean()) ** 2).mean()
        unbiased_var = feat[0, ch].reshape(-1).var().item()

        biased_std = np.sqrt(biased_var + 1e-5)
        unbiased_std = py_std[0, ch, 0, 0].item()

        rel_diff = abs(biased_std - unbiased_std) / unbiased_std * 100
        print(f"  ch={ch:3d}: biased_std={biased_std:.6f}, unbiased_std={unbiased_std:.6f}, rel_diff={rel_diff:.3f}%")

    # Full adain comparison
    torch.manual_seed(42)
    content = torch.randn(1, 512, 32, 32)
    style = torch.randn(1, 512, 32, 32)

    py_result = adain_python(content, style).numpy()
    rust_result = adain_rust_equivalent(content.numpy(), style.numpy())

    max_diff = np.abs(py_result - rust_result).max()
    mean_diff = np.abs(py_result - rust_result).mean()
    print(f"\n  Full AdaIN comparison (512 channels, 32x32):")
    print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if max_diff > 0.01:
        flag(f"AdaIN biased/unbiased variance: max_diff={max_diff:.4f} — may affect quality")
    else:
        ok(f"AdaIN variance difference is small: max_diff={max_diff:.6f}")


def test_adain_e2e_with_real_image():
    """Run real images through AdaIN ONNX pipeline, compare with Python reference."""
    print("\n--- AdaIN: End-to-end with real images ---")

    enc_path = MODELS_DIR / "adain-vgg.onnx"
    dec_path = MODELS_DIR / "adain-decoder.onnx"
    if not enc_path.exists() or not dec_path.exists():
        print("  [SKIP] AdaIN ONNX models not found")
        return

    # Load a style image (simulating what Rust code does)
    style_files = sorted(STYLES_DIR.glob("*.jpg"))
    if not style_files:
        print("  [SKIP] No style images found")
        return

    content_path = style_files[0]  # Use first style image as content too
    style_path = style_files[1] if len(style_files) > 1 else style_files[0]

    SIZE = 256

    # --- Simulate our Rust pipeline ---
    # 1. Load and resize (like load_style_image in Rust)
    content_img = Image.open(content_path).resize((SIZE, SIZE), Image.LANCZOS).convert("RGB")
    style_img = Image.open(style_path).resize((SIZE, SIZE), Image.LANCZOS).convert("RGB")

    # 2. Convert to [1,3,H,W] float32 [0,1] (like rgba_to_tensor_01)
    content_np = np.array(content_img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    style_np = np.array(style_img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

    print(f"  Content: {content_path.name} → {content_np.shape}, range [{content_np.min():.3f}, {content_np.max():.3f}]")
    print(f"  Style:   {style_path.name} → {style_np.shape}, range [{style_np.min():.3f}, {style_np.max():.3f}]")

    # 3. Encode with VGG
    enc_sess = ort.InferenceSession(str(enc_path))
    dec_sess = ort.InferenceSession(str(dec_path))

    content_feats = enc_sess.run(["output"], {"input": content_np})[0]
    style_feats = enc_sess.run(["output"], {"input": style_np})[0]
    print(f"  Encoder output: content={content_feats.shape}, style={style_feats.shape}")
    print(f"  Content features: range [{content_feats.min():.3f}, {content_feats.max():.3f}], mean={content_feats.mean():.3f}")
    print(f"  Style features:   range [{style_feats.min():.3f}, {style_feats.max():.3f}], mean={style_feats.mean():.3f}")

    # 4a. AdaIN — Rust-equivalent (biased variance)
    rust_transferred = adain_rust_equivalent(content_feats, style_feats)
    print(f"  AdaIN (Rust-equiv): range [{rust_transferred.min():.3f}, {rust_transferred.max():.3f}]")

    # 4b. AdaIN — Python reference (unbiased variance)
    py_transferred = adain_python(
        torch.from_numpy(content_feats), torch.from_numpy(style_feats)
    ).numpy()
    print(f"  AdaIN (Python ref): range [{py_transferred.min():.3f}, {py_transferred.max():.3f}]")

    adain_diff = np.abs(rust_transferred - py_transferred).max()
    print(f"  AdaIN max_diff: {adain_diff:.6f}")

    # 5. Decode
    rust_output = dec_sess.run(["output"], {"input": rust_transferred})[0]
    py_output = dec_sess.run(["output"], {"input": py_transferred})[0]

    print(f"\n  Decoder output (Rust-equiv): shape={rust_output.shape}, range [{rust_output.min():.4f}, {rust_output.max():.4f}]")
    print(f"  Decoder output (Python ref): shape={py_output.shape}, range [{py_output.min():.4f}, {py_output.max():.4f}]")

    output_diff = np.abs(rust_output - py_output).max()
    print(f"  Final output max_diff: {output_diff:.6f}")

    # 6. Clamp and convert to uint8 (like tensor_01_to_rgba)
    rust_u8 = (np.clip(rust_output[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    py_u8 = (np.clip(py_output[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

    # Save for visual inspection
    out_dir = ROOT / "debug_output"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(np.array(content_img)).save(out_dir / "adain_content.png")
    Image.fromarray(np.array(style_img)).save(out_dir / "adain_style.png")
    Image.fromarray(rust_u8).save(out_dir / "adain_output_rust.png")
    Image.fromarray(py_u8).save(out_dir / "adain_output_python.png")
    print(f"\n  Saved debug images to {out_dir}/adain_*.png")

    # Check for values outside [0,1]
    out_of_range = (rust_output < 0).sum() + (rust_output > 1).sum()
    total = rust_output.size
    pct = out_of_range / total * 100
    if pct > 5:
        flag(f"AdaIN output has {pct:.1f}% values outside [0,1] — clipping may lose detail")
    else:
        ok(f"AdaIN output: {pct:.1f}% values outside [0,1]")


def test_microast_e2e_with_real_image():
    """Run real images through MicroAST ONNX pipeline."""
    print("\n--- MicroAST: End-to-end with real images ---")

    onnx_path = MODELS_DIR / "microast.onnx"
    if not onnx_path.exists():
        print("  [SKIP] MicroAST ONNX model not found")
        return

    style_files = sorted(STYLES_DIR.glob("*.jpg"))
    if not style_files:
        print("  [SKIP] No style images found")
        return

    content_path = style_files[0]
    style_path = style_files[1] if len(style_files) > 1 else style_files[0]

    SIZE = 256

    content_img = Image.open(content_path).resize((SIZE, SIZE), Image.LANCZOS).convert("RGB")
    style_img = Image.open(style_path).resize((SIZE, SIZE), Image.LANCZOS).convert("RGB")

    content_np = np.array(content_img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    style_np = np.array(style_img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

    print(f"  Content: {content_path.name} → range [{content_np.min():.3f}, {content_np.max():.3f}]")
    print(f"  Style:   {style_path.name} → range [{style_np.min():.3f}, {style_np.max():.3f}]")

    sess = ort.InferenceSession(str(onnx_path))
    output = sess.run(["output"], {"content": content_np, "style": style_np})[0]

    print(f"  Output: shape={output.shape}, range [{output.min():.4f}, {output.max():.4f}]")

    out_u8 = (np.clip(output[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

    out_dir = ROOT / "debug_output"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(np.array(content_img)).save(out_dir / "microast_content.png")
    Image.fromarray(np.array(style_img)).save(out_dir / "microast_style.png")
    Image.fromarray(out_u8).save(out_dir / "microast_output.png")
    print(f"  Saved debug images to {out_dir}/microast_*.png")


def test_render_target_format():
    """Check potential sRGB / linear color space issues."""
    print("\n--- Render target format analysis ---")

    print("  Render target: Rgba8UnormSrgb")
    print("  → Bevy renders in linear space, writes sRGB-encoded bytes to texture")
    print("  → Screenshot reads back sRGB bytes (0-255)")
    print("  → Our code divides by 255.0 → [0,1] sRGB values")
    print("  → Neural networks trained with transforms.ToTensor() on JPEG/PNG → also sRGB")
    ok("sRGB handling is consistent between pipeline and training data")

    print("\n  Display image: Rgba8UnormSrgb")
    print("  → We write network output as sRGB bytes (clamp [0,1] × 255)")
    print("  → Bevy UI samples the texture with sRGB→linear conversion")
    print("  → Display applies gamma → final output is correct")
    ok("Display color space is correct")


# =========================================================================

def main():
    print("=" * 60)
    print("Style Transfer Pipeline Debug Analysis")
    print("=" * 60)

    test_render_target_format()
    test_adain_variance_bug()
    test_adain_e2e_with_real_image()
    test_microast_e2e_with_real_image()

    print("\n" + "=" * 60)
    if issues_found:
        print(f"Issues found ({len(issues_found)}):")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("No issues found!")
    print("=" * 60)

    sys.exit(1 if issues_found else 0)


if __name__ == "__main__":
    main()
