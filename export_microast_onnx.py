# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "torchvision>=0.15",
#     "onnx>=1.14",
#     "onnxruntime>=1.15",
#     "onnxscript",
#     "numpy",
#     "Pillow",
# ]
# ///
"""
Export MicroAST style transfer model to ONNX.

Usage:
    uv run export_microast_onnx.py

This script:
1. Clones the MicroAST repo (if not already present)
2. Loads the 4 pretrained checkpoints (content_encoder, style_encoder, modulator, decoder)
3. Wraps them in a single nn.Module with alpha=1.0 baked in
4. Exports to ONNX with 2 inputs: content [1,3,H,W] and style [1,3,H,W]
5. Verifies the exported model with onnxruntime

Input value range: [0, 1] (from torchvision.transforms.ToTensor())
Output value range: clamped to [0, 1]
"""

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Step 1: Clone MicroAST repo and get the source + weights
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).parent / "MicroAST"

def clone_repo():
    if REPO_DIR.exists() and (REPO_DIR / "net_microAST.py").exists():
        print(f"[OK] MicroAST repo already present at {REPO_DIR}")
        return
    print("[...] Cloning MicroAST repository...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "https://github.com/EndyWon/MicroAST.git", str(REPO_DIR)]
    )
    print(f"[OK] Cloned to {REPO_DIR}")


# ---------------------------------------------------------------------------
# Step 2: Inline the model definitions
# (We inline them to avoid sys.path hacks and import issues)
# ---------------------------------------------------------------------------

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


featMod = adaptive_instance_normalization

slim_factor = 1


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groupnum):
        super().__init__()
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groupnum)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        return x


class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, groupnum=1):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1, groupnum=groupnum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1, groupnum=groupnum)

    def forward(self, x, weight=None, bias=None, filterMod=False):
        if filterMod:
            x1 = self.conv1(x)
            x2 = weight * x1 + bias * x
            x3 = self.relu(x2)
            x4 = self.conv2(x3)
            x5 = weight * x4 + bias * x3
            return x + x5
        else:
            return x + self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            ConvLayer(3, int(16 * slim_factor), kernel_size=9, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(16 * slim_factor), int(32 * slim_factor), kernel_size=3, stride=2, groupnum=int(16 * slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32 * slim_factor), int(32 * slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(32 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=2, groupnum=int(32 * slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ResidualLayer(int(64 * slim_factor), kernel_size=3),
        )
        self.enc2 = nn.Sequential(
            ResidualLayer(int(64 * slim_factor), kernel_size=3),
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        return x1, x2


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec1 = ResidualLayer(int(64 * slim_factor), kernel_size=3)
        self.dec2 = ResidualLayer(int(64 * slim_factor), kernel_size=3)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(int(64 * slim_factor), int(32 * slim_factor), kernel_size=3, stride=1, groupnum=int(32 * slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32 * slim_factor), int(32 * slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(int(32 * slim_factor), int(16 * slim_factor), kernel_size=3, stride=1, groupnum=int(16 * slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(16 * slim_factor), int(16 * slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(16 * slim_factor), 3, kernel_size=9, stride=1, groupnum=1),
        )

    def forward(self, content_feats_0, content_feats_1, style_feats_0, style_feats_1, w0, b0, w1, b1):
        # Inline the alpha=1.0 logic: featMod is fully applied
        x1 = featMod(content_feats_1, style_feats_1)
        # alpha=1.0: x1 = 1.0 * x1 + 0.0 * content_feats_1 = x1

        x2 = self.dec1(x1, w1, b1, filterMod=True)

        x3 = featMod(x2, style_feats_0)
        # alpha=1.0: x3 = 1.0 * x3 + 0.0 * x2 = x3

        x4 = self.dec2(x3, w0, b0, filterMod=True)

        out = self.dec3(x4)
        return out


class Modulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Sequential(
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=1, groupnum=int(64 * slim_factor)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.bias1 = nn.Sequential(
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=1, groupnum=int(64 * slim_factor)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.weight2 = nn.Sequential(
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=1, groupnum=int(64 * slim_factor)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.bias2 = nn.Sequential(
            ConvLayer(int(64 * slim_factor), int(64 * slim_factor), kernel_size=3, stride=1, groupnum=int(64 * slim_factor)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x0, x1):
        w1 = self.weight1(x0)
        b1 = self.bias1(x0)
        w2 = self.weight2(x1)
        b2 = self.bias2(x1)
        return w1, w2, b1, b2


# ---------------------------------------------------------------------------
# Step 3: Combined export wrapper -- flattens all list args for tracing
# ---------------------------------------------------------------------------

class MicroASTExport(nn.Module):
    """
    Single-module wrapper for ONNX export.

    Inputs:
        content: [1, 3, H, W] float32, values in [0, 1]
        style:   [1, 3, H, W] float32, values in [0, 1]

    Output:
        output:  [1, 3, H, W] float32, values clamped to [0, 1]

    Alpha is baked in as 1.0 (full stylization).
    """

    def __init__(self, content_encoder, style_encoder, modulator, decoder):
        super().__init__()
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.modulator = modulator
        self.decoder = decoder

    def forward(self, content, style):
        # Encode
        content_f0, content_f1 = self.content_encoder(content)
        style_f0, style_f1 = self.style_encoder(style)

        # Modulate
        w0, w1, b0, b1 = self.modulator(style_f0, style_f1)

        # Decode (alpha=1.0 baked in)
        out = self.decoder(content_f0, content_f1, style_f0, style_f1, w0, b0, w1, b1)

        # Clamp output to [0, 1]
        out = torch.clamp(out, 0.0, 1.0)

        return out


# ---------------------------------------------------------------------------
# Step 4: Load weights and export
# ---------------------------------------------------------------------------

ONNX_OUTPUT = Path(__file__).parent / "assets" / "models" / "microast.onnx"
EXPORT_H = 256
EXPORT_W = 256


def load_weights():
    """Load the 4 pretrained checkpoints into our inlined model classes."""
    models_dir = REPO_DIR / "models"

    content_encoder = Encoder()
    style_encoder = Encoder()
    modulator = Modulator()
    decoder = Decoder()

    # Load state dicts. The original code uses torch.load without weights_only,
    # and checkpoints are .pth.tar files (just state_dicts, no full model objects).
    content_encoder.load_state_dict(
        torch.load(models_dir / "content_encoder_iter_160000.pth.tar", map_location="cpu", weights_only=True)
    )
    style_encoder.load_state_dict(
        torch.load(models_dir / "style_encoder_iter_160000.pth.tar", map_location="cpu", weights_only=True)
    )
    modulator.load_state_dict(
        torch.load(models_dir / "modulator_iter_160000.pth.tar", map_location="cpu", weights_only=True)
    )

    # The Decoder class has a different forward signature in our export version
    # (flattened args instead of lists). But the state_dict is the same because
    # only the forward() changed, not __init__. So loading works directly.
    decoder.load_state_dict(
        torch.load(models_dir / "decoder_iter_160000.pth.tar", map_location="cpu", weights_only=True)
    )

    content_encoder.eval()
    style_encoder.eval()
    modulator.eval()
    decoder.eval()

    return content_encoder, style_encoder, modulator, decoder


def export_onnx():
    """Export MicroAST to ONNX."""
    print("[...] Loading pretrained weights...")
    content_encoder, style_encoder, modulator, decoder = load_weights()

    model = MicroASTExport(content_encoder, style_encoder, modulator, decoder)
    model.eval()

    # Dummy inputs for tracing
    content_dummy = torch.randn(1, 3, EXPORT_H, EXPORT_W)
    style_dummy = torch.randn(1, 3, EXPORT_H, EXPORT_W)

    print(f"[...] Running PyTorch forward pass to verify model works...")
    with torch.no_grad():
        pt_output = model(content_dummy, style_dummy)
    print(f"[OK] PyTorch output shape: {pt_output.shape}, "
          f"range: [{pt_output.min().item():.4f}, {pt_output.max().item():.4f}]")

    # Export
    ONNX_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[...] Exporting to ONNX: {ONNX_OUTPUT}")

    torch.onnx.export(
        model,
        (content_dummy, style_dummy),
        str(ONNX_OUTPUT),
        opset_version=11,
        input_names=["content", "style"],
        output_names=["output"],
        dynamic_axes={
            "content": {2: "height", 3: "width"},
            "style": {2: "style_height", 3: "style_width"},
            "output": {2: "height", 3: "width"},
        },
    )

    file_size_mb = ONNX_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"[OK] Exported ONNX model: {ONNX_OUTPUT} ({file_size_mb:.2f} MB)")

    return pt_output, content_dummy, style_dummy


def verify_onnx(pt_output, content_dummy, style_dummy):
    """Verify the exported ONNX model with onnxruntime."""
    print("[...] Verifying ONNX model with onnx.checker...")
    onnx_model = onnx.load(str(ONNX_OUTPUT))
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model passed checker validation")

    # Print model info
    print(f"\n--- ONNX Model Info ---")
    for inp in onnx_model.graph.input:
        shape = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name:10s} shape={shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name:10s} shape={shape}")
    print(f"  Opset version: {onnx_model.opset_import[0].version}")
    print(f"  IR version: {onnx_model.ir_version}")
    print()

    # Run inference with onnxruntime
    print("[...] Running onnxruntime inference...")
    sess = ort.InferenceSession(str(ONNX_OUTPUT))

    content_np = content_dummy.numpy()
    style_np = style_dummy.numpy()

    ort_outputs = sess.run(
        ["output"],
        {"content": content_np, "style": style_np},
    )
    ort_output = ort_outputs[0]
    print(f"[OK] ORT output shape: {ort_output.shape}, "
          f"range: [{ort_output.min():.4f}, {ort_output.max():.4f}]")

    # Compare PyTorch vs ORT outputs
    pt_np = pt_output.numpy()
    max_diff = np.abs(pt_np - ort_output).max()
    mean_diff = np.abs(pt_np - ort_output).mean()
    print(f"[OK] Max diff (PyTorch vs ORT): {max_diff:.6f}")
    print(f"[OK] Mean diff (PyTorch vs ORT): {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("[OK] Outputs match within tolerance (1e-3)")
    elif max_diff < 1e-2:
        print("[WARN] Outputs have small differences (max_diff < 1e-2), likely float precision")
    else:
        print(f"[WARN] Large difference detected: max_diff={max_diff:.6f}")

    # Test with different resolution to verify dynamic axes
    print("\n[...] Testing dynamic resolution (512x384)...")
    content_hires = torch.randn(1, 3, 384, 512).numpy()
    style_hires = torch.randn(1, 3, 256, 256).numpy()
    ort_hires = sess.run(["output"], {"content": content_hires, "style": style_hires})
    print(f"[OK] Dynamic resolution output shape: {ort_hires[0].shape}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MicroAST ONNX Export Script")
    print("=" * 60)

    # Step 1: Clone repo
    clone_repo()

    # Step 2-3: Load weights + export
    pt_output, content_dummy, style_dummy = export_onnx()

    # Step 4: Verify
    print("\n" + "-" * 60)
    verify_onnx(pt_output, content_dummy, style_dummy)

    print("\n" + "=" * 60)
    print(f"DONE. ONNX model saved to: {ONNX_OUTPUT}")
    print(f"\nModel details:")
    print(f"  Inputs:  content [1, 3, H, W]  (float32, range [0,1])")
    print(f"           style   [1, 3, H, W]  (float32, range [0,1])")
    print(f"  Output:  output  [1, 3, H, W]  (float32, clamped [0,1])")
    print(f"  Alpha:   1.0 (baked in, full stylization)")
    print(f"  Opset:   11")
    print("=" * 60)


if __name__ == "__main__":
    main()
