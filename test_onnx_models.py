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
Test that exported ONNX models match their PyTorch originals.

Usage:
    uv run test_onnx_models.py

Runs deterministic comparison tests for:
  - AdaIN encoder (VGG) and decoder
  - MicroAST combined model

Fails with non-zero exit code if max absolute difference exceeds tolerance.
"""

import sys
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "assets" / "models"

# Tolerance for PyTorch vs ORT comparison (float32 precision)
ATOL = 1e-4

# Fixed seed for reproducibility
SEED = 42

passed = 0
failed = 0


def report(name: str, ok: bool, msg: str):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}: {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {name}: {msg}")


# =========================================================================
# AdaIN: build PyTorch models (same as export_adain_onnx.py)
# =========================================================================

WEIGHTS_DIR = ROOT / ".adain_weights"


def build_adain_encoder():
    """VGG-19 encoder truncated to relu4_1 (first 31 layers)."""
    full_vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    )
    full_vgg.load_state_dict(
        torch.load(str(WEIGHTS_DIR / "vgg_normalised.pth"), map_location="cpu", weights_only=True)
    )
    encoder = nn.Sequential(*list(full_vgg.children())[:31])
    encoder.eval()
    return encoder


def build_adain_decoder():
    decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )
    decoder.load_state_dict(
        torch.load(str(WEIGHTS_DIR / "decoder.pth"), map_location="cpu", weights_only=True)
    )
    decoder.eval()
    return decoder


# =========================================================================
# MicroAST: build PyTorch model (same as export_microast_onnx.py)
# =========================================================================

MICROAST_REPO = ROOT / "MicroAST"


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
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


class MicroEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            ConvLayer(3, 16, kernel_size=9, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(16, 32, kernel_size=3, stride=2, groupnum=16),
            nn.ReLU(inplace=True),
            ConvLayer(32, 32, kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(32, 64, kernel_size=3, stride=2, groupnum=32),
            nn.ReLU(inplace=True),
            ConvLayer(64, 64, kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ResidualLayer(64, kernel_size=3),
        )
        self.enc2 = nn.Sequential(
            ResidualLayer(64, kernel_size=3),
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        return x1, x2


class MicroDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec1 = ResidualLayer(64, kernel_size=3)
        self.dec2 = ResidualLayer(64, kernel_size=3)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(64, 32, kernel_size=3, stride=1, groupnum=32),
            nn.ReLU(inplace=True),
            ConvLayer(32, 32, kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(32, 16, kernel_size=3, stride=1, groupnum=16),
            nn.ReLU(inplace=True),
            ConvLayer(16, 16, kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(16, 3, kernel_size=9, stride=1, groupnum=1),
        )

    def forward(self, content_feats_0, content_feats_1, style_feats_0, style_feats_1, w0, b0, w1, b1):
        x1 = featMod(content_feats_1, style_feats_1)
        x2 = self.dec1(x1, w1, b1, filterMod=True)
        x3 = featMod(x2, style_feats_0)
        x4 = self.dec2(x3, w0, b0, filterMod=True)
        out = self.dec3(x4)
        return out


class MicroModulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Sequential(
            ConvLayer(64, 64, kernel_size=3, stride=1, groupnum=64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.bias1 = nn.Sequential(
            ConvLayer(64, 64, kernel_size=3, stride=1, groupnum=64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.weight2 = nn.Sequential(
            ConvLayer(64, 64, kernel_size=3, stride=1, groupnum=64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.bias2 = nn.Sequential(
            ConvLayer(64, 64, kernel_size=3, stride=1, groupnum=64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x0, x1):
        w1 = self.weight1(x0)
        b1 = self.bias1(x0)
        w2 = self.weight2(x1)
        b2 = self.bias2(x1)
        return w1, w2, b1, b2


class MicroASTCombined(nn.Module):
    def __init__(self, content_encoder, style_encoder, modulator, decoder):
        super().__init__()
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.modulator = modulator
        self.decoder = decoder

    def forward(self, content, style):
        content_f0, content_f1 = self.content_encoder(content)
        style_f0, style_f1 = self.style_encoder(style)
        w0, w1, b0, b1 = self.modulator(style_f0, style_f1)
        out = self.decoder(content_f0, content_f1, style_f0, style_f1, w0, b0, w1, b1)
        return torch.clamp(out, 0.0, 1.0)


def load_microast_pytorch():
    models_dir = MICROAST_REPO / "models"
    content_enc = MicroEncoder()
    style_enc = MicroEncoder()
    modulator = MicroModulator()
    decoder = MicroDecoder()

    content_enc.load_state_dict(torch.load(models_dir / "content_encoder_iter_160000.pth.tar", map_location="cpu", weights_only=True))
    style_enc.load_state_dict(torch.load(models_dir / "style_encoder_iter_160000.pth.tar", map_location="cpu", weights_only=True))
    modulator.load_state_dict(torch.load(models_dir / "modulator_iter_160000.pth.tar", map_location="cpu", weights_only=True))
    decoder.load_state_dict(torch.load(models_dir / "decoder_iter_160000.pth.tar", map_location="cpu", weights_only=True))

    for m in [content_enc, style_enc, modulator, decoder]:
        m.eval()

    model = MicroASTCombined(content_enc, style_enc, modulator, decoder)
    model.eval()
    return model


# =========================================================================
# Tests
# =========================================================================

def compare(name: str, pt_out: np.ndarray, ort_out: np.ndarray, atol: float = ATOL):
    max_diff = np.abs(pt_out - ort_out).max()
    mean_diff = np.abs(pt_out - ort_out).mean()
    ok = max_diff < atol
    report(name, ok, f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} (atol={atol:.0e})")
    return ok


def test_adain_encoder():
    print("\n--- AdaIN Encoder ---")
    enc_path = MODELS_DIR / "adain-vgg.onnx"
    if not enc_path.exists():
        report("adain_encoder_exists", False, f"{enc_path} not found")
        return
    report("adain_encoder_exists", True, f"{enc_path.stat().st_size / 1e6:.1f} MB")

    # PyTorch
    encoder = build_adain_encoder()
    torch.manual_seed(SEED)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pt_out = encoder(x).numpy()

    # ORT
    sess = ort.InferenceSession(str(enc_path))
    ort_out = sess.run(["output"], {"input": x.numpy()})[0]

    # Shape
    report("adain_encoder_shape", pt_out.shape == ort_out.shape,
           f"pt={pt_out.shape} ort={ort_out.shape}")

    # Values (VGG-19 is 31 layers deep, float32 error accumulates — use 2e-4)
    compare("adain_encoder_values_224", pt_out, ort_out, atol=2e-4)

    # Dynamic resolution
    torch.manual_seed(SEED + 1)
    x2 = torch.randn(1, 3, 192, 256)
    with torch.no_grad():
        pt_out2 = encoder(x2).numpy()
    ort_out2 = sess.run(["output"], {"input": x2.numpy()})[0]
    report("adain_encoder_dynamic_shape", pt_out2.shape == ort_out2.shape,
           f"pt={pt_out2.shape} ort={ort_out2.shape}")
    compare("adain_encoder_values_256x192", pt_out2, ort_out2, atol=2e-4)


def test_adain_decoder():
    print("\n--- AdaIN Decoder ---")
    dec_path = MODELS_DIR / "adain-decoder.onnx"
    if not dec_path.exists():
        report("adain_decoder_exists", False, f"{dec_path} not found")
        return
    report("adain_decoder_exists", True, f"{dec_path.stat().st_size / 1e6:.1f} MB")

    # PyTorch
    decoder = build_adain_decoder()
    torch.manual_seed(SEED)
    x = torch.randn(1, 512, 28, 28)
    with torch.no_grad():
        pt_out = decoder(x).numpy()

    # ORT
    sess = ort.InferenceSession(str(dec_path))
    ort_out = sess.run(["output"], {"input": x.numpy()})[0]

    # Shape
    report("adain_decoder_shape", pt_out.shape == ort_out.shape,
           f"pt={pt_out.shape} ort={ort_out.shape}")

    # Values
    compare("adain_decoder_values_28x28", pt_out, ort_out)

    # Dynamic resolution
    torch.manual_seed(SEED + 1)
    x2 = torch.randn(1, 512, 16, 24)
    with torch.no_grad():
        pt_out2 = decoder(x2).numpy()
    ort_out2 = sess.run(["output"], {"input": x2.numpy()})[0]
    report("adain_decoder_dynamic_shape", pt_out2.shape == ort_out2.shape,
           f"pt={pt_out2.shape} ort={ort_out2.shape}")
    compare("adain_decoder_values_24x16", pt_out2, ort_out2)


def test_adain_roundtrip():
    """End-to-end: encode → decode, compare PyTorch vs ORT."""
    print("\n--- AdaIN Roundtrip (encode -> decode) ---")
    enc_path = MODELS_DIR / "adain-vgg.onnx"
    dec_path = MODELS_DIR / "adain-decoder.onnx"
    if not enc_path.exists() or not dec_path.exists():
        report("adain_roundtrip", False, "ONNX models not found")
        return

    encoder = build_adain_encoder()
    decoder = build_adain_decoder()

    torch.manual_seed(SEED)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        feats = encoder(x)
        pt_out = decoder(feats).numpy()

    enc_sess = ort.InferenceSession(str(enc_path))
    dec_sess = ort.InferenceSession(str(dec_path))
    ort_feats = enc_sess.run(["output"], {"input": x.numpy()})[0]
    ort_out = dec_sess.run(["output"], {"input": ort_feats})[0]

    compare("adain_roundtrip_values", pt_out, ort_out, atol=5e-4)


def test_microast():
    print("\n--- MicroAST ---")
    onnx_path = MODELS_DIR / "microast.onnx"
    if not onnx_path.exists():
        report("microast_exists", False, f"{onnx_path} not found")
        return
    report("microast_exists", True, f"{onnx_path.stat().st_size / 1e6:.2f} MB")

    if not (MICROAST_REPO / "models").exists():
        report("microast_weights", False, f"MicroAST repo not found at {MICROAST_REPO}")
        return

    # PyTorch
    model = load_microast_pytorch()
    torch.manual_seed(SEED)
    content = torch.randn(1, 3, 256, 256)
    style = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        pt_out = model(content, style).numpy()

    # ORT
    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(["output"], {"content": content.numpy(), "style": style.numpy()})[0]

    # Shape
    report("microast_shape", pt_out.shape == ort_out.shape,
           f"pt={pt_out.shape} ort={ort_out.shape}")

    # Values (MicroAST uses bilinear upsample which can have slightly larger diffs)
    compare("microast_values_256", pt_out, ort_out, atol=1e-3)

    # Dynamic resolution
    torch.manual_seed(SEED + 1)
    content2 = torch.randn(1, 3, 224, 224)
    style2 = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pt_out2 = model(content2, style2).numpy()
    ort_out2 = sess.run(["output"], {"content": content2.numpy(), "style": style2.numpy()})[0]
    report("microast_dynamic_shape", pt_out2.shape == ort_out2.shape,
           f"pt={pt_out2.shape} ort={ort_out2.shape}")
    compare("microast_values_224", pt_out2, ort_out2, atol=1e-3)

    # Output range check (should be clamped to [0, 1])
    ort_min, ort_max = ort_out.min(), ort_out.max()
    in_range = ort_min >= -0.01 and ort_max <= 1.01
    report("microast_output_range", in_range,
           f"min={ort_min:.4f}, max={ort_max:.4f}")


# =========================================================================
# Main
# =========================================================================

def main():
    global passed, failed

    print("=" * 60)
    print("ONNX Model Correctness Tests")
    print("=" * 60)

    # Check prerequisites
    if not (WEIGHTS_DIR / "vgg_normalised.pth").exists():
        print(f"[SKIP] AdaIN weights not found at {WEIGHTS_DIR}")
        print("       Run: uv run export_adain_onnx.py")
    else:
        test_adain_encoder()
        test_adain_decoder()
        test_adain_roundtrip()

    if not (MICROAST_REPO / "models").exists():
        print(f"\n[SKIP] MicroAST repo not found at {MICROAST_REPO}")
        print("       Run: uv run export_microast_onnx.py")
    else:
        test_microast()

    # Summary
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
