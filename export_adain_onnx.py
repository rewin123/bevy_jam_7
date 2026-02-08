# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "onnx>=1.14",
#     "onnxruntime>=1.15",
#     "onnxscript",
#     "numpy",
# ]
# ///
"""
Export AdaIN style transfer model to ONNX with dynamic spatial dimensions.

Usage:
    uv run export_adain_onnx.py

Downloads weights from naoto0804/pytorch-AdaIN releases, exports encoder
(truncated VGG-19) and decoder as separate ONNX files with dynamic H,W axes.

Encoder input: [1,3,H,W] float32 range [0,1]
Encoder output: [1,512,H/8,W/8] float32
Decoder input: [1,512,h,w] float32
Decoder output: [1,3,H',W'] float32 range ~[0,1]
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

MODELS_DIR = Path(__file__).parent / "assets" / "models"
WEIGHTS_DIR = Path(__file__).parent / ".adain_weights"

VGG_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
DECODER_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"


def download_weights():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    vgg_path = WEIGHTS_DIR / "vgg_normalised.pth"
    dec_path = WEIGHTS_DIR / "decoder.pth"

    if not vgg_path.exists():
        print(f"[...] Downloading VGG weights...")
        subprocess.check_call(["curl", "-L", "-o", str(vgg_path), VGG_URL])
    else:
        print(f"[OK] VGG weights already present")

    if not dec_path.exists():
        print(f"[...] Downloading decoder weights...")
        subprocess.check_call(["curl", "-L", "-o", str(dec_path), DECODER_URL])
    else:
        print(f"[OK] Decoder weights already present")

    return vgg_path, dec_path


# VGG-19 encoder truncated to relu4_1 (first 31 layers of normalised VGG)
def build_vgg_encoder():
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4_1
    )


def build_decoder():
    return nn.Sequential(
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


def main():
    print("=" * 60)
    print("AdaIN ONNX Export Script (dynamic axes)")
    print("=" * 60)

    vgg_path, dec_path = download_weights()

    # Load full VGG and take first 31 layers
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
    full_vgg.load_state_dict(torch.load(str(vgg_path), map_location="cpu", weights_only=True))

    encoder = nn.Sequential(*list(full_vgg.children())[:31])
    encoder.eval()

    decoder = build_decoder()
    decoder.load_state_dict(torch.load(str(dec_path), map_location="cpu", weights_only=True))
    decoder.eval()

    # Test forward pass
    print("[...] Testing forward pass...")
    dummy_img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        enc_out = encoder(dummy_img)
        dec_out = decoder(enc_out)
    print(f"[OK] Encoder: {dummy_img.shape} -> {enc_out.shape}")
    print(f"[OK] Decoder: {enc_out.shape} -> {dec_out.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Export encoder (dynamo=False to embed weights inline)
    enc_path = MODELS_DIR / "adain-vgg.onnx"
    print(f"[...] Exporting encoder to {enc_path}...")
    torch.onnx.export(
        encoder, dummy_img, str(enc_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "h", 3: "w"},
        },
    )
    enc_size = enc_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Encoder exported: {enc_size:.2f} MB")

    # Export decoder (dynamo=False to embed weights inline)
    dec_path_onnx = MODELS_DIR / "adain-decoder.onnx"
    dummy_feat = torch.randn(1, 512, 28, 28)  # 224/8 = 28
    print(f"[...] Exporting decoder to {dec_path_onnx}...")
    torch.onnx.export(
        decoder, dummy_feat, str(dec_path_onnx),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "input": {2: "h", 3: "w"},
            "output": {2: "height", 3: "width"},
        },
    )
    dec_size = dec_path_onnx.stat().st_size / (1024 * 1024)
    print(f"[OK] Decoder exported: {dec_size:.2f} MB")

    # Verify with ORT
    print("\n[...] Verifying with ONNX Runtime...")
    enc_sess = ort.InferenceSession(str(enc_path))
    dec_sess = ort.InferenceSession(str(dec_path_onnx))

    img_np = dummy_img.numpy()
    enc_result = enc_sess.run(["output"], {"input": img_np})[0]
    dec_result = dec_sess.run(["output"], {"input": enc_result})[0]
    print(f"[OK] ORT encoder: {img_np.shape} -> {enc_result.shape}")
    print(f"[OK] ORT decoder: {enc_result.shape} -> {dec_result.shape}")

    # Test at different resolution
    print("\n[...] Testing dynamic resolution (256x192)...")
    img_256 = np.random.randn(1, 3, 192, 256).astype(np.float32)
    enc_256 = enc_sess.run(["output"], {"input": img_256})[0]
    dec_256 = dec_sess.run(["output"], {"input": enc_256})[0]
    print(f"[OK] 256x192: encoder {enc_256.shape}, decoder {dec_256.shape}")

    print("\n" + "=" * 60)
    print("DONE.")
    print(f"  Encoder: {enc_path} ({enc_size:.2f} MB)")
    print(f"  Decoder: {dec_path_onnx} ({dec_size:.2f} MB)")
    print(f"  Input names: 'input' / Output names: 'output'")
    print(f"  Dynamic spatial axes supported")
    print("=" * 60)


if __name__ == "__main__":
    main()
