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
Compare MicroAST PyTorch output vs ONNX output to find edge artifacts.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).parent
REPO_DIR = ROOT / "MicroAST"
ONNX_PATH = ROOT / "assets" / "models" / "microast.onnx"
STYLES_DIR = ROOT / "assets" / "styles"

# -------------------------------------------------------------------
# Inline MicroAST model (from export script)
# -------------------------------------------------------------------
def calc_mean_std(feat, eps=1e-5):
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def featMod(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, groupnum):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv_layer = nn.Conv2d(in_c, out_c, kernel_size, stride, groups=groupnum)
    def forward(self, x):
        return self.conv_layer(self.reflection_pad(x))


class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, groupnum=1):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, 1, groupnum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size, 1, groupnum)

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
            ConvLayer(3, 16, 9, 1, 1), nn.ReLU(inplace=True),
            ConvLayer(16, 32, 3, 2, 16), nn.ReLU(inplace=True),
            ConvLayer(32, 32, 1, 1, 1), nn.ReLU(inplace=True),
            ConvLayer(32, 64, 3, 2, 32), nn.ReLU(inplace=True),
            ConvLayer(64, 64, 1, 1, 1), nn.ReLU(inplace=True),
            ResidualLayer(64, 3),
        )
        self.enc2 = nn.Sequential(ResidualLayer(64, 3))

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        return [x1, x2]  # Returns a list like original


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec1 = ResidualLayer(64, 3)
        self.dec2 = ResidualLayer(64, 3)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(64, 32, 3, 1, 32), nn.ReLU(inplace=True),
            ConvLayer(32, 32, 1, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(32, 16, 3, 1, 16), nn.ReLU(inplace=True),
            ConvLayer(16, 16, 1, 1, 1), nn.ReLU(inplace=True),
            ConvLayer(16, 3, 9, 1, 1),
        )

    def forward(self, x, s, w, b, alpha):
        x1 = featMod(x[1], s[1])
        x1 = alpha * x1 + (1 - alpha) * x[1]
        x2 = self.dec1(x1, w[1], b[1], filterMod=True)
        x3 = featMod(x2, s[0])
        x3 = alpha * x3 + (1 - alpha) * x2
        x4 = self.dec2(x3, w[0], b[0], filterMod=True)
        return self.dec3(x4)


class Modulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Sequential(ConvLayer(64, 64, 3, 1, 64), nn.AdaptiveAvgPool2d((1,1)))
        self.bias1 = nn.Sequential(ConvLayer(64, 64, 3, 1, 64), nn.AdaptiveAvgPool2d((1,1)))
        self.weight2 = nn.Sequential(ConvLayer(64, 64, 3, 1, 64), nn.AdaptiveAvgPool2d((1,1)))
        self.bias2 = nn.Sequential(ConvLayer(64, 64, 3, 1, 64), nn.AdaptiveAvgPool2d((1,1)))

    def forward(self, x):
        return [self.weight1(x[0]), self.weight2(x[1])], [self.bias1(x[0]), self.bias2(x[1])]


class TestNet(nn.Module):
    def __init__(self, ce, se, mod, dec):
        super().__init__()
        self.content_encoder = ce
        self.style_encoder = se
        self.modulator = mod
        self.decoder = dec

    def forward(self, content, style, alpha=1.0):
        style_feats = self.style_encoder(style)
        fw, fb = self.modulator(style_feats)
        content_feats = self.content_encoder(content)
        return self.decoder(content_feats, style_feats, fw, fb, alpha)


# -------------------------------------------------------------------

def load_pytorch_model():
    models_dir = REPO_DIR / "models"
    ce = Encoder(); se = Encoder(); mod = Modulator(); dec = Decoder()
    ce.load_state_dict(torch.load(models_dir / "content_encoder_iter_160000.pth.tar", map_location="cpu", weights_only=True))
    se.load_state_dict(torch.load(models_dir / "style_encoder_iter_160000.pth.tar", map_location="cpu", weights_only=True))
    mod.load_state_dict(torch.load(models_dir / "modulator_iter_160000.pth.tar", map_location="cpu", weights_only=True))
    dec.load_state_dict(torch.load(models_dir / "decoder_iter_160000.pth.tar", map_location="cpu", weights_only=True))
    net = TestNet(ce, se, mod, dec)
    net.eval()
    return net


def main():
    print("=" * 60)
    print("MicroAST: PyTorch vs ONNX Edge Artifact Analysis")
    print("=" * 60)

    if not ONNX_PATH.exists():
        print(f"ONNX model not found at {ONNX_PATH}")
        sys.exit(1)

    # Load models
    print("\n[1] Loading PyTorch model...")
    net = load_pytorch_model()

    print("[2] Loading ONNX model...")
    sess = ort.InferenceSession(str(ONNX_PATH))

    # Load real images
    style_files = sorted(STYLES_DIR.glob("*.jpg"))
    content_path = style_files[0]
    style_path = style_files[1] if len(style_files) > 1 else style_files[0]

    SIZE = 512
    content_img = Image.open(content_path).resize((SIZE, SIZE), Image.LANCZOS).convert("RGB")
    style_img = Image.open(style_path).resize((SIZE, SIZE), Image.LANCZOS).convert("RGB")

    content_t = torch.from_numpy(np.array(content_img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0)
    style_t = torch.from_numpy(np.array(style_img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0)

    print(f"  Content: {content_path.name} [{SIZE}x{SIZE}]")
    print(f"  Style:   {style_path.name} [{SIZE}x{SIZE}]")

    # Run PyTorch
    print("\n[3] Running PyTorch inference...")
    with torch.no_grad():
        pt_out = net(content_t, style_t, alpha=1.0)
    pt_np = pt_out.numpy()
    print(f"  PyTorch output: range [{pt_np.min():.4f}, {pt_np.max():.4f}]")

    # Run ONNX
    print("[4] Running ONNX inference...")
    ort_out = sess.run(["output"], {"content": content_t.numpy(), "style": style_t.numpy()})[0]
    print(f"  ONNX output: range [{ort_out.min():.4f}, {ort_out.max():.4f}]")

    # Note: ONNX model has clamp(0,1) baked in, PyTorch doesn't
    pt_clamped = np.clip(pt_np, 0, 1)

    # Global comparison
    diff = np.abs(pt_clamped - ort_out)
    print(f"\n[5] Global diff (after clamp): max={diff.max():.6f}, mean={diff.mean():.6f}")

    # Edge-specific comparison
    border = 16  # pixels from edge
    center = pt_clamped[:, :, border:-border, border:-border]
    center_ort = ort_out[:, :, border:-border, border:-border]
    edge_mask = np.ones_like(pt_clamped, dtype=bool)
    edge_mask[:, :, border:-border, border:-border] = False

    edge_diff = diff[edge_mask]
    center_diff = np.abs(center - center_ort)

    print(f"\n[6] Edge vs center diff:")
    print(f"  Edge  (border {border}px): max={edge_diff.max():.6f}, mean={edge_diff.mean():.6f}")
    print(f"  Center:                   max={center_diff.max():.6f}, mean={center_diff.mean():.6f}")

    if edge_diff.mean() > center_diff.mean() * 2:
        print(f"  [!!] Edge error is {edge_diff.mean()/center_diff.mean():.1f}x higher than center — possible Pad/ReflectionPad2d issue!")
    else:
        print(f"  [OK] Edge and center errors are similar — no Pad issue detected")

    # Check if output values are in valid range (unclamped PyTorch)
    out_of_range = (pt_np < 0).sum() + (pt_np > 1).sum()
    pct = out_of_range / pt_np.size * 100
    print(f"\n[7] PyTorch unclamped: {pct:.1f}% values outside [0,1], range [{pt_np.min():.4f}, {pt_np.max():.4f}]")

    # Save comparison images
    out_dir = ROOT / "debug_output"
    out_dir.mkdir(exist_ok=True)

    def save(arr, name):
        img = (np.clip(arr[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(out_dir / name)

    save(pt_clamped, "microast_pytorch.png")
    save(ort_out, "microast_onnx.png")

    # Save diff map (amplified)
    diff_amplified = np.clip(diff[0].transpose(1, 2, 0) * 50, 0, 1)  # 50x amplification
    diff_img = (diff_amplified * 255).astype(np.uint8)
    Image.fromarray(diff_img).save(out_dir / "microast_diff_50x.png")

    # Save edge crop comparisons (top-left corner 64x64)
    crop = 64
    pt_corner = (np.clip(pt_clamped[0, :, :crop, :crop].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    ort_corner = (np.clip(ort_out[0, :, :crop, :crop].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pt_corner).resize((256, 256), Image.NEAREST).save(out_dir / "microast_corner_pytorch.png")
    Image.fromarray(ort_corner).resize((256, 256), Image.NEAREST).save(out_dir / "microast_corner_onnx.png")

    print(f"\n  Saved comparison images to {out_dir}/microast_*.png")

    # ---- Check ONNX graph for Pad nodes ----
    print(f"\n[8] ONNX graph analysis...")
    import onnx
    model = onnx.load(str(ONNX_PATH))

    pad_nodes = [n for n in model.graph.node if n.op_type == "Pad"]
    resize_nodes = [n for n in model.graph.node if n.op_type == "Resize"]
    upsample_nodes = [n for n in model.graph.node if n.op_type == "Upsample"]

    print(f"  Pad nodes: {len(pad_nodes)}")
    for p in pad_nodes:
        mode = "unknown"
        for attr in p.attribute:
            if attr.name == "mode":
                mode = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
        print(f"    {p.name}: mode={mode}")

    print(f"  Resize nodes: {len(resize_nodes)}")
    for r in resize_nodes:
        for attr in r.attribute:
            if attr.name == "coordinate_transformation_mode":
                val = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
                print(f"    {r.name}: coord_transform={val}")
            if attr.name == "mode":
                val = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
                print(f"    {r.name}: mode={val}")

    print(f"  Upsample nodes: {len(upsample_nodes)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
