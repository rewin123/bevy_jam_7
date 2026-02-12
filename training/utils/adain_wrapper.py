"""Wrapper around pytorch-AdaIN for fast golden image generation.

Provides AdaINStylizer class — same interface as NNSTStylizer but uses
AdaIN (Adaptive Instance Normalization) which runs in ~60ms per image
instead of ~20s for NNST.

Usage:
    stylizer = AdaINStylizer(style_paths, device)
    golden = stylizer.stylize(content_tensor)  # [1,3,H,W] -> [1,3,H,W]
"""

import hashlib
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_ADAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "vendor", "AdaIN")
_ADAIN_DIR = os.path.abspath(_ADAIN_DIR)

_DECODER_PATH = os.path.join(_ADAIN_DIR, "models", "decoder.pth")
_VGG_PATH = os.path.join(_ADAIN_DIR, "models", "vgg_normalised.pth")


def _adaptive_instance_normalization(content_feat, style_feat):
    """AdaIN: align content feature statistics to style feature statistics."""
    size = content_feat.size()
    N, C = size[:2]
    content_mean = content_feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    content_var = content_feat.view(N, C, -1).var(dim=2) + 1e-5
    content_std = content_var.sqrt().view(N, C, 1, 1)
    style_mean = style_feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    style_var = style_feat.view(N, C, -1).var(dim=2) + 1e-5
    style_std = style_var.sqrt().view(N, C, 1, 1)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _build_decoder():
    """AdaIN decoder architecture (from net.py)."""
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


def _build_vgg():
    """VGG-19 encoder with normalized weights (from net.py)."""
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1 — encoder output
    )


class AdaINStylizer:
    """Wraps AdaIN style transfer for fast golden generation.

    Single forward pass through VGG encoder + AdaIN + decoder.
    ~60ms per image at 512px on GPU.
    """

    def __init__(
        self,
        style_paths: list[str],
        device: torch.device,
        alpha: float = 1.0,
        **kwargs,  # accept and ignore NNST-specific params for compatibility
    ):
        self.style_paths = sorted(style_paths)
        self.device = device
        self.alpha = alpha

        # Load VGG encoder (truncated to relu4_1 = first 31 layers of full VGG)
        vgg_full = _build_vgg()
        # The full VGG has layers up to relu5-4 but we only need up to relu4-1
        # _build_vgg already returns only up to relu4-1 (31 layers)
        vgg_full.load_state_dict(
            torch.load(_VGG_PATH, map_location="cpu", weights_only=True),
            strict=False,
        )
        self._encoder = vgg_full.to(device).eval()
        for p in self._encoder.parameters():
            p.requires_grad = False

        # Load decoder
        self._decoder = _build_decoder()
        self._decoder.load_state_dict(
            torch.load(_DECODER_PATH, map_location="cpu", weights_only=True),
        )
        self._decoder = self._decoder.to(device).eval()
        for p in self._decoder.parameters():
            p.requires_grad = False

        # Pre-encode style images (only need to encode once)
        self._style_feats: list[torch.Tensor] = []
        for sp in self.style_paths:
            img = self._load_image(sp).to(device)
            with torch.no_grad():
                feat = self._encoder(img)
            self._style_feats.append(feat)

        print(f"AdaIN loaded: {len(self._style_feats)} style(s), alpha={alpha}")

    def _load_image(self, path: str) -> torch.Tensor:
        """Load image as [1,3,H,W] tensor in [0,1]."""
        from PIL import Image
        from torchvision import transforms

        img = Image.open(path).convert("RGB")
        tensor = transforms.ToTensor()(img)
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def stylize(
        self, content: torch.Tensor, style_index: int = 0
    ) -> torch.Tensor:
        """Stylize content image using AdaIN.

        Args:
            content: [1,3,H,W] tensor in [0,1] on self.device
            style_index: which style image to use (for round-robin multi-style)

        Returns:
            [1,3,H,W] tensor in [0,1] on self.device
        """
        idx = style_index % len(self._style_feats)
        style_feat = self._style_feats[idx]

        content_feat = self._encoder(content)

        # AdaIN: align content statistics to style statistics
        t = _adaptive_instance_normalization(content_feat, style_feat)
        t = self.alpha * t + (1 - self.alpha) * content_feat

        output = self._decoder(t)
        return output.clamp(0.0, 1.0)

    def style_hash(self) -> str:
        """Hash of style images + AdaIN parameters for cache invalidation."""
        h = hashlib.sha256()
        h.update(b"adain_v1")
        for sp in self.style_paths:
            with open(sp, "rb") as f:
                h.update(f.read())
        h.update(f"alpha={self.alpha}".encode())
        return h.hexdigest()[:12]

    def cleanup(self):
        """Free GPU memory."""
        del self._encoder
        del self._decoder
        del self._style_feats
        self._style_feats = []
        torch.cuda.empty_cache()
