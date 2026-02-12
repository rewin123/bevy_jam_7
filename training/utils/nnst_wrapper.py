"""Wrapper around NeuralNeighborStyleTransfer for golden image generation.

Provides NNSTStylizer class that wraps produce_stylization() with:
- sys.path isolation (NNST's nnst_utils vs our utils)
- Style image loading and caching
- Deterministic multi-style assignment via round-robin
- style_hash() for cache invalidation
"""

import hashlib
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

_NNST_DIR = os.path.join(os.path.dirname(__file__), "..", "vendor", "NNST")
_NNST_DIR = os.path.abspath(_NNST_DIR)


def _import_nnst():
    """Import NNST modules with path isolation."""
    sys.path.insert(0, _NNST_DIR)
    try:
        from nnst_utils.stylize import produce_stylization
        from nnst_pretrained.vgg import Vgg16Pretrained
        import nnst_utils.misc as nnst_misc
    finally:
        if _NNST_DIR in sys.path:
            sys.path.remove(_NNST_DIR)
    return produce_stylization, Vgg16Pretrained, nnst_misc


class NNSTStylizer:
    """Wraps NNST produce_stylization for batch golden generation.

    Usage:
        stylizer = NNSTStylizer(style_paths, device)
        golden = stylizer.stylize(content_tensor)  # [1,3,H,W] -> [1,3,H,W]
    """

    def __init__(
        self,
        style_paths: list[str],
        device: torch.device,
        max_iter: int = 200,
        content_weight: float = 0.5,
        max_scls: int = 4,
        flip_aug: bool = True,
    ):
        self.style_paths = sorted(style_paths)
        self.device = device
        self.max_iter = max_iter
        self.content_weight = content_weight
        self.max_scls = max_scls
        self.flip_aug = flip_aug

        # Import NNST
        self._produce_stylization, Vgg16Pretrained, self._nnst_misc = _import_nnst()

        # Set NNST GPU flag
        self._nnst_misc.USE_GPU = device.type == "cuda"

        # Load NNST VGG16
        self._vgg = Vgg16Pretrained().to(device)
        self._vgg.eval()
        self._phi = lambda x, y, z: self._vgg.forward(x, inds=y, concat=z)

        # Pre-load style images as tensors [1,3,H,W] in [0,1]
        self._style_tensors: list[torch.Tensor] = []
        for sp in self.style_paths:
            img = self._load_image(sp)
            self._style_tensors.append(img.to(device))

    def _load_image(self, path: str) -> torch.Tensor:
        """Load image as [1,3,H,W] tensor in [0,1]."""
        from PIL import Image
        from torchvision import transforms

        img = Image.open(path).convert("RGB")
        tensor = transforms.ToTensor()(img)  # [3,H,W] in [0,1]
        return tensor.unsqueeze(0)

    def stylize(
        self, content: torch.Tensor, style_index: int = 0
    ) -> torch.Tensor:
        """Stylize content image using NNST.

        Args:
            content: [1,3,H,W] tensor in [0,1] on self.device
            style_index: which style image to use (for round-robin multi-style)

        Returns:
            [1,3,H,W] tensor in [0,1] on self.device
        """
        idx = style_index % len(self._style_tensors)
        style = self._style_tensors[idx]

        # Resize style to match content's longer side for NNST
        _, _, ch, cw = content.shape
        max_side = max(ch, cw)
        _, _, sh, sw = style.shape
        style_max = max(sh, sw)
        if style_max != max_side:
            scale = max_side / style_max
            new_h = int(sh * scale)
            new_w = int(sw * scale)
            style = F.interpolate(style, (new_h, new_w), mode="bilinear", align_corners=False)

        # NOTE: no torch.no_grad() here — NNST internally optimizes
        # Laplacian pyramid coefficients with Adam and needs gradients.
        output = self._produce_stylization(
            content,
            style,
            self._phi,
            max_iter=self.max_iter,
            lr=2e-3,
            content_weight=self.content_weight,
            max_scls=self.max_scls,
            flip_aug=self.flip_aug,
            content_loss=False,
            dont_colorize=False,
        )

        return output.clamp(0.0, 1.0)

    def style_hash(self) -> str:
        """Hash of style images + NNST parameters for cache invalidation."""
        h = hashlib.sha256()
        for sp in self.style_paths:
            with open(sp, "rb") as f:
                h.update(f.read())
        h.update(f"{self.max_iter},{self.content_weight},{self.max_scls},{self.flip_aug}".encode())
        return h.hexdigest()[:12]

    def cleanup(self):
        """Free GPU memory used by NNST models."""
        del self._vgg
        del self._style_tensors
        self._style_tensors = []
        torch.cuda.empty_cache()
