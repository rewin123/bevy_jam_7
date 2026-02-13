"""Model5Seq — sequence-aware variant of Model5 for temporal coherence.

Same architecture as Model5 but extends SequenceStyleModel with explicit
encode/decode split for temporal loss training, plus split-channel
spatial self-attention before the inverted bottleneck residual block.

Parameter breakdown:
  Encoder:    2,000  (same as Model5)
  Attention:    256  (ReLU linear attention on 16 of 32 channels)
  Residual:  44,544  (3× InvertedBottleneck 32→192→32, dilation 1/2/4)
  Decoder:    3,235  (same as Model5)
  Total:    ~50,035
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SequenceStyleModel
from .model5 import InvertedBottleneck


class LinearSpatialAttention(nn.Module):
    """ReLU Linear Attention (EfficientViT-style).

    True spatial self-attention with O(N·d²) complexity instead of O(N²·d).
    Uses ReLU as the kernel function and exploits matrix multiplication
    associativity: Q·(K^T·V) instead of (Q·K^T)·V.

    For in_channels=16, dim=4: 256 parameters, ~0.4ms on GPU at 128×128.
    """

    def __init__(self, in_channels: int, dim: int = 4, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.qkv = nn.Conv2d(in_channels, 3 * dim, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # Project to Q, K, V: each (B, dim, N)
        qkv = self.qkv(x).reshape(B, 3, self.dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # ReLU kernel (non-negative, required for valid attention weights)
        q = F.relu(q)
        k = F.relu(k)

        # Linear attention: compute (V @ K^T) @ Q instead of (Q @ K^T) @ V
        # KV matrix is only (dim × dim) — tiny regardless of spatial size
        v_pad = F.pad(v, (0, 0, 0, 1), value=1.0)  # (B, dim+1, N)
        kv = v_pad @ k.transpose(-1, -2)             # (B, dim+1, dim)
        out = kv @ q                                  # (B, dim+1, N)

        # Normalize by sum of attention weights (last row of padded output)
        out = out[:, :-1] / (out[:, -1:] + self.eps)  # (B, dim, N)

        out = self.proj(out.reshape(B, self.dim, H, W))
        return out + x


class Model5Seq(SequenceStyleModel):
    """Model5 architecture with split-channel spatial attention.

    encode() returns 32-channel feature map at 1/4 resolution.
    decode() reconstructs [0,1] image via transposed convolutions + sigmoid.

    Before the InvertedBottleneck block, the first 16 channels pass through
    ReLU linear self-attention while the other 16 skip unchanged.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 32, kernel_size=5, stride=2, bias=False),
            # nn.InstanceNorm2d(16, affine=True),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, groups=16, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            # nn.InstanceNorm2d(32, affine=True),
            nn.SiLU(inplace=True),
        )

        self.attn = LinearSpatialAttention(in_channels=16, dim=4)
        # Dilated residual stack: dilation 1, 2, 4 for large receptive field
        self.residual = nn.Sequential(
            InvertedBottleneck(64, 192, dilation=1),
            InvertedBottleneck(64, 192, dilation=2),
            InvertedBottleneck(64, 192, dilation=4),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.SiLU(inplace=True),

            nn.ReflectionPad2d(2),
            nn.Conv2d(32, 32, kernel_size=5, bias=False),
            nn.SiLU(inplace=True),

            # nn.InstanceNorm2d(16, affine=True),
            # nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, bias=False),
            # nn.InstanceNorm2d(8, affine=True),
            # nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),

            nn.ReflectionPad2d(2),
            nn.Conv2d(16, 16, kernel_size=5, bias=False),
            nn.PReLU(16),

            # nn.ReflectionPad2d(2),
            nn.Conv2d(16, 3, kernel_size=1, bias=False),
        )

    def train_resolution(self) -> tuple[int, int]:
        return (288, 512)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.residual(x)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(features))
