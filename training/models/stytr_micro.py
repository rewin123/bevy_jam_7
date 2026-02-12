"""StyTR-Micro — lightweight transformer-based style transfer model.

Inspired by StyTR-2 (diyiiyiii/StyTR-2): content-aware positional encoding
(CAPE) and multi-head self-attention in a compact SequenceStyleModel.

Unlike StyTR-2 which uses a full VGG encoder + separate style/content
transformer encoders + heavy decoder (~30M+ params), this model packs
the key ideas into <100K parameters:
  - CAPE: positional encoding derived from content features (not fixed)
  - Multi-head ReLU linear attention: O(N·d²) global self-attention
  - Transformer blocks: LN → MHLA → LN → FFN (standard pre-norm)
  - InvertedBottleneck for local receptive field

Parameter breakdown:
  Encoder:      4,224  (strided conv + depthwise-separable)
  CAPE:         2,304  (adaptive pool → 1×1 conv → upsample)
  Transformer: 37,248  (2 × [LN + MHLA + LN + FFN])
  Residual:    15,792  (1× InvertedBottleneck, dilation=2)
  Decoder:     11,268  (transposed conv + refine)
  Total:      ~70,836
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SequenceStyleModel
from .model5 import InvertedBottleneck


class ContentAwarePosEncoding(nn.Module):
    """CAPE: derive positional encoding from content features.

    Pools features to a small spatial grid, projects with a learned 1×1
    convolution, then bilinearly upsamples back to full feature resolution.
    This produces position-dependent embeddings that adapt to the input
    content — the key idea from StyTR-2.

    Params: channels² (e.g. 48²=2304)
    """

    def __init__(self, channels: int, pool_size: int = 8):
        super().__init__()
        self.pool_size = pool_size
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = F.adaptive_avg_pool2d(x, self.pool_size)
        pos = self.proj(pos)
        pos = F.interpolate(
            pos, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return pos


class MultiHeadLinearAttention(nn.Module):
    """Multi-head ReLU linear attention.

    Following StyTR-2: Q and K are projected from (x + pos), V from x alone.
    Uses ReLU kernel for O(N·d²) complexity via the associativity trick:
    Q·(K^T·V) instead of (Q·K^T)·V.

    Params: channels × (2·channels) + channels × channels + channels × channels
          = 4 × channels²  (e.g. 4×48²=9216)
    """

    def __init__(self, channels: int, num_heads: int = 4, eps: float = 1e-6):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.eps = eps

        # Q, K from (x + pos); V from x — split projections
        self.qk_proj = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # StyTR-2: pos added to Q/K input, not to V
        x_pos = x + pos

        qk = self.qk_proj(x_pos).reshape(B, 2, self.num_heads, self.head_dim, N)
        q, k = qk[:, 0], qk[:, 1]  # (B, heads, d, N)
        v = self.v_proj(x).reshape(B, self.num_heads, self.head_dim, N)

        # ReLU kernel (non-negative → valid attention weights)
        q = F.relu(q)
        k = F.relu(k)

        # Linear attention: (V_pad @ K^T) @ Q
        v_pad = F.pad(v, (0, 0, 0, 1), value=1.0)  # (B, heads, d+1, N)
        kv = v_pad @ k.transpose(-1, -2)  # (B, heads, d+1, d)
        out = kv @ q  # (B, heads, d+1, N)

        # Normalize by sum of attention weights (last row)
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)  # (B, heads, d, N)

        out = out.reshape(B, C, H, W)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer: LN + MHLA + residual → LN + FFN + residual.

    Uses GroupNorm(1, C) as channel-wise LayerNorm (equivalent for [B,C,H,W]).
    FFN expansion ratio 2× keeps params low.
    """

    def __init__(self, channels: int, num_heads: int = 4, ffn_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.attn = MultiHeadLinearAttention(channels, num_heads)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * ffn_ratio, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * ffn_ratio, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.ffn(self.norm2(x))
        return x


class StyTRMicro(SequenceStyleModel):
    """Lightweight transformer-based style transfer.

    Architecture:
      Encoder:     CNN 3→48ch @ 1/4 resolution
      CAPE:        Content-aware positional encoding (StyTR-2)
      Transformer: 2 × [pre-norm MHLA + FFN] with 4-head linear attention
      Residual:    1× InvertedBottleneck(48→144→48, dilation=2) for local RF
      Decoder:     Progressive upsampling CNN → sigmoid

    encode() returns 48-channel feature map at 1/4 resolution.
    decode() reconstructs [0,1] image.
    """

    def __init__(self):
        super().__init__()

        # --- Encoder: 3→48ch @ 1/4 resolution ---
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 32, kernel_size=5, stride=2, bias=False),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, groups=32, bias=False),
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
        )

        # --- Content-Aware Positional Encoding (StyTR-2) ---
        self.cape = ContentAwarePosEncoding(48, pool_size=8)

        # --- Transformer: 2 layers, 4 heads, FFN ratio 2 ---
        self.transformer = nn.ModuleList(
            [TransformerLayer(48, num_heads=4, ffn_ratio=2) for _ in range(2)]
        )

        # --- Local texture: 1× InvertedBottleneck (dilation=2 for RF) ---
        self.residual = InvertedBottleneck(48, 144, dilation=2)

        # --- Decoder: 48→3ch @ full resolution ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2, bias=False),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(24, 24, kernel_size=3, bias=False),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2, bias=False),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(12, 3, kernel_size=3, bias=False),
        )

    def train_resolution(self) -> tuple[int, int]:
        return (288, 512)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        pos = self.cape(x)
        for layer in self.transformer:
            x = layer(x, pos)
        return self.residual(x)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(features))
