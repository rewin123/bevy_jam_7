"""Model5Seq — sequence-aware variant of Model5 for temporal coherence.

Same architecture as Model5 (20,083 parameters, single inverted bottleneck)
but extends SequenceStyleModel with explicit encode/decode split for
temporal loss training (optical flow, feature/output temporal losses).
"""

import torch
import torch.nn as nn

from .base import SequenceStyleModel
from .model5 import InvertedBottleneck


class Model5Seq(SequenceStyleModel):
    """Model5 architecture wrapped in SequenceStyleModel interface.

    encode() returns 32-channel feature map at 1/4 resolution.
    decode() reconstructs [0,1] image via transposed convolutions + sigmoid.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, groups=16, bias=False),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.residual = InvertedBottleneck(32, 192)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(8, 3, kernel_size=5),
        )

    def train_resolution(self) -> tuple[int, int]:
        return (512, 512)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(features))
