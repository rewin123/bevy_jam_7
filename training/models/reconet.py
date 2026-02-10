"""ReCoNet: Real-time Coherent Video Style Transfer Network.

Ported from ReCoNet/model.py. ~1.68M parameters.
Architecture: Encoder(3->48->96->192, 4 ResBlocks) + Decoder(upsample+conv, tanh).
Uses InstanceNorm + ReLU, reflection padding throughout.

Extends SequenceStyleModel with explicit encode/decode split for temporal loss.
"""

import torch
import torch.nn as nn

from .base import SequenceStyleModel


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            ConvLayer(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.branch = nn.Sequential(
            ConvNormLayer(channels, channels, kernel_size, 1),
            ConvNormLayer(channels, channels, kernel_size, 1, activation=False),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.branch(x))


class ConvTanhLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReCoNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvNormLayer(3, 48, 9, 1),
            ConvNormLayer(48, 96, 3, 2),
            ConvNormLayer(96, 192, 3, 2),
            ResLayer(192, 3),
            ResLayer(192, 3),
            ResLayer(192, 3),
            ResLayer(192, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReCoNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvNormLayer(192, 96, 3, 1),
            nn.Upsample(scale_factor=2),
            ConvNormLayer(96, 48, 3, 1),
            ConvTanhLayer(48, 3, 9, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReCoNet(SequenceStyleModel):
    """ReCoNet wrapped in SequenceStyleModel interface.

    Internal convention: [-1, 1] (tanh output).
    External convention (StyleModel contract): [0, 1].
    encode/decode handle the conversion.
    """

    def __init__(self):
        super().__init__()
        self._encoder = ReCoNetEncoder()
        self._decoder = ReCoNetDecoder()

    def train_resolution(self) -> tuple[int, int]:
        return (360, 640)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # [0,1] -> [-1,1] for ReCoNet internal convention
        return self._encoder(x * 2 - 1)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        # [-1,1] tanh output -> [0,1] for StyleModel convention
        return (self._decoder(features) + 1) / 2
