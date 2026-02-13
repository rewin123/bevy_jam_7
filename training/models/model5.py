"""Model 5 from 'Design and experimental research of on-device style transfer
models for mobile environments' (Hwang & Oh, 2025, Table 5).

20,083 parameters. Single inverted bottleneck residual block.
Uses depthwise separable convolutions, ConvTranspose2d(kernel=2) for upsampling.
BN in encoder/decoder, IN in residual block. ReLU6 throughout.

Param breakdown (from paper Table 5):
  Encoder:  1216 + 32 + 144 + 544 + 64           = 2,000
  Residual: 6144 + 384 + 1728 + 384 + 6144 + 64  = 14,848
  Decoder:  2064 + 32 + 520 + 16 + 603            = 3,235
  Total:                                           = 20,083
"""

import torch
import torch.nn as nn

from .base import SingleFrameStyleModel


class InvertedBottleneck(nn.Module):
    """Inverted bottleneck: channels -> expanded -> channels with skip connection.

    Expansion via 1x1 conv, depthwise 3x3 conv, compression via 1x1 conv.
    InstanceNorm + ReLU6 after expansion and depthwise. IN + skip after compression.
    All convs bias=False (norm layers handle bias).
    """

    def __init__(self, channels: int, expanded: int, dilation: int = 1):
        super().__init__()
        pad = dilation  # ReflectionPad = dilation for 3x3 dilated conv
        self.block = nn.Sequential(
            # Expand: 32->192, 6144 params
            nn.Conv2d(channels, expanded, 1, bias=False),
            # nn.InstanceNorm2d(expanded, affine=True),  # 384 params
            nn.SiLU(inplace=True),
            # Depthwise: 192 dw, 1728 params (dilation increases RF, 0 extra params)
            nn.ReflectionPad2d(pad),
            nn.Conv2d(expanded, expanded, 3, groups=expanded, bias=False,
                      dilation=dilation),
            # nn.InstanceNorm2d(expanded, affine=True),  # 384 params
            nn.SiLU(inplace=True),
            # Compress: 192->32, 6144 params
            nn.Conv2d(expanded, channels, 1, bias=False),
            # nn.InstanceNorm2d(channels, affine=True),  # 64 params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Model5(SingleFrameStyleModel):
    """Paper Table 5: 20,083 parameters.

    Encoder:
      RP(2) + Conv(3->16, 5x5, s=2, bias) + BN + ReLU6      [1216+32]
      RP(1) + DWConv(16, 3x3, s=2) + PWConv(16->32, bias) + BN + ReLU6  [144+544+64]

    Residual (single inverted bottleneck):
      Conv1x1(32->192) + IN + ReLU6     [6144+384]
      RP(1) + DWConv(192, 3x3) + IN + ReLU6  [1728+384]
      Conv1x1(192->32) + IN + skip      [6144+64]

    Decoder:
      ConvTranspose2d(32->16, k=2, s=2, bias) + BN + ReLU6  [2064+32]
      ConvTranspose2d(16->8, k=2, s=2, bias) + BN + ReLU6   [520+16]
      RP(2) + Conv(8->3, 5x5, bias)                          [603]
      Sigmoid
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Conv 3->16, 5x5, stride=2: 3*16*25+16 = 1216
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),  # 32
            nn.ReLU6(inplace=True),
            # Depthwise 16, 3x3, stride=2: 16*9 = 144
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, groups=16, bias=False),
            # Pointwise 16->32: 16*32+32 = 544
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),  # 64
            nn.ReLU6(inplace=True),
        )

        # Residual block (single inverted bottleneck): 14,848
        self.residual = InvertedBottleneck(32, 192)

        # Decoder
        self.decoder = nn.Sequential(
            # ConvTranspose2d 32->16, k=2, s=2: 32*16*4+16 = 2064
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),  # 32
            nn.ReLU6(inplace=True),
            # ConvTranspose2d 16->8, k=2, s=2: 16*8*4+8 = 520
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),  # 16
            nn.ReLU6(inplace=True),
            # Conv 8->3, 5x5: 8*3*25+3 = 603
            nn.ReflectionPad2d(2),
            nn.Conv2d(8, 3, kernel_size=5),
        )

    def train_resolution(self) -> tuple[int, int]:
        return (288, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return torch.sigmoid(x)
