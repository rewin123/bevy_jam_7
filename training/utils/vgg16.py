"""Frozen VGG16 feature extractor for perceptual losses.

Extracts features at relu1_2, relu2_2, relu3_3, relu4_3
(torchvision VGG16 feature indices 3, 8, 15, 22).

Based on ReCoNet/vgg.py.
"""

import torch
import torch.nn as nn


class VGG16Features(nn.Module):
    """Frozen VGG16 feature extractor.

    Returns list of 4 feature maps at relu1_2, relu2_2, relu3_3, relu4_3.
    Input must be ImageNet-normalized (use preprocess_for_vgg from losses.py).
    """

    LAYERS_OF_INTEREST = {3, 8, 15, 22}

    def __init__(self, weights_path: str | None = None):
        super().__init__()

        if weights_path:
            from torchvision.models import vgg16

            model = vgg16(weights=None)
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
        else:
            from torchvision.models import vgg16, VGG16_Weights

            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        features = list(model.features)[:23]  # up to relu4_3
        self.layers = nn.ModuleList(features)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        results = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.LAYERS_OF_INTEREST:
                results.append(x)
        return results
