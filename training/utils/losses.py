"""Loss functions for style transfer training.

Single-frame losses:
  - content_loss: MSE between feature maps
  - style_loss: Gram matrix MSE across VGG layers
  - total_variation_loss: smoothness regularizer

Sequence-frame losses (additive):
  - output_temporal_loss: luminance-weighted warped output difference
  - feature_temporal_loss: warped encoder feature difference

Based on ReCoNet/train.py and ReCoNet/utils.py.
"""

import torch


def preprocess_for_vgg(images_batch: torch.Tensor) -> torch.Tensor:
    """ImageNet normalization for VGG input. Expects [B,3,H,W] in [0,1]."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=images_batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images_batch.device).view(1, 3, 1, 1)
    return (images_batch - mean) / std


def gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """Gram matrix for style loss. Input: [N,C,H,W]. Output: [N,C,C]."""
    n, c, h, w = feature_map.shape
    f = feature_map.reshape(n, c, h * w)
    return f.bmm(f.transpose(1, 2)) / (c * h * w)


def content_loss(output_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """MSE between feature maps (content preservation)."""
    n, c, h, w = output_features.shape
    return (output_features - target_features).pow(2).sum() / (c * h * w)


def style_loss(
    output_features_list: list[torch.Tensor],
    style_gram_matrices: list[torch.Tensor],
) -> torch.Tensor:
    """Sum of Gram matrix MSE across VGG layers."""
    loss = torch.tensor(0.0, device=output_features_list[0].device)
    for out_feat, style_gm in zip(output_features_list, style_gram_matrices):
        out_gm = gram_matrix(out_feat)
        loss = loss + (out_gm - style_gm).pow(2).sum()
    return loss


def total_variation_loss(y: torch.Tensor) -> torch.Tensor:
    """Total variation regularizer for smoothness."""
    return (
        torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))
        + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
    )


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    """RGB to luminance using ITU-R BT.709 coefficients. Input: [N,3,H,W]."""
    return x[:, 0] * 0.2126 + x[:, 1] * 0.7152 + x[:, 2] * 0.0722


def output_temporal_loss(
    input_frame: torch.Tensor,
    prev_input_frame: torch.Tensor,
    output_frame: torch.Tensor,
    prev_output_frame: torch.Tensor,
    reverse_flow: torch.Tensor,
    occlusion_mask: torch.Tensor,
) -> torch.Tensor:
    """Luminance-weighted warped output difference for temporal consistency.

    All frames in model's internal range. Flow is [N,H,W,2] NHWC.
    Occlusion mask is [N,1,H,W].
    """
    from .optical_flow import warp_optical_flow

    input_diff = input_frame - warp_optical_flow(prev_input_frame, reverse_flow)
    output_diff = output_frame - warp_optical_flow(prev_output_frame, reverse_flow)
    luminance = rgb_to_luminance(input_diff).unsqueeze(1)

    n, c, h, w = input_frame.shape
    return (occlusion_mask * (output_diff - luminance)).pow(2).sum() / (h * w)


def feature_temporal_loss(
    features: torch.Tensor,
    prev_features: torch.Tensor,
    reverse_flow: torch.Tensor,
    occlusion_mask: torch.Tensor,
) -> torch.Tensor:
    """Warped encoder feature difference for temporal consistency.

    Features: [N,C,h,w]. Flow: [N,H,W,2] (original resolution, will be resized).
    Occlusion mask: [N,1,H,W] (original resolution, will be resized).
    """
    from .optical_flow import warp_optical_flow, resize_optical_flow

    n, c, h, w = features.shape
    flow_resized = resize_optical_flow(reverse_flow, h, w)
    mask_resized = torch.nn.functional.interpolate(
        occlusion_mask, size=(h, w), mode="nearest"
    )

    diff = features - warp_optical_flow(prev_features, flow_resized)
    return (mask_resized * diff).pow(2).sum() / (c * h * w)
