"""Optical flow I/O, warping, and occlusion mask utilities.

Based on ReCoNet/utils.py and ReCoNet/IO.py.
"""

import re

import numpy as np
import torch
import torch.nn.functional as F


def read_pfm(path: str) -> tuple[np.ndarray, float]:
    """Read Portable Float Map file."""
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {path}")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("ascii"))
        if not dim_match:
            raise ValueError(f"Malformed PFM header: {path}")
        width, height = map(int, dim_match.groups())

        scale = float(f.readline().decode("ascii").rstrip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale


def read_flow(path: str) -> np.ndarray:
    """Read optical flow file (.flo or .pfm). Returns [H,W,2] float32."""
    if path.endswith(".pfm") or path.endswith(".PFM"):
        return read_pfm(path)[0][:, :, 0:2]

    with open(path, "rb") as f:
        header = f.read(4)
        if header.decode("utf-8") != "PIEH":
            raise ValueError(f"Flow file header does not contain PIEH: {path}")

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        flow = np.fromfile(f, np.float32, width * height * 2).reshape(
            (height, width, 2)
        )
    return flow.astype(np.float32)


def warp_optical_flow(source: torch.Tensor, reverse_flow: torch.Tensor) -> torch.Tensor:
    """Warp source using reverse optical flow via grid_sample.

    source: [N,C,H,W]
    reverse_flow: [N,H,W,2] in pixel coordinates (NHWC)
    """
    device = source.device
    n, h, w, _ = reverse_flow.shape

    grid = reverse_flow.clone()
    grid[..., 0] += torch.arange(w, device=device).view(1, 1, w)
    grid[..., 0] = grid[..., 0] * 2 / w - 1
    grid[..., 1] += torch.arange(h, device=device).view(1, h, 1)
    grid[..., 1] = grid[..., 1] * 2 / h - 1

    return F.grid_sample(source, grid, padding_mode="border", align_corners=False)


def resize_optical_flow(
    optical_flow: torch.Tensor, h: int, w: int
) -> torch.Tensor:
    """Resize optical flow and scale magnitude proportionally.

    optical_flow: [N,H,W,2] NHWC -> resized to [N,h,w,2]
    """
    flow_nchw = optical_flow.permute(0, 3, 1, 2)  # NHWC -> NCHW
    old_h, old_w = flow_nchw.shape[-2:]
    flow_resized = F.interpolate(flow_nchw, size=(h, w), mode="bilinear", align_corners=False)
    flow_resized = flow_resized.permute(0, 2, 3, 1)  # NCHW -> NHWC

    h_scale, w_scale = h / old_h, w / old_w
    flow_resized[..., 0] *= w_scale
    flow_resized[..., 1] *= h_scale
    return flow_resized


def _magnitude_squared(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum(-1)


def occlusion_mask_from_flow(
    optical_flow: torch.Tensor,
    reverse_optical_flow: torch.Tensor,
    motion_boundaries: torch.Tensor,
) -> torch.Tensor:
    """Compute occlusion mask from forward-backward flow consistency.

    optical_flow: [N,H,W,2] forward flow (NHWC)
    reverse_optical_flow: [N,H,W,2] backward flow (NHWC)
    motion_boundaries: [N,H,W] bool tensor

    Returns: [N,1,H,W] float32 mask (1 = valid, 0 = occluded)
    """
    # Warp forward flow using reverse flow
    flow_nchw = optical_flow.permute(0, 3, 1, 2)
    warped_flow_nchw = warp_optical_flow(flow_nchw, reverse_optical_flow)
    warped_flow = warped_flow_nchw.permute(0, 2, 3, 1)

    # Forward-backward consistency check
    forward_mag = _magnitude_squared(warped_flow)
    reverse_mag = _magnitude_squared(reverse_optical_flow)
    sum_mag = _magnitude_squared(warped_flow + reverse_optical_flow)

    mask = sum_mag < (0.01 * (forward_mag + reverse_mag) + 0.5)
    mask = mask & (~motion_boundaries)
    return mask.to(torch.float32).unsqueeze(1)
