"""Flow-aware data transforms for video + optical flow datasets.

Each transform receives and returns a dict with keys:
  frame, previous_frame, optical_flow, reverse_optical_flow, motion_boundaries

Based on ReCoNet/custom_transforms.py.
"""

import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class FlowAwareToTensor:
    """Convert PIL images to tensors, numpy flow to tensors, motion boundaries to bool."""

    _to_tensor = transforms.ToTensor()

    def __call__(self, sample: dict) -> dict:
        return {
            "frame": self._to_tensor(sample["frame"]),
            "previous_frame": self._to_tensor(sample["previous_frame"]),
            "optical_flow": torch.from_numpy(sample["optical_flow"]),
            "reverse_optical_flow": torch.from_numpy(sample["reverse_optical_flow"]),
            "motion_boundaries": torch.from_numpy(
                np.array(sample["motion_boundaries"]).astype(bool)
            ),
        }


class FlowAwareResize:
    """Resize images and scale flow magnitudes proportionally."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def _resize_image(self, image: Image.Image) -> Image.Image:
        return image.resize((self.width, self.height))

    def _resize_flow(self, flow: np.ndarray) -> np.ndarray:
        orig_h, orig_w = flow.shape[:2]
        resized = cv2.resize(flow, (self.width, self.height))
        w_scale = self.width / orig_w
        h_scale = self.height / orig_h
        resized[..., 0] *= w_scale
        resized[..., 1] *= h_scale
        return resized

    def __call__(self, sample: dict) -> dict:
        return {
            "frame": self._resize_image(sample["frame"]),
            "previous_frame": self._resize_image(sample["previous_frame"]),
            "optical_flow": self._resize_flow(sample["optical_flow"]),
            "reverse_optical_flow": self._resize_flow(sample["reverse_optical_flow"]),
            "motion_boundaries": self._resize_image(sample["motion_boundaries"]),
        }


class FlowAwareRandomHorizontalFlip:
    """Random horizontal flip of images + flow (negate horizontal component)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    @staticmethod
    def _flip_image(image: Image.Image) -> Image.Image:
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    @staticmethod
    def _flip_flow(flow: np.ndarray) -> np.ndarray:
        flow = np.flip(flow, axis=1).copy()
        flow[..., 0] *= -1
        return flow

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            return {
                "frame": self._flip_image(sample["frame"]),
                "previous_frame": self._flip_image(sample["previous_frame"]),
                "optical_flow": self._flip_flow(sample["optical_flow"]),
                "reverse_optical_flow": self._flip_flow(sample["reverse_optical_flow"]),
                "motion_boundaries": self._flip_image(sample["motion_boundaries"]),
            }
        return sample
