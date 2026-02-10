from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ImageSequenceDesc:
    """Describes training data requirements for a model."""

    num_frames: int  # 1 = single image, 2 = consecutive pair
    resolution: tuple[int, int]  # (H, W) training resolution
    needs_optical_flow: bool = False
    needs_motion_boundaries: bool = False


class StyleModel(nn.Module, ABC):
    """Root ABC. All style transfer models share these contracts.

    I/O convention: [B, 3, H, W] float32 in [0, 1].
    """

    @abstractmethod
    def image_sequence_needs(self) -> ImageSequenceDesc:
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] float32 in [0,1] -> [B,3,H,W] float32 in [0,1]."""
        ...

    @abstractmethod
    def inference_frame(
        self, img: torch.Tensor, state: Any
    ) -> tuple[torch.Tensor, Any]:
        """Streaming inference. img=[1,3,H,W]. state=None on first call.
        Returns (styled_img, new_state)."""
        ...


class SingleFrameStyleModel(StyleModel, ABC):
    """Single-frame models process each image independently.

    Subclasses implement forward() and train_resolution() only.
    image_sequence_needs() and inference_frame() are auto-derived.
    """

    @abstractmethod
    def train_resolution(self) -> tuple[int, int]:
        """(H, W) for training crops."""
        ...

    def image_sequence_needs(self) -> ImageSequenceDesc:
        return ImageSequenceDesc(
            num_frames=1,
            resolution=self.train_resolution(),
        )

    def inference_frame(self, img: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        with torch.no_grad():
            return self.forward(img), None


class SequenceStyleModel(StyleModel, ABC):
    """Sequence-frame models with explicit encode/decode split.

    Required for temporal losses (which need encoder features of both frames).
    Subclasses implement encode(), decode(), train_resolution().
    forward() = decode(encode(x)) is auto-derived.
    inference_frame() carries {prev_features, prev_output} in state.
    """

    @abstractmethod
    def train_resolution(self) -> tuple[int, int]:
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder forward. Returns feature tensor used by temporal loss."""
        ...

    @abstractmethod
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decoder forward. Returns styled image in [0,1]."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def image_sequence_needs(self) -> ImageSequenceDesc:
        return ImageSequenceDesc(
            num_frames=2,
            resolution=self.train_resolution(),
            needs_optical_flow=True,
            needs_motion_boundaries=True,
        )

    def inference_frame(self, img: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        with torch.no_grad():
            features = self.encode(img)
            output = self.decode(features)
        new_state = {"prev_features": features, "prev_output": output}
        return output, new_state
