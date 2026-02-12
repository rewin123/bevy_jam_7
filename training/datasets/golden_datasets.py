"""Datasets that load content images paired with pre-computed NNST golden stylizations.

Golden images are stored as PNG files in a cache directory that mirrors the
source dataset directory structure.  Each dataset class loads both the original
content and its golden counterpart, applying identical transforms to both.

Classes:
    GoldenCOCODataset       — flat folder of images + golden pairs
    GoldenSintelDataset     — Sintel frame pairs + golden for frame_t
    GoldenFlyingChairsDataset — FlyingChairs pairs + golden for frame_t
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from utils.optical_flow import read_flow

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _golden_path(content_path: str, dataset_root: str, golden_root: str) -> str:
    """Map a content image path to its golden counterpart in the cache.

    Example:
        content_path = "data/coco2017/val2017/000000000139.jpg"
        dataset_root = "data/coco2017/val2017"
        golden_root  = "data/golden_cache/abc123/coco2017/val2017"
        => "data/golden_cache/abc123/coco2017/val2017/000000000139.png"
    """
    rel = os.path.relpath(content_path, dataset_root)
    base = os.path.splitext(rel)[0] + ".png"
    return os.path.join(golden_root, base)


# ---------------------------------------------------------------------------
# Static image dataset
# ---------------------------------------------------------------------------

class GoldenCOCODataset(Dataset):
    """Flat folder of images paired with golden stylizations.

    Returns dict: {"content": [3,H,W], "golden": [3,H,W]} both in [0,1].
    Random horizontal flip is applied identically to both images.
    """

    def __init__(
        self,
        root_dir: str,
        golden_dir: str,
        resolution: tuple[int, int] = (256, 256),
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.golden_dir = os.path.abspath(golden_dir)
        self.resolution = resolution

        h, w = resolution
        self._resize = transforms.Resize((h, w))
        self._to_tensor = transforms.ToTensor()

        # Collect paths that have a golden counterpart
        all_paths = sorted(
            p for p in Path(root_dir).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self.paths: list[Path] = []
        self.golden_paths: list[str] = []
        skipped = 0
        for p in all_paths:
            gp = _golden_path(str(p), self.root_dir, self.golden_dir)
            if os.path.exists(gp):
                self.paths.append(p)
                self.golden_paths.append(gp)
            else:
                skipped += 1
        if skipped > 0:
            print(f"GoldenCOCODataset: {skipped}/{len(all_paths)} images without golden, skipped")
        if not self.paths:
            raise FileNotFoundError(
                f"No golden pairs found (root={root_dir}, golden={golden_dir})"
            )

    def __getitem__(self, index: int) -> dict:
        content_img = Image.open(str(self.paths[index])).convert("RGB")
        golden_img = Image.open(self.golden_paths[index]).convert("RGB")

        # Exact resize content to training resolution (matches precompute)
        content_img = self._resize(content_img)
        # Golden is already at correct resolution from precompute — no resize

        # Random horizontal flip (same coin for both)
        if random.random() < 0.5:
            content_img = content_img.transpose(Image.FLIP_LEFT_RIGHT)
            golden_img = golden_img.transpose(Image.FLIP_LEFT_RIGHT)

        return {
            "content": self._to_tensor(content_img),
            "golden": self._to_tensor(golden_img),
        }

    def __len__(self) -> int:
        return len(self.paths)


# ---------------------------------------------------------------------------
# Flow-aware transforms that forward extra image keys (golden_frame)
# ---------------------------------------------------------------------------

_BASE_IMAGE_KEYS = {"frame", "previous_frame", "motion_boundaries"}
_BASE_FLOW_KEYS = {"optical_flow", "reverse_optical_flow"}
_EXTRA_IMAGE_KEYS = {"golden_frame"}


class GoldenFlowAwareResize:
    """Resize images and scale flow magnitudes. Handles golden_frame too."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def _resize_image(self, image: Image.Image) -> Image.Image:
        return image.resize((self.width, self.height))

    def _resize_flow(self, flow: np.ndarray) -> np.ndarray:
        import cv2
        orig_h, orig_w = flow.shape[:2]
        resized = cv2.resize(flow, (self.width, self.height))
        resized[..., 0] *= self.width / orig_w
        resized[..., 1] *= self.height / orig_h
        return resized

    def __call__(self, sample: dict) -> dict:
        out = {}
        for k, v in sample.items():
            if k in _BASE_IMAGE_KEYS or k in _EXTRA_IMAGE_KEYS:
                out[k] = self._resize_image(v)
            elif k in _BASE_FLOW_KEYS:
                out[k] = self._resize_flow(v)
            else:
                out[k] = v
        return out


class GoldenFlowAwareRandomHorizontalFlip:
    """Random horizontal flip for images + flow. Handles golden_frame too."""

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
            out = {}
            for k, v in sample.items():
                if k in _BASE_IMAGE_KEYS or k in _EXTRA_IMAGE_KEYS:
                    out[k] = self._flip_image(v)
                elif k in _BASE_FLOW_KEYS:
                    out[k] = self._flip_flow(v)
                else:
                    out[k] = v
            return out
        return sample


class GoldenFlowAwareToTensor:
    """Convert PIL images to tensors, numpy flow to tensors. Handles golden_frame."""

    _to_tensor = transforms.ToTensor()

    def __call__(self, sample: dict) -> dict:
        out = {}
        for k, v in sample.items():
            if k in _BASE_IMAGE_KEYS or k in _EXTRA_IMAGE_KEYS:
                if k == "motion_boundaries":
                    out[k] = torch.from_numpy(np.array(v).astype(bool))
                else:
                    out[k] = self._to_tensor(v)
            elif k in _BASE_FLOW_KEYS:
                out[k] = torch.from_numpy(v)
            else:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# Video datasets with golden
# ---------------------------------------------------------------------------

class GoldenSintelDataset(Dataset):
    """MPI Sintel frame pairs with golden stylization for frame_t.

    Returns dict with standard video keys + "golden_frame" [3,H,W].
    """

    def __init__(self, root_dir: str, golden_dir: str, transform=None,
                 pass_type: str = "clean"):
        from collections import namedtuple

        self.transform = transform
        self.root_dir = os.path.abspath(root_dir)
        self.golden_dir = os.path.abspath(golden_dir)

        FrameEntry = namedtuple(
            "FrameEntry",
            ("frame", "previous_frame", "optical_flow", "reverse_optical_flow",
             "motion_boundaries"),
        )
        self.entries = []

        frames_dir = os.path.join(root_dir, "training", pass_type)
        flow_dir = os.path.join(root_dir, "training", "flow")
        occ_dir = os.path.join(root_dir, "training", "occlusions")

        if not os.path.isdir(frames_dir):
            return

        skipped = 0
        for scene in sorted(os.listdir(frames_dir)):
            scene_frames = os.path.join(frames_dir, scene)
            if not os.path.isdir(scene_frames):
                continue

            frame_files = sorted(
                f for f in os.listdir(scene_frames) if f.endswith(".png")
            )

            for i in range(1, len(frame_files)):
                frame_path = os.path.join(scene_frames, frame_files[i])
                prev_frame_path = os.path.join(scene_frames, frame_files[i - 1])

                frame_num_prev = os.path.splitext(frame_files[i - 1])[0]
                frame_num_curr = os.path.splitext(frame_files[i])[0]

                fwd_flow_path = os.path.join(flow_dir, scene, f"{frame_num_prev}.flo")
                occ_path = os.path.join(occ_dir, scene, f"{frame_num_curr}.png")

                # Check golden exists for frame_t
                golden_path = _golden_path(frame_path, self.root_dir, self.golden_dir)

                if os.path.exists(fwd_flow_path) and os.path.exists(golden_path):
                    self.entries.append(FrameEntry(
                        frame_path,
                        prev_frame_path,
                        fwd_flow_path,
                        fwd_flow_path,
                        occ_path if os.path.exists(occ_path) else None,
                    ))
                elif os.path.exists(fwd_flow_path):
                    skipped += 1

        if skipped > 0:
            print(f"GoldenSintelDataset: {skipped} pairs without golden, skipped")

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        flow = read_flow(entry.optical_flow).copy()
        rev_flow = -flow.copy()

        if entry.motion_boundaries and os.path.exists(entry.motion_boundaries):
            mb = Image.open(entry.motion_boundaries)
        else:
            h, w = flow.shape[:2]
            mb = Image.new("L", (w, h), 0)

        # Load golden for frame_t
        golden_path = _golden_path(entry.frame, self.root_dir, self.golden_dir)
        golden_img = Image.open(golden_path).convert("RGB")

        sample = {
            "frame": Image.open(entry.frame).convert("RGB"),
            "previous_frame": Image.open(entry.previous_frame).convert("RGB"),
            "golden_frame": golden_img,
            "optical_flow": flow,
            "reverse_optical_flow": rev_flow,
            "motion_boundaries": mb,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.entries)


class GoldenFlyingChairsDataset(Dataset):
    """FlyingChairs frame pairs with golden stylization for frame_t.

    Returns dict with standard video keys + "golden_frame" [3,H,W].
    """

    def __init__(self, root_dir: str, golden_dir: str, transform=None):
        from collections import namedtuple

        self.transform = transform
        self.root_dir = os.path.abspath(root_dir)
        self.golden_dir = os.path.abspath(golden_dir)

        FrameEntry = namedtuple(
            "FrameEntry",
            ("frame", "previous_frame", "optical_flow", "reverse_optical_flow",
             "motion_boundaries"),
        )
        self.entries = []

        data_dir = os.path.join(root_dir, "data")
        if not os.path.isdir(data_dir):
            return

        flow_files = sorted(f for f in os.listdir(data_dir) if f.endswith("_flow.flo"))

        skipped = 0
        for flow_file in flow_files:
            prefix = flow_file.replace("_flow.flo", "")
            img1 = os.path.join(data_dir, f"{prefix}_img1.ppm")
            img2 = os.path.join(data_dir, f"{prefix}_img2.ppm")
            flow_path = os.path.join(data_dir, flow_file)

            if not (os.path.exists(img1) and os.path.exists(img2)):
                continue

            # frame_t is img2 in FlyingChairs convention
            golden_path = _golden_path(img2, self.root_dir, self.golden_dir)

            if os.path.exists(golden_path):
                self.entries.append(FrameEntry(
                    img2, img1, flow_path, flow_path, None,
                ))
            else:
                skipped += 1

        if skipped > 0:
            print(f"GoldenFlyingChairsDataset: {skipped} pairs without golden, skipped")

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        flow = read_flow(entry.optical_flow).copy()
        rev_flow = -flow.copy()

        h, w = flow.shape[:2]
        mb = Image.new("L", (w, h), 0)

        golden_path = _golden_path(entry.frame, self.root_dir, self.golden_dir)
        golden_img = Image.open(golden_path).convert("RGB")

        sample = {
            "frame": Image.open(entry.frame).convert("RGB"),
            "previous_frame": Image.open(entry.previous_frame).convert("RGB"),
            "golden_frame": golden_img,
            "optical_flow": flow,
            "reverse_optical_flow": rev_flow,
            "motion_boundaries": mb,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.entries)
