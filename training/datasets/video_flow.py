"""Video + optical flow datasets for sequence-frame style transfer training.

Supports MPI Sintel and FlyingChairs datasets.
Based on ReCoNet/dataset.py and MicroAST/sampler.py.
"""

import os
from collections import namedtuple

import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset

from utils.optical_flow import read_flow

FramePairEntry = namedtuple(
    "FramePairEntry",
    ("frame", "previous_frame", "optical_flow", "reverse_optical_flow", "motion_boundaries"),
)


class SintelDataset(Dataset):
    """MPI Sintel dataset: consecutive frame pairs + forward/backward optical flow.

    Directory structure:
      {root}/training/clean/{scene}/*.png      (or final/)
      {root}/training/flow/{scene}/frame_XXXX.flo
      {root}/training/occlusions/{scene}/frame_XXXX.png
    """

    def __init__(self, root_dir: str, transform=None, pass_type: str = "clean"):
        self.transform = transform
        self.entries: list[FramePairEntry] = []

        frames_dir = os.path.join(root_dir, "training", pass_type)
        flow_dir = os.path.join(root_dir, "training", "flow")
        occ_dir = os.path.join(root_dir, "training", "occlusions")

        if not os.path.isdir(frames_dir):
            return

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

                # Flow files: frame_XXXX.flo (forward flow from frame i-1 to i)
                frame_num_prev = os.path.splitext(frame_files[i - 1])[0]
                frame_num_curr = os.path.splitext(frame_files[i])[0]

                fwd_flow_path = os.path.join(flow_dir, scene, f"{frame_num_prev}.flo")
                # Sintel only has forward flow; we use it as both directions
                # and approximate occlusion from the occlusion maps
                rev_flow_path = fwd_flow_path  # will negate at load time
                occ_path = os.path.join(occ_dir, scene, f"{frame_num_curr}.png")

                if os.path.exists(fwd_flow_path):
                    self.entries.append(
                        FramePairEntry(
                            frame_path,
                            prev_frame_path,
                            fwd_flow_path,
                            fwd_flow_path,  # Sintel only provides forward flow
                            occ_path if os.path.exists(occ_path) else None,
                        )
                    )

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        flow = read_flow(entry.optical_flow).copy()

        # For reverse flow, negate the forward flow as approximation
        rev_flow = -flow.copy()

        # Load occlusion/motion boundary mask
        if entry.motion_boundaries and os.path.exists(entry.motion_boundaries):
            mb = Image.open(entry.motion_boundaries)
        else:
            # Create empty motion boundaries if not available
            h, w = flow.shape[:2]
            mb = Image.new("L", (w, h), 0)

        sample = {
            "frame": Image.open(entry.frame).convert("RGB"),
            "previous_frame": Image.open(entry.previous_frame).convert("RGB"),
            "optical_flow": flow,
            "reverse_optical_flow": rev_flow,
            "motion_boundaries": mb,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.entries)


class FlyingChairsDataset(Dataset):
    """FlyingChairs dataset: image pairs + forward optical flow.

    Directory structure:
      {root}/data/{XXXXX}_img1.ppm
      {root}/data/{XXXXX}_img2.ppm
      {root}/data/{XXXXX}_flow.flo
    """

    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.entries: list[FramePairEntry] = []

        data_dir = os.path.join(root_dir, "data")
        if not os.path.isdir(data_dir):
            return

        # Find all flow files and derive image pairs
        flow_files = sorted(f for f in os.listdir(data_dir) if f.endswith("_flow.flo"))

        for flow_file in flow_files:
            prefix = flow_file.replace("_flow.flo", "")
            img1 = os.path.join(data_dir, f"{prefix}_img1.ppm")
            img2 = os.path.join(data_dir, f"{prefix}_img2.ppm")
            flow_path = os.path.join(data_dir, flow_file)

            if os.path.exists(img1) and os.path.exists(img2):
                self.entries.append(
                    FramePairEntry(
                        img2,       # current frame
                        img1,       # previous frame
                        flow_path,  # forward flow
                        flow_path,  # reverse flow (approximated by negation)
                        None,       # no motion boundaries
                    )
                )

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        flow = read_flow(entry.optical_flow).copy()
        rev_flow = -flow.copy()

        h, w = flow.shape[:2]
        mb = Image.new("L", (w, h), 0)  # no motion boundaries

        sample = {
            "frame": Image.open(entry.frame).convert("RGB"),
            "previous_frame": Image.open(entry.previous_frame).convert("RGB"),
            "optical_flow": flow,
            "reverse_optical_flow": rev_flow,
            "motion_boundaries": mb,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.entries)


def InfiniteSampler(n: int):
    """Generate infinite sequence of shuffled indices."""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """PyTorch sampler wrapper for infinite iteration."""

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self) -> int:
        return 2**31
