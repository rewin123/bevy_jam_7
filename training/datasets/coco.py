"""COCO2017 flat-folder dataset for single-frame style transfer training.

Based on MicroAST/train_microAST.py FlatFolderDataset.
"""

from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCODataset(Dataset):
    """Flat folder of images. Returns [3,H,W] tensor in [0,1]."""

    def __init__(self, root_dir: str, resolution: tuple[int, int] = (256, 256)):
        """
        Args:
            root_dir: Path to directory containing images.
            resolution: (H, W) for output size.
        """
        self.paths = sorted(
            p
            for p in Path(root_dir).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root_dir}")

        h, w = resolution
        self.transform = transforms.Compose(
            [
                transforms.Resize(max(h, w)),
                transforms.CenterCrop((h, w)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index: int):
        img = Image.open(str(self.paths[index])).convert("RGB")
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.paths)
