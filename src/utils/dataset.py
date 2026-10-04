"""PyTorch Datasets for MVTec AD.

Train split (`train/good`) is unlabeled -- self-supervised setup, only
normal samples. Test split (`test/*`) carries a good/anomaly label per
image and, for anomalous images, a pixel-level ground-truth mask.
"""

from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_transform(image_size: int = 256) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def default_mask_transform(image_size: int = 256) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )


class MVTecTrainDataset(Dataset):
    """Only `train/good` images. Returns just the image tensor (no label)."""

    def __init__(
        self,
        category: str,
        data_dir: Path = DATA_DIR,
        image_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(data_dir) / category / "train" / "good"
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} not found. Did you run "
                f"`python -m src.utils.download_mvtec --category {category}`?"
            )
        self.image_paths = sorted(self.root.glob("*.png"))
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.root}")
        self.transform = transform or default_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


class MVTecTestDataset(Dataset):
    """`test/*` images with label (0=good, 1=anomaly) and ground-truth mask.

    Each item is a dict:
        image:       Tensor [3, H, W], normalized
        label:       int, 0 (good) or 1 (anomaly)
        mask:        Tensor [1, H, W], binary (all zeros for good images)
        defect_type: str, e.g. "good", "broken_large", ...
        path:        str, original image path
    """

    def __init__(
        self,
        category: str,
        data_dir: Path = DATA_DIR,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        self.category_dir = Path(data_dir) / category
        self.test_dir = self.category_dir / "test"
        self.gt_dir = self.category_dir / "ground_truth"
        if not self.test_dir.exists():
            raise FileNotFoundError(
                f"{self.test_dir} not found. Did you run "
                f"`python -m src.utils.download_mvtec --category {category}`?"
            )

        self.image_size = image_size
        self.transform = transform or default_transform(image_size)
        self.mask_transform = mask_transform or default_mask_transform(image_size)

        self.samples = []  # (image_path, defect_type, mask_path | None, label)
        for defect_dir in sorted(p for p in self.test_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if label == 1:
                    candidate = self.gt_dir / defect_type / f"{image_path.stem}_mask.png"
                    mask_path = candidate if candidate.exists() else None
                self.samples.append((image_path, defect_type, mask_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {self.test_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, defect_type, mask_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros((1, self.image_size, self.image_size))

        return {
            "image": image,
            "label": label,
            "mask": mask,
            "defect_type": defect_type,
            "path": str(image_path),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the MVTec AD datasets.")
    parser.add_argument("--category", default="bottle")
    args = parser.parse_args()

    train_ds = MVTecTrainDataset(args.category)
    test_ds = MVTecTestDataset(args.category)

    print(f"train/good: {len(train_ds)} images, sample shape {train_ds[0].shape}")
    print(f"test:       {len(test_ds)} images")

    defect_types = sorted({s[1] for s in test_ds.samples})
    print(f"defect types: {defect_types}")

    sample = test_ds[len(test_ds) // 2]
    print(
        f"sample -> label={sample['label']}, defect_type={sample['defect_type']}, "
        f"image={tuple(sample['image'].shape)}, mask={tuple(sample['mask'].shape)}, "
        f"mask_sum={sample['mask'].sum().item()}"
    )
