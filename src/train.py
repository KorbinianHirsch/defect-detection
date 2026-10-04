"""Fit a PatchCore memory bank on train/good and save it to disk.

Usage:
    python -m src.train --category bottle
"""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.models.patchcore import PatchCore
from src.utils.dataset import MVTecTrainDataset

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--coreset-ratio", type=float, default=0.02)
    args = parser.parse_args()

    dataset = MVTecTrainDataset(args.category, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = PatchCore(coreset_ratio=args.coreset_ratio)
    print(f"Extracting features from {len(dataset)} training images on {model.device} ...")
    model.fit(loader)

    threshold = model.compute_threshold(loader)
    print(f"Anomaly threshold: {threshold:.2f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / f"{args.category}_patchcore.pt"
    model.save(out_path)
    print(f"Saved memory bank to {out_path}")


if __name__ == "__main__":
    main()
