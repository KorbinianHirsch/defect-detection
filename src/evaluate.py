"""Evaluate a fitted PatchCore model on the MVTec AD test split.

Reports image-level and pixel-level ROC-AUC.

Usage:
    python -m src.evaluate --category bottle
"""

import argparse
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.models.patchcore import PatchCore
from src.utils.dataset import MVTecTestDataset

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint or (ARTIFACT_DIR / f"{args.category}_patchcore.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found. Run `python -m src.train --category {args.category}` first."
        )

    dataset = MVTecTestDataset(args.category, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = PatchCore()
    model.load(checkpoint)

    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []

    for batch in loader:
        out = model.predict(batch["image"])
        image_scores.append(out["scores"])
        image_labels.append(batch["label"])
        pixel_scores.append(out["anomaly_maps"].flatten(1))
        pixel_labels.append(batch["mask"].flatten(1))

    image_scores = torch.cat(image_scores).numpy()
    image_labels = torch.cat(image_labels).numpy()
    pixel_scores = torch.cat(pixel_scores).numpy().ravel()
    pixel_labels = torch.cat(pixel_labels).numpy().ravel().astype(int)

    image_auroc = roc_auc_score(image_labels, image_scores)
    pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)

    print(f"Category:          {args.category}")
    print(f"Test images:       {len(image_labels)}")
    print(f"Image-level AUROC: {image_auroc:.4f}")
    print(f"Pixel-level AUROC: {pixel_auroc:.4f}")


if __name__ == "__main__":
    main()
