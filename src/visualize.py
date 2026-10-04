"""Save example anomaly heatmaps for a fitted PatchCore model.

For each defect type (including "good"), saves a few original / predicted
heatmap / ground-truth panels to data/processed/visualizations/<category>/.

Usage:
    python -m src.visualize --category bottle
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.models.patchcore import PatchCore
from src.utils.dataset import MVTecTestDataset

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_display_image(path: str, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    return np.asarray(img) / 255.0


def overlay_heatmap(
    image: np.ndarray, anomaly_map: np.ndarray, vmin: float, vmax: float, alpha: float = 0.5
) -> np.ndarray:
    """Blend a colormap over `image`. vmin/vmax should be shared across a set of
    images being compared -- per-image min-max normalization would make every
    single image (including defect-free ones) look equally "hot"."""
    norm = np.clip((anomaly_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heatmap = plt.cm.jet(norm)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heatmap, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--per-type", type=int, default=2, help="examples per defect type")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint or (ARTIFACT_DIR / f"{args.category}_patchcore.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found. Run `python -m src.train --category {args.category}` first."
        )

    dataset = MVTecTestDataset(args.category, image_size=args.image_size)
    model = PatchCore()
    model.load(checkpoint)

    out_dir = ARTIFACT_DIR / "visualizations" / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    by_type = {}
    for i, (_, defect_type, _, _) in enumerate(dataset.samples):
        by_type.setdefault(defect_type, []).append(i)

    selected = [
        idx for indices in by_type.values() for idx in indices[: args.per_type]
    ]

    # Run inference once, up front, so all panels share one color scale.
    results = {}
    for idx in selected:
        sample = dataset[idx]
        out = model.predict(sample["image"].unsqueeze(0))
        results[idx] = {
            "sample": sample,
            "anomaly_map": out["anomaly_maps"][0].numpy(),
            "score": out["scores"][0].item(),
        }

    all_maps = np.stack([r["anomaly_map"] for r in results.values()])
    vmin, vmax = float(all_maps.min()), float(all_maps.max())

    for defect_type, indices in by_type.items():
        for n, idx in enumerate(indices[: args.per_type]):
            sample = results[idx]["sample"]
            anomaly_map = results[idx]["anomaly_map"]
            score = results[idx]["score"]

            display_img = load_display_image(sample["path"], args.image_size)
            overlay = overlay_heatmap(display_img, anomaly_map, vmin, vmax)
            gt_mask = sample["mask"][0].numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(display_img)
            axes[0].set_title("Original")
            axes[1].imshow(overlay)
            axes[1].set_title(f"Predicted (score={score:.2f})")
            axes[2].imshow(gt_mask, cmap="gray")
            axes[2].set_title("Ground truth")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{args.category} / {defect_type}")
            fig.tight_layout()

            out_path = out_dir / f"{defect_type}_{n}.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
