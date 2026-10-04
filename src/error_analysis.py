"""Per-category error analysis for a fitted PatchCore model.

Reports a confusion matrix at the trained threshold, per-defect-type
recall, and saves heatmap panels for the hardest failure cases: missed
defects (false negatives, lowest scores among true anomalies) and false
alarms (false positives, highest scores among true-good images).

Usage:
    python -m src.error_analysis --category pill
"""

import argparse
import json
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


def overlay_heatmap(image: np.ndarray, anomaly_map: np.ndarray, vmax: float, alpha: float = 0.5) -> np.ndarray:
    norm = np.clip(anomaly_map / (vmax + 1e-8), 0, 1)
    heatmap = plt.cm.jet(norm)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heatmap, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-examples", type=int, default=3, help="failure panels to save per bucket")
    args = parser.parse_args()

    checkpoint = args.checkpoint or (ARTIFACT_DIR / f"{args.category}_patchcore.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found. Run `python -m src.train --category {args.category}` first."
        )

    dataset = MVTecTestDataset(args.category, image_size=args.image_size)
    model = PatchCore()
    model.load(checkpoint)
    threshold = model.threshold
    if threshold is None:
        raise RuntimeError("Checkpoint has no threshold -- retrain with the current src/train.py.")

    records = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        out = model.predict(sample["image"].unsqueeze(0))
        score = out["scores"][0].item()
        records.append(
            {
                "idx": idx,
                "path": sample["path"],
                "defect_type": sample["defect_type"],
                "label": sample["label"],
                "score": score,
                "predicted": int(score > threshold),
                "anomaly_map": out["anomaly_maps"][0].numpy(),
            }
        )

    tp = sum(1 for r in records if r["label"] == 1 and r["predicted"] == 1)
    fn = sum(1 for r in records if r["label"] == 1 and r["predicted"] == 0)
    tn = sum(1 for r in records if r["label"] == 0 and r["predicted"] == 0)
    fp = sum(1 for r in records if r["label"] == 0 and r["predicted"] == 1)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    print(f"Category: {args.category}  (threshold={threshold:.2f})")
    print(f"TP={tp} FN={fn} TN={tn} FP={fp}  Precision={precision:.3f}  Recall={recall:.3f}")

    by_type = {}
    for r in records:
        if r["label"] == 1:
            by_type.setdefault(r["defect_type"], []).append(r["predicted"])
    per_type_recall = {}
    print("\nRecall per defect type:")
    for defect_type, preds in sorted(by_type.items()):
        rec = sum(preds) / len(preds)
        per_type_recall[defect_type] = rec
        print(f"  {defect_type:20s} {rec:.3f}  (n={len(preds)})")

    out_dir = ARTIFACT_DIR / "error_analysis" / args.category
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "category": args.category,
        "threshold": threshold,
        "confusion_matrix": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "precision": precision,
        "recall": recall,
        "recall_per_defect_type": per_type_recall,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_dir / 'summary.json'}")

    false_negatives = sorted(
        (r for r in records if r["label"] == 1 and r["predicted"] == 0), key=lambda r: r["score"]
    )
    false_positives = sorted(
        (r for r in records if r["label"] == 0 and r["predicted"] == 1), key=lambda r: -r["score"]
    )

    vmax = threshold * 1.5

    def save_panel(record: dict, tag: str) -> None:
        display_img = load_display_image(record["path"], args.image_size)
        overlay = overlay_heatmap(display_img, record["anomaly_map"], vmax)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(display_img)
        axes[0].set_title("Original")
        axes[1].imshow(overlay)
        axes[1].set_title(f"score={record['score']:.2f} (thr={threshold:.2f})")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{args.category} / {record['defect_type']} / {tag}")
        fig.tight_layout()
        out_path = out_dir / f"{tag}_{record['defect_type']}_{record['idx']}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved {out_path}")

    for r in false_negatives[: args.num_examples]:
        save_panel(r, "missed_defect")
    for r in false_positives[: args.num_examples]:
        save_panel(r, "false_alarm")

    if not false_negatives and not false_positives:
        print("\nNo misclassifications at this threshold on this test split.")


if __name__ == "__main__":
    main()
