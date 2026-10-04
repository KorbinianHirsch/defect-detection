"""Simulates a camera on a production line.

Periodically copies a real MVTec AD test image (mixed good/defect, the way
a real line would produce a mix of good and bad parts) into
data/production/incoming/, where the worker picks it up and scores it.

The filename encodes the true label (`<ts>__<true_label>__<original>.png`)
purely so the dashboard can show a live accuracy panel -- a real camera
obviously wouldn't provide that.

Usage:
    python -m src.production.simulator --category bottle --interval 2 --count 100
"""

import argparse
import random
import shutil
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INCOMING_DIR = DATA_DIR / "production" / "incoming"


def collect_source_images(category: str) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Returns (good_images, [(path, defect_type), ...]) separately, since a
    real line sees mostly good parts -- MVTec's test split is roughly
    balanced good/defect, which would give an unrealistic reject rate if
    sampled uniformly."""
    test_dir = DATA_DIR / category / "test"
    good, defects = [], []
    for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for img_path in sorted(defect_dir.glob("*.png")):
            if defect_dir.name == "good":
                good.append(img_path)
            else:
                defects.append((img_path, defect_dir.name))
    return good, defects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between parts")
    parser.add_argument("--count", type=int, default=0, help="0 = run forever")
    parser.add_argument(
        "--defect-rate",
        type=float,
        default=0.1,
        help="fraction of parts that are defective (default 0.1, closer to a real line "
        "than MVTec's ~roughly-balanced test split)",
    )
    args = parser.parse_args()

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    good, defects = collect_source_images(args.category)
    if not good or not defects:
        raise RuntimeError(f"No good/defect test images found for category '{args.category}'")

    print(
        f"Simulating camera feed for '{args.category}': {len(good)} good / {len(defects)} "
        f"defect source images, 1 part every {args.interval}s, defect rate {args.defect_rate:.0%}"
    )

    i = 0
    while args.count == 0 or i < args.count:
        if random.random() < args.defect_rate:
            src_path, true_label = random.choice(defects)
        else:
            src_path, true_label = random.choice(good), "good"
        dest_name = f"{int(time.time() * 1000)}__{true_label}__{src_path.name}"
        dest_path = INCOMING_DIR / dest_name
        shutil.copy2(src_path, dest_path)
        print(f"[{i + 1}] captured {dest_name}")
        i += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
