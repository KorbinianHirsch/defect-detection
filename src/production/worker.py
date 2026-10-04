"""Production inference worker.

Watches data/production/incoming/ for new captures (as if a camera just
wrote a frame), scores each with PatchCore, sorts the file into accepted/
or rejected/ (simulating a physical reject mechanism on the line), saves a
heatmap for rejects, and logs the decision + latency to the SQLite event
log the dashboard reads from.

Usage:
    python -m src.production.worker --category bottle
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.patchcore import PatchCore
from src.production.db import connect, insert_event
from src.utils.dataset import default_transform
from src.utils.viz import overlay_heatmap, save_png

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROD_DIR = DATA_DIR / "production"
INCOMING_DIR = PROD_DIR / "incoming"
ACCEPTED_DIR = PROD_DIR / "accepted"
REJECTED_DIR = PROD_DIR / "rejected"
ARTIFACT_DIR = DATA_DIR / "processed"
IMAGE_SIZE = 224


def parse_true_label(filename: str) -> str | None:
    parts = filename.split("__")
    return parts[1] if len(parts) >= 3 else None


def process_file(model: PatchCore, transform, path: Path, category: str) -> None:
    start = time.perf_counter()
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    out = model.predict(tensor)
    score = out["scores"][0].item()
    latency_ms = (time.perf_counter() - start) * 1000

    is_anomaly = model.threshold is not None and score > model.threshold
    dest_dir = REJECTED_DIR if is_anomaly else ACCEPTED_DIR
    dest_path = dest_dir / path.name
    path.replace(dest_path)

    if is_anomaly:
        display = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
        anomaly_map = out["anomaly_maps"][0].numpy()
        vmax = model.threshold * 1.5
        overlay = overlay_heatmap(display, anomaly_map, vmax)
        save_png(overlay, dest_dir / f"{path.stem}_heatmap.png")

    with connect() as conn:
        insert_event(
            conn,
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
            filename=path.name,
            true_label=parse_true_label(path.name),
            score=score,
            threshold=model.threshold,
            is_anomaly=is_anomaly,
            latency_ms=latency_ms,
            sorted_path=str(dest_path),
        )

    verdict = "REJECT" if is_anomaly else "accept"
    print(
        f"{verdict:7s} score={score:6.2f} thr={model.threshold:6.2f} "
        f"latency={latency_ms:5.1f}ms  {path.name}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    args = parser.parse_args()

    for d in (INCOMING_DIR, ACCEPTED_DIR, REJECTED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    checkpoint = ARTIFACT_DIR / f"{args.category}_patchcore.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found. Run `python -m src.train --category {args.category}` first."
        )

    model = PatchCore()
    model.load(checkpoint)
    transform = default_transform(IMAGE_SIZE)

    print(
        f"Worker started for '{args.category}' (threshold={model.threshold:.2f}). "
        f"Watching {INCOMING_DIR} ..."
    )

    while True:
        pending = sorted(INCOMING_DIR.glob("*.png"))
        for path in pending:
            try:
                process_file(model, transform, path, args.category)
            except Exception as e:
                print(f"ERROR processing {path.name}: {e}")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
