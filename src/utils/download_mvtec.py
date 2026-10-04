"""
Download and extract the MVTec AD dataset.

MVTec AD is distributed by MVTec Software GmbH for academic/research use
under a CC BY-NC-SA 4.0 license (non-commercial). There is no anonymous
direct-download link: the official page requires filling in a short form
(name/email/institution) before it reveals the actual download URL, and that
URL is per-visitor / expires, so it cannot be hardcoded into this script.
Official page: https://www.mvtec.com/company/research/datasets/mvtec-ad

This script does NOT attempt an automatic download. It only extracts an
archive you already downloaded manually (see MANUAL_INSTRUCTIONS below).

Usage:
    python -m src.utils.download_mvtec --category bottle --archive C:\\path\\to\\mvtec_anomaly_detection.tar.xz
"""

import argparse
import sys
import tarfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
ARCHIVE_NAME = "mvtec_anomaly_detection.tar.xz"

MANUAL_INSTRUCTIONS = f"""
No local archive found. The MVTec AD dataset has no anonymous direct-download
link -- the official page hands out a (short-lived, per-visitor) URL only
after you fill in a short form.

Manual steps to get the dataset:
  1. Open https://www.mvtec.com/company/research/datasets/mvtec-ad in a browser.
  2. Click "Download dataset" and fill in the form (name/email/institution).
  3. Download the full archive ("mvtec_anomaly_detection.tar.xz", ~4.9 GB) --
     or, if the page offers a single category archive (e.g. "bottle.tar.xz"),
     that works too.
  4. Save the downloaded .tar.xz file into:
       {RAW_DIR}
  5. Re-run this script with --archive pointing at the file, e.g.:
       python -m src.utils.download_mvtec --category bottle --archive {RAW_DIR / ARCHIVE_NAME}
"""


def extract(archive_path: Path, category: str) -> bool:
    print(f"Extracting {archive_path.name} ...")
    with tarfile.open(archive_path) as tar:
        members = tar.getmembers()
        category_members = [m for m in members if m.name.split("/")[0] == category]
        if not category_members:
            print(
                f"Category '{category}' not found in archive. "
                f"Available top-level entries: "
                f"{sorted({m.name.split('/')[0] for m in members})[:20]}"
            )
            return False
        tar.extractall(path=DATA_DIR, members=category_members, filter="data")
    print(f"Extracted '{category}' to {DATA_DIR / category}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        default="bottle",
        help="MVTec AD category to extract (default: bottle)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to a manually-downloaded mvtec_anomaly_detection.tar.xz "
        f"(default: {RAW_DIR / ARCHIVE_NAME})",
    )
    args = parser.parse_args()

    target_dir = DATA_DIR / args.category / "train" / "good"
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"'{args.category}' already present at {DATA_DIR / args.category}, nothing to do.")
        return

    archive_path = args.archive or (RAW_DIR / ARCHIVE_NAME)

    if not archive_path.exists():
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    ok = extract(archive_path, args.category)
    if not ok:
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)


if __name__ == "__main__":
    main()
