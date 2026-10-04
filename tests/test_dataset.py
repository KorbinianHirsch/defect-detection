from pathlib import Path

import pytest

from src.utils.dataset import MVTecTestDataset, MVTecTrainDataset

CATEGORY = "bottle"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CATEGORY_PRESENT = (DATA_DIR / CATEGORY / "train" / "good").exists()

pytestmark = pytest.mark.skipif(
    not CATEGORY_PRESENT,
    reason=f"MVTec AD category '{CATEGORY}' not downloaded yet",
)


def test_train_dataset_loads_good_images():
    ds = MVTecTrainDataset(CATEGORY)
    assert len(ds) > 0
    image = ds[0]
    assert image.shape == (3, 256, 256)


def test_test_dataset_has_good_and_anomalous_samples():
    ds = MVTecTestDataset(CATEGORY)
    labels = {ds[i]["label"] for i in range(len(ds))}
    assert 0 in labels
    assert 1 in labels


def test_anomalous_sample_has_nonempty_mask():
    ds = MVTecTestDataset(CATEGORY)
    anomalous = next(s for s in ds if s["label"] == 1)
    assert anomalous["mask"].sum() > 0


def test_good_sample_has_empty_mask():
    ds = MVTecTestDataset(CATEGORY)
    good = next(s for s in ds if s["label"] == 0)
    assert good["mask"].sum() == 0
