from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHECKPOINT = DATA_DIR / "processed" / "bottle_patchcore.pt"
SAMPLE_GOOD_IMAGE = DATA_DIR / "bottle" / "test" / "good" / "000.png"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="No trained PatchCore checkpoint for 'bottle' yet",
)


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_score_and_heatmap(client):
    with open(SAMPLE_GOOD_IMAGE, "rb") as f:
        response = client.post("/predict", files={"file": ("bottle.png", f, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["score"], float)
    assert body["category"] == "bottle"
    assert len(body["heatmap_png_base64"]) > 0


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("note.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
