"""FastAPI backend for defect detection.

Loads a fitted PatchCore model at startup and exposes an endpoint that
accepts an image upload and returns an anomaly score, an is_anomaly
decision (score vs. the threshold computed during training), and a
heatmap overlay as a base64-encoded PNG.

Run:
    uvicorn src.api.main:app --reload
"""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.models.patchcore import PatchCore
from src.utils.dataset import default_transform
from src.utils.viz import encode_png_base64, overlay_heatmap

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CATEGORY = os.environ.get("MVTEC_CATEGORY", "bottle")
IMAGE_SIZE = 224

_transform = default_transform(IMAGE_SIZE)
_state: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = ARTIFACT_DIR / f"{CATEGORY}_patchcore.pt"
    if not checkpoint.exists():
        raise RuntimeError(
            f"No checkpoint at {checkpoint}. "
            f"Run `python -m src.train --category {CATEGORY}` first."
        )
    model = PatchCore()
    model.load(checkpoint)
    _state["model"] = model
    yield
    _state["model"] = None


app = FastAPI(title="Defect Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    category: str
    score: float
    threshold: Optional[float]
    is_anomaly: Optional[bool]
    heatmap_png_base64: str


@app.get("/health")
def health():
    return {"status": "ok", "category": CATEGORY, "model_loaded": _state["model"] is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    model: Optional[PatchCore] = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    raw = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    display_image = np.asarray(pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
    tensor = _transform(pil_image).unsqueeze(0)

    out = model.predict(tensor)
    score = out["scores"][0].item()
    anomaly_map = out["anomaly_maps"][0].numpy()

    vmax = model.threshold if model.threshold is not None else float(anomaly_map.max())
    overlay = overlay_heatmap(display_image, anomaly_map, vmax)
    heatmap_b64 = encode_png_base64(overlay)

    is_anomaly = score > model.threshold if model.threshold is not None else None

    return PredictResponse(
        category=CATEGORY,
        score=score,
        threshold=model.threshold,
        is_anomaly=is_anomaly,
        heatmap_png_base64=heatmap_b64,
    )
