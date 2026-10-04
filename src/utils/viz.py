"""Shared heatmap-overlay helpers used by the API, worker, and offline scripts."""

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def overlay_heatmap(image: np.ndarray, anomaly_map: np.ndarray, vmax: float, alpha: float = 0.5) -> np.ndarray:
    """Blend a jet colormap over `image`, scaled to [0, vmax] (not per-image
    min-max -- see src/api/main.py for why that matters for single-image scoring)."""
    norm = np.clip(anomaly_map / (vmax + 1e-8), 0, 1)
    heatmap = plt.cm.jet(norm)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heatmap, 0, 1)


def save_png(image_float01: np.ndarray, path) -> None:
    Image.fromarray((image_float01 * 255).astype(np.uint8)).save(path)


def encode_png_base64(image_float01: np.ndarray) -> str:
    img = Image.fromarray((image_float01 * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
