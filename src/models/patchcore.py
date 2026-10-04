"""PatchCore anomaly detection.

Pretrained-backbone mid-level patch features + a coreset-subsampled memory
bank of "normal" patches. Anomaly score for a test patch = distance to its
nearest neighbor in the memory bank; image score = max over patches.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly
Detection" (PatchCore), CVPR 2022. This is a simplified implementation
(no image-score reweighting refinement) intended as a strong baseline.
"""

from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchFeatureExtractor(nn.Module):
    """Locally-aggregated mid-level patch features from a frozen pretrained backbone."""

    def __init__(self, backbone: str = "wide_resnet50_2", layers=("layer2", "layer3")):
        super().__init__()
        indices = self._resolve_indices(backbone, layers)
        self.backbone = timm.create_model(
            backbone, pretrained=True, features_only=True, out_indices=indices
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _resolve_indices(backbone: str, layers) -> tuple:
        probe = timm.create_model(backbone, features_only=True, pretrained=False)
        names = [d["module"] for d in probe.feature_info.get_dicts()]
        indices = tuple(i for i, name in enumerate(names) if any(l in name for l in layers))
        if len(indices) != len(layers):
            raise ValueError(f"Could not resolve indices for layers {layers} among {names}")
        return indices

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> patch feature grid [B,C,Hf,Wf] at the finest layer's resolution."""
        feature_maps = self.backbone(x)
        pooled = [F.avg_pool2d(fm, kernel_size=3, stride=1, padding=1) for fm in feature_maps]
        target_h, target_w = pooled[0].shape[-2:]
        aligned = [pooled[0]] + [
            F.interpolate(fm, size=(target_h, target_w), mode="bilinear", align_corners=False)
            for fm in pooled[1:]
        ]
        return torch.cat(aligned, dim=1)


def greedy_coreset_indices(
    features: torch.Tensor, n_samples: int, projection_dim: int = 128, seed: int = 0
) -> torch.Tensor:
    """Greedy min-max (farthest-point) coreset subsampling, PatchCore-style.

    A random projection to `projection_dim` keeps the O(n_samples * N)
    distance computations cheap on CPU.
    """
    n = features.shape[0]
    if n_samples >= n:
        return torch.arange(n)

    generator = torch.Generator().manual_seed(seed)
    if projection_dim < features.shape[1]:
        projection = torch.randn(features.shape[1], projection_dim, generator=generator)
        projection /= projection_dim**0.5
        proj = features @ projection
    else:
        proj = features

    selected = torch.empty(n_samples, dtype=torch.long)
    selected[0] = torch.randint(0, n, (1,), generator=generator)
    min_dist = torch.cdist(proj, proj[selected[0:1]]).squeeze(1)

    for i in range(1, n_samples):
        next_idx = torch.argmax(min_dist)
        selected[i] = next_idx
        new_dist = torch.cdist(proj, proj[next_idx : next_idx + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, new_dist)

    return selected


class PatchCore:
    """Fit a coreset memory bank on normal patches; score new images by NN-distance."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_ratio: float = 0.02,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = PatchFeatureExtractor(backbone, layers).to(self.device)
        self.coreset_ratio = coreset_ratio
        self.memory_bank: Optional[torch.Tensor] = None
        self.threshold: Optional[float] = None

    @torch.no_grad()
    def _extract_patches(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(images.to(self.device))
        b, c, h, w = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(-1, c)

    @torch.no_grad()
    def fit(self, dataloader) -> None:
        all_patches = []
        for batch in dataloader:
            images = batch if torch.is_tensor(batch) else batch["image"]
            all_patches.append(self._extract_patches(images).cpu())
        patches = torch.cat(all_patches, dim=0)

        n_samples = max(1, int(len(patches) * self.coreset_ratio))
        idx = greedy_coreset_indices(patches, n_samples)
        self.memory_bank = patches[idx].to(self.device)
        print(
            f"Memory bank: {len(patches)} patches -> {len(idx)} coreset samples "
            f"({self.coreset_ratio:.0%})"
        )

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> dict:
        """Returns {"scores": [B], "anomaly_maps": [B, image_h, image_w]}."""
        if self.memory_bank is None:
            raise RuntimeError("Call fit() (or load()) before predict().")

        image_h, image_w = images.shape[-2:]
        feats = self.extractor(images.to(self.device))
        b, c, h, w = feats.shape
        patches = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)

        scores = torch.empty(b)
        maps = torch.empty(b, h, w)
        for i in range(b):
            dist = torch.cdist(patches[i], self.memory_bank)  # [h*w, K]
            nn_dist, _ = dist.min(dim=1)
            maps[i] = nn_dist.reshape(h, w).cpu()
            scores[i] = nn_dist.max().cpu()

        maps = F.interpolate(
            maps.unsqueeze(1), size=(image_h, image_w), mode="bilinear", align_corners=False
        ).squeeze(1)
        return {"scores": scores, "anomaly_maps": maps}

    @torch.no_grad()
    def compute_threshold(self, dataloader, margin: float = 1.1) -> float:
        """Sets self.threshold from the max image score seen on `dataloader`
        (expected to be train/good), scaled by `margin`. Heuristic: nothing in
        that set should be anomalous, so its highest score is a reasonable
        boundary between "normal" and "anomalous" for the demo/API."""
        max_score = 0.0
        for batch in dataloader:
            images = batch if torch.is_tensor(batch) else batch["image"]
            scores = self.predict(images)["scores"]
            max_score = max(max_score, scores.max().item())
        self.threshold = max_score * margin
        return self.threshold

    def save(self, path: Path) -> None:
        torch.save(
            {
                "memory_bank": self.memory_bank.cpu(),
                "coreset_ratio": self.coreset_ratio,
                "threshold": self.threshold,
            },
            path,
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.memory_bank = ckpt["memory_bank"].to(self.device)
        self.coreset_ratio = ckpt["coreset_ratio"]
        self.threshold = ckpt.get("threshold")
