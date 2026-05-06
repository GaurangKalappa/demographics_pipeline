# ============================================================
#  models/mtl_model.py  –  MobileNetV3-Small Face MTL model
#
#  Multi-task heads:
#      • age_head    → continuous regression  (float, clamped 0–100)
#      • gender_head → binary classification  (P Female via sigmoid)
#
#  Plug in your weights:
#      model = MTLModel("weights/mtl_model.pt")
# ============================================================
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models


class MTLModel(nn.Module):
    """
    MobileNetV3-Small backbone with two task heads.

    Backbone : MobileNetV3-Small feature extractor
               (features + adaptive_avg_pool2d only; classifier discarded)
    Heads    :
        • age_head    → regression  → scalar float (continuous age 0–100)
        • gender_head → binary      → P(Female) via sigmoid
    """

    # MobileNetV3-Small pooled feature dimension (fixed by architecture)
    _FEATURE_DIM = 576

    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        # ── Backbone: features + pool only ───────────────────────────────
        # Identical pattern to PARModel: extract features and avgpool
        # sub-modules directly, discarding the classifier block.
        _full = tv_models.mobilenet_v3_small(weights=None)
        self.features = _full.features            # Conv feature extractor
        self.pool     = _full.avgpool             # AdaptiveAvgPool2d(1,1)
        del _full

        feature_dim = self._FEATURE_DIM

        # ── Task heads ───────────────────────────────────────────────────
        self.age_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),                    # raw age; clamped in predict()
        )
        self.gender_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),                    # raw logit; sigmoid in predict()
        )

        self.to(self.device)

        # ── Load weights ─────────────────────────────────────────────────
        self._weights_loaded = False
        if weights_path and os.path.exists(weights_path):
            self._load(weights_path)
        elif weights_path:
            print(f"[MTL] ⚠  Weights not found at '{weights_path}'. "
                  "Running in scaffold mode (random weights).")
        else:
            print("[MTL] ℹ  No weights path provided – scaffold mode.")

    # ── Weight loading ───────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        """Load a state-dict checkpoint. Supports both raw state_dict
        and {'state_dict': ...} wrapper formats."""
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state, strict=False)
        self._weights_loaded = True
        print(f"[MTL] ✔  Weights loaded from '{path}'")

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.features(x)                  # (B, 576, H', W')
        feats = self.pool(feats)                  # (B, 576, 1, 1)
        feats = feats.flatten(1)                  # (B, 576)
        return {
            "age":    self.age_head(feats),        # (B, 1)
            "gender": self.gender_head(feats),     # (B, 1)
        }

    # ── Inference helper ─────────────────────────────────────────────────────

    def predict(self, chw_array: np.ndarray) -> dict:
        """
        Run inference on a single pre-processed face crop.

        Parameters
        ----------
        chw_array : np.ndarray
            Shape (3, H, W), dtype float32, ImageNet-normalised.
            Must be C-contiguous (guaranteed by prepare_face_crop()).

        Returns
        -------
        dict with keys:
            gender_face_score : float   P(Female) in [0, 1]
            age_raw           : float   continuous age estimate [0, 100]
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(chw_array).unsqueeze(0).to(self.device)
            out    = self.forward(tensor)

            # age: index [0,0] to get scalar, then clamp to human range
            age_raw      = float(torch.clamp(out["age"][0, 0], 0.0, 100.0).item())
            # gender: index [0,0] to get scalar logit, then sigmoid
            gender_score = float(torch.sigmoid(out["gender"][0, 0]).item())

        return {
            "gender_face_score": gender_score,
            "age_raw":           age_raw,
        }