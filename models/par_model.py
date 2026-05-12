# ============================================================
#  models/par_model.py  –  PP-Attribute / MobileNetV3 Body PAR model
#
#  Architecture scaffold – plug in your trained weights via:
#      model = PARModel("weights/par_model.pt")
# ============================================================
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models


# ── Model definition ─────────────────────────────────────────────────────────

class PARModel(nn.Module):
    """
    Pedestrian Attribute Recognition model.

    Backbone : MobileNetV3-Large feature extractor
               (features + adaptive_avg_pool2d only; classifier discarded)
    Heads    :
        • gender_head  → sigmoid  → P(Female) in [0, 1]
        • age_head     → softmax  → [Child, Young Adult, Adult, Senior]
        • orient_head  → softmax  → [Front, Side, Back]

    Use predict() for processed inference outputs.
    """

    AGE_LABELS    = ["Child", "Adult", "Senior"]
    ORIENT_LABELS = ["Front", "Side", "Back"]

    # MobileNetV3-Large pooled feature dimension (fixed by architecture)
    #_FEATURE_DIM=960
    _FEATURE_DIM = 1280

    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        # ── Backbone: features + pool only ───────────────────────────────
        # We extract only the convolutional feature block and the adaptive
        # average pool.  The original classifier is NOT used – this avoids
        # relying on classifier[0].in_features which can break if torchvision
        # changes the head layout in a future release.
        #_full = tv_models.mobilenet_v3_large(weights=None)
        _full = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        #self.features = _full.features            # Conv feature extractor
        #self.pool     = _full.avgpool             # AdaptiveAvgPool2d(1,1)
        self.features = _full.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        del _full                                 # free the unused classifier

        feature_dim = self._FEATURE_DIM

        # ── Task heads ───────────────────────────────────────────────────
        self.gender_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),                    # raw logit; sigmoid in predict()
        )
        self.age_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, len(self.AGE_LABELS)),
        )
        self.orient_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(self.ORIENT_LABELS)),
        )

        self.to(self.device)

        # ── Load weights ─────────────────────────────────────────────────
        self._weights_loaded = False
        if weights_path and os.path.exists(weights_path):
            self._load(weights_path)
        elif weights_path:
            print(f"[PAR] ⚠  Weights not found at '{weights_path}'. "
                  "Running in scaffold mode (random weights).")
        else:
            print("[PAR] ℹ  No weights path provided – scaffold mode.")

    # ── Weight loading ───────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        """Load a state-dict checkpoint. Supports both raw state_dict
        and {'state_dict': ...} wrapper formats."""
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state, strict=False)
        self._weights_loaded = True
        print(f"[PAR] ✔  Weights loaded from '{path}'")

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.features(x)                  # (B, 960, H', W')
        feats = self.pool(feats)                  # (B, 960, 1, 1)
        feats = feats.flatten(1)                  # (B, 960)
        return {
            "gender": self.gender_head(feats),    # (B, 1)  raw logit
            "age":    self.age_head(feats),        # (B, 3)  raw logits
            "orient": self.orient_head(feats),     # (B, 3)  raw logits
        }

    # ── Inference helper ─────────────────────────────────────────────────────

    def predict(self, chw_array: np.ndarray) -> dict:
        """
        Run inference on a single pre-processed body crop.

        Parameters
        ----------
        chw_array : np.ndarray
            Shape (3, H, W), dtype float32, pixel values normalised to [0, 1].
            Must be C-contiguous (guaranteed by prepare_body_crop()).

        Returns
        -------
        dict with keys:
            gender_body_score : float   P(Female) in [0, 1]
            age_coarse        : str     one of AGE_LABELS
            orientation       : str     one of ORIENT_LABELS
            confidence        : float   max softmax prob of age head (= C_body)
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(chw_array).unsqueeze(0).to(self.device)
            out    = self.forward(tensor)

            # gender: index [0,0] to get scalar logit, then sigmoid
            gender_prob  = torch.sigmoid(out["gender"][0, 0]).item()

            age_probs    = torch.softmax(out["age"],    dim=1)[0]   # (4,)
            orient_probs = torch.softmax(out["orient"], dim=1)[0]   # (3,)

            age_idx    = int(age_probs.argmax().item())
            orient_idx = int(orient_probs.argmax().item())
            confidence = float(age_probs[age_idx].item())

        return {
            "gender_body_score": float(gender_prob),
            "age_coarse":        self.AGE_LABELS[age_idx],
            "orientation":       self.ORIENT_LABELS[orient_idx],
            "confidence":        confidence,
        }