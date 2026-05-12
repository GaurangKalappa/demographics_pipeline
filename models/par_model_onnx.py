# ============================================================
#  models/par_model_onnx.py  –  ONNX Runtime wrapper for the PAR model
#
#  Identical predict() interface to PARModel (PyTorch version).
#  Pipeline code selects this class automatically when USE_ONNX = True.
#
#  Generate the .onnx file first:
#      python models/export_onnx.py
# ============================================================
from __future__ import annotations

import numpy as np
from scipy.special import softmax, expit   # CPU-only, no torch needed

from models.onnx_runner import ONNXRunner


class PARModelONNX(ONNXRunner):
    """
    ONNX Runtime inference wrapper for the PAR body model.

    ONNX output order (set during export):
        index 0 → "gender"  shape (1, 1)   raw logit
        index 1 → "age"     shape (1, 4)   raw logits
        index 2 → "orient"  shape (1, 3)   raw logits
    """

    AGE_LABELS    = ["Child", "Young Adult", "Adult", "Senior"]
    ORIENT_LABELS = ["Front", "Side", "Back"]

    def __init__(self, onnx_path: str) -> None:
        super().__init__(onnx_path)

    def predict(self, chw_array: np.ndarray) -> dict:
        """
        Identical interface to PARModel.predict().

        Parameters
        ----------
        chw_array : np.ndarray  shape (3, H, W), float32, values in [0, 1]

        Returns
        -------
        dict with keys:
            gender_body_score : float
            age_coarse        : str
            orientation       : str
            confidence        : float
        """
        outputs = self._run(self._batch(chw_array))

        gender_logit = outputs[0][0, 0]            # scalar
        age_logits   = outputs[1][0]               # (4,)
        orient_logits= outputs[2][0]               # (3,)

        gender_prob  = float(expit(gender_logit))  # sigmoid
        age_probs    = softmax(age_logits)
        orient_probs = softmax(orient_logits)

        age_idx    = int(np.argmax(age_probs))
        orient_idx = int(np.argmax(orient_probs))
        confidence = float(age_probs[age_idx])

        return {
            "gender_body_score": gender_prob,
            "age_coarse":        self.AGE_LABELS[age_idx],
            "orientation":       self.ORIENT_LABELS[orient_idx],
            "confidence":        confidence,
        }