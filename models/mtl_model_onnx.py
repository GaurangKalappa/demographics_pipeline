# ============================================================
#  models/mtl_model_onnx.py  –  ONNX Runtime wrapper for the MTL model
#
#  Identical predict() interface to MTLModel (PyTorch version).
#  Pipeline code selects this class automatically when USE_ONNX = True.
#
#  Generate the .onnx file first:
#      python models/export_onnx.py
# ============================================================
from __future__ import annotations

import numpy as np
from scipy.special import expit   # sigmoid, CPU-only

from models.onnx_runner import ONNXRunner


class MTLModelONNX(ONNXRunner):
    """
    ONNX Runtime inference wrapper for the MTL face model.

    ONNX output order (set during export):
        index 0 → "age"    shape (1, 1)   raw regression output
        index 1 → "gender" shape (1, 1)   raw logit
    """

    def __init__(self, onnx_path: str) -> None:
        super().__init__(onnx_path)

    def predict(self, chw_array: np.ndarray) -> dict:
        """
        Identical interface to MTLModel.predict().

        Parameters
        ----------
        chw_array : np.ndarray  shape (3, H, W), float32, ImageNet-normalised

        Returns
        -------
        dict with keys:
            gender_face_score : float
            age_raw           : float
        """
        outputs = self._run(self._batch(chw_array))

        age_raw      = float(np.clip(outputs[0][0, 0], 0.0, 100.0))
        gender_score = float(expit(outputs[1][0, 0]))

        return {
            "gender_face_score": gender_score,
            "age_raw":           age_raw,
        }