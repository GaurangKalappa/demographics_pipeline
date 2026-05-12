# ============================================================
#  models/onnx_runner.py  –  Shared ONNX Runtime base class
#
#  Both PARModelONNX and MTLModelONNX inherit from ONNXRunner.
#  All session management, provider selection, and I/O discovery
#  lives here — subclasses only implement predict().
#
#  Requires:
#      pip install onnxruntime       # CPU-only
#      pip install onnxruntime-gpu   # CUDA (use instead of onnxruntime)
# ============================================================
from __future__ import annotations

import os
import numpy as np


class ONNXRunner:
    """
    Generic ONNX Runtime inference session wrapper.

    Automatically selects CUDAExecutionProvider if available,
    falls back to CPUExecutionProvider otherwise.

    Subclasses call self._run(input_array) and parse the returned
    list of numpy arrays.
    """

    def __init__(self, onnx_path: str) -> None:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Run  python models/export_onnx.py  first to generate it.")

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is not installed.\n"
                "Install it with:  pip install onnxruntime\n"
                "  or for GPU:     pip install onnxruntime-gpu")

        # Provider selection: CUDA if available, CPU otherwise
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"[ONNX] {os.path.basename(onnx_path)} → CUDA provider")
        else:
            providers = ["CPUExecutionProvider"]
            print(f"[ONNX] {os.path.basename(onnx_path)} → CPU provider")

        self._session      = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name   = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    # ── Shared inference call ─────────────────────────────────────────────────

    def _run(self, input_array: np.ndarray) -> list[np.ndarray]:
        """
        Run one forward pass.

        Parameters
        ----------
        input_array : np.ndarray  shape (1, C, H, W), dtype float32, C-contiguous

        Returns
        -------
        List of numpy arrays, one per model output, in the order
        they were named during export (see export_onnx.py).
        """
        feeds = {self._input_name: input_array}
        return self._session.run(self._output_names, feeds)

    def _batch(self, chw_array: np.ndarray) -> np.ndarray:
        """Add batch dimension and ensure C-contiguous float32."""
        arr = np.ascontiguousarray(chw_array[np.newaxis], dtype=np.float32)
        return arr