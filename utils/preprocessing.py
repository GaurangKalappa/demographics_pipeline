# ============================================================
#  utils/preprocessing.py  –  CLAHE, crop helpers, tensor prep
#
#  All array outputs are C-contiguous (np.ascontiguousarray) so
#  torch.from_numpy() works without copying and no read-only
#  view issues arise from numpy 2.0 stride semantics.
# ============================================================
from __future__ import annotations

import cv2
import numpy as np

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, PAR_INPUT_SIZE, MTL_INPUT_SIZE


# ── CLAHE instance (reused across calls) ──────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)


def apply_clahe(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to a BGR image crop.
    Converts to LAB colour space, equalises the L channel, converts back.
    Returns a new C-contiguous BGR array.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop
    lab  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)   # cv2 always returns contiguous


def safe_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              pad: int = 0) -> np.ndarray | None:
    """
    Safely crop a region from a frame with optional padding.
    Returns None if the resulting crop is empty.
    The returned array is a contiguous copy (not a view).
    """
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.ascontiguousarray(frame[y1:y2, x1:x2])


def prepare_body_crop(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Pre-process a full-body crop for the PAR model.

    Steps:
      1. CLAHE normalisation
      2. Resize to PAR_INPUT_SIZE  (W × H)
      3. BGR → RGB
      4. Normalise pixels to [0, 1]  (float32)
      5. HWC → CHW layout
      6. Ensure C-contiguous memory for torch.from_numpy()

    Returns
    -------
    np.ndarray  shape (3, H, W), dtype float32, C-contiguous
    """
    #crop = apply_clahe(bgr_crop)
    crop = bgr_crop.copy()
    crop = cv2.resize(crop, PAR_INPUT_SIZE)                      # (H, W, 3) BGR uint8
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)                 # (H, W, 3) RGB uint8
    #crop = crop.astype(np.float32) / 255.0 
    #crop = crop.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    crop = crop.astype(np.float32) / 255.0
    crop = (crop - mean) / std
    crop = crop.transpose(2, 0, 1)
    
    return np.ascontiguousarray(crop)                            # guarantee contiguous


def prepare_face_crop(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Pre-process a face crop for the MTL model.

    Steps:
      1. CLAHE normalisation
      2. Resize to MTL_INPUT_SIZE  (W × H)
      3. BGR → RGB
      4. Normalise to [0, 1]  (float32)
      5. ImageNet mean/std normalisation
      6. HWC → CHW layout
      7. Ensure C-contiguous memory for torch.from_numpy()

    Returns
    -------
    np.ndarray  shape (3, H, W), dtype float32, C-contiguous, ImageNet-normalised
    """
    # ImageNet statistics (RGB order)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    #crop = apply_clahe(bgr_crop)
    crop = bgr_crop.copy()
    crop = cv2.resize(crop, MTL_INPUT_SIZE)                      # (H, W, 3) BGR uint8
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)                 # (H, W, 3) RGB uint8
    crop = crop.astype(np.float32) / 255.0                       # (H, W, 3) float32
    crop = (crop - mean) / std                                   # (H, W, 3) normalised
    crop = crop.transpose(2, 0, 1)                               # (3, H, W)
    return np.ascontiguousarray(crop)