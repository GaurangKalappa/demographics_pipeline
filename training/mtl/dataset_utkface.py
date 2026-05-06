# ============================================================
#  training/mtl/dataset_utkface.py
#
#  PyTorch Dataset for UTKFace → trains the MTL (face age+gender) model.
#
#  Dataset source:
#    Official : https://susanqq.github.io/UTKFace/
#    Kaggle   : https://www.kaggle.com/datasets/jangedoo/utkface-new
#
#  Filename convention (already embedded in every image name):
#    [age]_[gender]_[race]_[date&time].jpg
#    gender: 0 = Male, 1 = Female
#
#  Download → unzip → point UTK_ROOT at the folder containing the .jpg files.
# ============================================================
from __future__ import annotations

import os
import glob
import random
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UTKFaceDataset(Dataset):
    """
    Loads UTKFace images and returns:
        image  : torch.Tensor  (3, H, W)  float32, ImageNet-normalised
        age    : torch.Tensor  scalar      float32  [0, 100]
        gender : torch.Tensor  scalar      float32  0.0=Male / 1.0=Female
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        root: str,
        img_size: tuple[int, int] = (224, 224),
        split: str = "train",          # "train" | "val"
        val_fraction: float = 0.10,
        seed: int = 42,
        augment: bool = True,
    ):
        """
        Parameters
        ----------
        root          : path to folder containing UTKFace .jpg files
        img_size      : (W, H) resize target
        split         : "train" or "val"
        val_fraction  : fraction held out for validation
        seed          : random seed for reproducible split
        augment       : apply data augmentation on train split
        """
        self.img_size = img_size
        self.augment  = augment and (split == "train")

        all_paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        if not all_paths:
            # Some Kaggle versions use .png or nested folders
            all_paths = sorted(glob.glob(os.path.join(root, "**", "*.jpg"),
                                         recursive=True))
        if not all_paths:
            raise FileNotFoundError(
                f"No .jpg images found in '{root}'. "
                "Check UTK_ROOT points at the right directory.")

        # Parse valid samples (skip malformed filenames)
        samples = []
        for p in all_paths:
            parts = os.path.basename(p).split("_")
            if len(parts) < 2:
                continue
            try:
                age    = int(parts[0])
                gender = int(parts[1])
            except ValueError:
                continue
            if age < 0 or age > 116 or gender not in (0, 1):
                continue
            age = min(age, 100)   # cap at 100 for the regression head
            samples.append((p, age, float(gender)))

        # Reproducible train/val split
        rng = random.Random(seed)
        rng.shuffle(samples)
        n_val = max(1, int(len(samples) * val_fraction))
        if split == "val":
            self.samples = samples[:n_val]
        else:
            self.samples = samples[n_val:]

        print(f"[UTKFace] {split}: {len(self.samples)} samples "
              f"(from {len(all_paths)} total found)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, age, gender = self.samples[idx]

        img = cv2.imread(path)
        if img is None:
            # Return a black image rather than crashing the DataLoader
            img = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)

        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            img = self._augment(img)

        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # CHW

        return {
            "image":  torch.from_numpy(img),
            "age":    torch.tensor(float(age),    dtype=torch.float32),
            "gender": torch.tensor(float(gender), dtype=torch.float32),
        }

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Lightweight augmentations safe for face images."""
        # Horizontal flip (50%)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        # Brightness / contrast jitter
        if random.random() < 0.4:
            alpha = random.uniform(0.75, 1.25)  # contrast
            beta  = random.randint(-20, 20)      # brightness
            img   = np.clip(alpha * img.astype(np.float32) + beta,
                            0, 255).astype(np.uint8)

        # Small rotation ±10°
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            h, w  = img.shape[:2]
            M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img   = cv2.warpAffine(img, M, (w, h),
                                   borderMode=cv2.BORDER_REFLECT)
        return img