# ============================================================
#  training/par/dataset_pa100k.py
#
#  PyTorch Dataset for PA-100K → trains the PAR body model.
#
#  Dataset source (request form — academic use only):
#    https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset
#
#  The dataset ships as:
#    release_data/
#      release_data/   ← folder of 100,000 .jpg images
#      release_data/annotation/
#        list_attr_train.txt
#        list_attr_val.txt
#        list_attr_test.txt
#
#  Attribute layout in the annotation files (26 columns):
#    Col 0  : filename
#    Col 1  : Female   (binary)   ← we use this as gender label
#    Col 2  : AgeOver60           ← we map these to age group
#    Col 3  : Age18-60
#    Col 4  : AgeLess18
#    Cols 5–25: clothing / accessory attributes (unused here)
#
#  Age group mapping (from the three binary PA-100K age columns):
#    AgeLess18=1                   → "Child/Teen"   (mapped to idx 0)
#    Age18-60=1                    → "Young/Adult"  (mapped to idx 1)
#    AgeOver60=1                   → "Senior"       (mapped to idx 2)
#    none set (rare noisy label)   → "Young/Adult"  (fallback)
# ============================================================
from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Map our 3-way PA-100K age to the 4-class label space used by PARModel
# PARModel.AGE_LABELS = ["Child", "Young Adult", "Adult", "Senior"]
_PA100K_AGE_TO_IDX = {
    "child_teen":   0,   # "Child"       (AgeLess18)
    "young_adult":  1,   # "Young Adult" (Age18-60, lower half)
    "adult":        2,   # "Adult"       (Age18-60, upper half — PA-100K doesn't split)
    "senior":       3,   # "Senior"      (AgeOver60)
}


class PA100KDataset(Dataset):
    """
    Loads PA-100K body images and returns:
        image       : torch.Tensor (3, H, W) float32, pixel values in [0, 1]
        gender      : torch.Tensor scalar    float32  0=Male / 1=Female
        age_idx     : torch.Tensor scalar    int64    0–3  (4-class)
        orient_idx  : torch.Tensor scalar    int64    0=Front / 1=Side / 2=Back
                      (PA-100K doesn't annotate orientation directly;
                       we return 0 as a placeholder → orientation head is
                       trained from scratch, orient loss is weighted to 0
                       in the training script unless you add your own labels)
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        root: str,
        split: str = "train",           # "train" | "val" | "test"
        img_size: tuple[int, int] = (128, 256),   # (W, H) — PAR input size
        augment: bool = True,
    ):
        """
        Parameters
        ----------
        root     : path to the 'release_data' folder (contains images + annotation/)
        split    : one of "train", "val", "test"
        img_size : (W, H)
        augment  : apply augmentation on train split
        """
        self.img_dir  = os.path.join(root, "release_data")
        self.img_size = img_size
        self.augment  = augment and (split == "train")

        ann_file = os.path.join(root, "annotation", f"list_attr_{split}.txt")
        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                "Make sure root points at the 'release_data' parent directory.")

        self.samples = self._parse(ann_file)
        print(f"[PA-100K] {split}: {len(self.samples)} samples")

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, ann_file: str) -> list[tuple]:
        samples = []
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                fname    = parts[0]
                female   = int(parts[1])    # 1=Female, 0=Male
                over60   = int(parts[2])
                age1860  = int(parts[3])
                less18   = int(parts[4])

                # Derive 4-class age index
                if less18:
                    age_idx = 0   # Child
                elif age1860 and not over60:
                    age_idx = 2   # Adult  (PA-100K doesn't split 18-60)
                elif over60:
                    age_idx = 3   # Senior
                else:
                    age_idx = 1   # Young Adult (fallback)

                samples.append((fname, float(female), age_idx))
        return samples

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fname, gender, age_idx = self.samples[idx]
        path = os.path.join(self.img_dir, fname)

        img = cv2.imread(path)
        if img is None:
            img = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)

        img = cv2.resize(img, self.img_size)   # (W, H) → (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            img = self._augment(img)

        img = img.astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # CHW

        return {
            "image":      torch.from_numpy(img),
            "gender":     torch.tensor(gender,   dtype=torch.float32),
            "age_idx":    torch.tensor(age_idx,  dtype=torch.long),
            "orient_idx": torch.tensor(0,         dtype=torch.long),  # placeholder
        }

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # Horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        # Brightness/contrast
        if random.random() < 0.4:
            alpha = random.uniform(0.8, 1.2)
            beta  = random.randint(-15, 15)
            img   = np.clip(alpha * img.astype(np.float32) + beta,
                            0, 255).astype(np.uint8)
        # Random crop (pad 10px, then crop back)
        if random.random() < 0.3:
            h, w = img.shape[:2]
            pad  = 10
            img  = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                      cv2.BORDER_REFLECT)
            x = random.randint(0, 2 * pad)
            y = random.randint(0, 2 * pad)
            img = img[y:y + h, x:x + w]
        return img