# ============================================================
#  training/par/dataset_pa100k.py
#
#  PyTorch Dataset for PA-100K → trains the PAR body model.
#
#  Kaggle source (direct download, no request form required):
#    https://www.kaggle.com/datasets/twpeixinho/pa-100k
#
#  Expected folder layout after download + extraction:
#    <root>/
#      data/               ← 100,000 .jpg pedestrian images
#      train.csv           ← training split annotations
#      val.csv             ← validation split annotations
#      test.csv            ← test split annotations
#      annotation.mat      ← original MATLAB file (not used here)
#      README.txt
#
#  CSV column layout (27 columns):
#    Image        : filename  e.g. "090001.jpg"
#    Female       : 1=Female / 0=Male
#    AgeOver60    : binary
#    Age18-60     : binary
#    AgeLess18    : binary
#    Front        : binary  ← orientation labels now available
#    Side         : binary
#    Back         : binary
#    Hat, Glasses, HandBag, ShoulderBag, Backpack,
#    HoldObjects, ShortSleeve, LongSleeve, UpperStride,
#    UpperLogo, UpperPlaid, UpperSplice, LowerStripe,
#    LowerPattern, LongCoat, Trousers, Shorts,
#    Skirt&Dress, boots      ← remaining attributes (unused)
#
#  Age group mapping → PARModel.AGE_LABELS index:
#    AgeLess18 = 1           → 0  "Child"
#    Age18-60  = 1           → 2  "Adult"   (PA-100K does not split 18-60)
#    AgeOver60 = 1           → 3  "Senior"
#    none set  (noisy label) → 1  "Young Adult"  (fallback)
#
#  Orientation mapping → PARModel.ORIENT_LABELS index:
#    Front = 1               → 0
#    Side  = 1               → 1
#    Back  = 1               → 2
#    none set (noisy label)  → 0  "Front"  (fallback)
# ============================================================
from __future__ import annotations

import csv
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PA100KDataset(Dataset):
    """
    Loads PA-100K body images from the Kaggle layout and returns:
        image       : torch.Tensor (3, H, W)  float32, pixel values in [0, 1]
        gender      : torch.Tensor  scalar    float32  0=Male / 1=Female
        age_idx     : torch.Tensor  scalar    int64    0–3  (4-class)
        orient_idx  : torch.Tensor  scalar    int64    0=Front / 1=Side / 2=Back
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        root: str,
        split: str = "train",                   # "train" | "val" | "test"
        img_size: tuple[int, int] = (128, 256), # (W, H) — PAR input size
        augment: bool = True,
    ):
        """
        Parameters
        ----------
        root     : path to the PA-100K root folder that contains
                   train.csv / val.csv / test.csv and the data/ subfolder.
        split    : "train", "val", or "test"
        img_size : (W, H) resize target
        augment  : apply augmentation (only active on the train split)
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.img_dir  = os.path.join(root, "data")
        self.img_size = img_size
        self.augment  = augment and (split == "train")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(
                f"Image folder not found: {self.img_dir}\n"
                "Make sure --data points at the PA-100K root folder "
                "that contains both the data/ subfolder and the CSV files.")

        csv_path = os.path.join(root, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Annotation CSV not found: {csv_path}\n"
                "Make sure --data points at the folder that contains "
                "train.csv / val.csv / test.csv.")

        self.samples = self._parse(csv_path)
        print(f"[PA-100K] {split:5s}: {len(self.samples)} samples  "
              f"(images in {self.img_dir})")

    # ── CSV parsing ───────────────────────────────────────────────────────────

    def _parse(self, csv_path: str) -> list[tuple]:
        """
        Parse one CSV file into a list of
            (filename, gender_float, age_idx, orient_idx)
        tuples.
        """
        samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["Image"].strip()

                # ── Gender ────────────────────────────────────────────────
                female = int(row["Female"])   # 1=Female, 0=Male

                # ── Age group → 4-class index ─────────────────────────────
                over60  = int(row["AgeOver60"])
                age1860 = int(row["Age18-60"])
                less18  = int(row["AgeLess18"])

                if less18:
                    age_idx = 0          # Child
                elif over60:
                    age_idx = 3          # Senior
                elif age1860:
                    age_idx = 2          # Adult (18-60 not split further)
                else:
                    age_idx = 1          # Young Adult — noisy-label fallback

                # ── Orientation → 3-class index ───────────────────────────
                # Front / Side / Back are present in the Kaggle CSVs.
                front = int(row["Front"])
                side  = int(row["Side"])
                back  = int(row["Back"])

                if front:
                    orient_idx = 0
                elif side:
                    orient_idx = 1
                elif back:
                    orient_idx = 2
                else:
                    orient_idx = 0       # Front — noisy-label fallback

                samples.append((fname, float(female), age_idx, orient_idx))

        return samples

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fname, gender, age_idx, orient_idx = self.samples[idx]
        path = os.path.join(self.img_dir, fname)

        img = cv2.imread(path)
        if img is None:
            # Corrupt / missing image → return a black frame rather than crash
            img = np.zeros((self.img_size[1], self.img_size[0], 3),
                           dtype=np.uint8)

        img = cv2.resize(img, self.img_size)          # uint8 BGR (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # uint8 RGB (H, W, 3)

        if self.augment:
            img = self._augment(img)

        img = img.astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # CHW float32

        return {
            "image":      torch.from_numpy(img),
            "gender":     torch.tensor(gender,     dtype=torch.float32),
            "age_idx":    torch.tensor(age_idx,    dtype=torch.long),
            "orient_idx": torch.tensor(orient_idx, dtype=torch.long),
        }

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Lightweight augmentations suitable for pedestrian body crops."""
        # Horizontal flip (50%)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        # Brightness / contrast jitter (40%)
        if random.random() < 0.4:
            alpha = random.uniform(0.8, 1.2)
            beta  = random.randint(-15, 15)
            img   = np.clip(alpha * img.astype(np.float32) + beta,
                            0, 255).astype(np.uint8)

        # Random crop: pad 10 px each side, then crop back to original size
        if random.random() < 0.3:
            h, w = img.shape[:2]
            pad  = 10
            img  = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                      cv2.BORDER_REFLECT)
            x = random.randint(0, 2 * pad)
            y = random.randint(0, 2 * pad)
            img = img[y:y + h, x:x + w]

        return img