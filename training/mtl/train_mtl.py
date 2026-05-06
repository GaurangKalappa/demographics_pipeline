# ============================================================
#  training/mtl/train_mtl.py
#
#  Trains the MobileNetV3-Small MTL model (age regression + gender
#  classification) on UTKFace, then saves weights to:
#      demographics_pipeline/weights/mtl_model.pt
#
#  Usage:
#      python training/mtl/train_mtl.py \
#          --data  /path/to/UTKFace \
#          --out   weights/mtl_model.pt \
#          --epochs 30 \
#          --batch  64 \
#          --lr     1e-3
#
#  Recommended hardware: GPU (any CUDA card).
#  On CPU the run is slower but still finishes in ~2 h for 30 epochs.
# ============================================================
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Make sure the project root is importable ──────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.mtl_model import MTLModel
from training.mtl.dataset_utkface import UTKFaceDataset


# ── Loss helpers ──────────────────────────────────────────────────────────────

def age_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on the raw age regression output (both normalised to [0,1])."""
    pred_norm   = pred.squeeze(1)   / 100.0
    target_norm = target            / 100.0
    return nn.functional.mse_loss(pred_norm, target_norm)


def gender_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for gender (target: 0=Male, 1=Female)."""
    return nn.functional.binary_cross_entropy_with_logits(
        logit.squeeze(1), target)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, device, age_w, gender_w):
    model.train()
    total_loss = total_age = total_gender = 0.0

    for batch in loader:
        imgs   = batch["image"].to(device)
        ages   = batch["age"].to(device)
        genders= batch["gender"].to(device)

        optimiser.zero_grad()
        out = model(imgs)

        l_age    = age_loss(out["age"], ages)
        l_gender = gender_loss(out["gender"], genders)
        loss     = age_w * l_age + gender_w * l_gender

        loss.backward()
        optimiser.step()

        total_loss   += loss.item()
        total_age    += l_age.item()
        total_gender += l_gender.item()

    n = len(loader)
    return total_loss / n, total_age / n, total_gender / n


@torch.no_grad()
def evaluate(model, loader, device, age_w, gender_w):
    model.eval()
    total_loss = mae = correct = total = 0.0

    for batch in loader:
        imgs    = batch["image"].to(device)
        ages    = batch["age"].to(device)
        genders = batch["gender"].to(device)

        out = model(imgs)

        l_age    = age_loss(out["age"], ages)
        l_gender = gender_loss(out["gender"], genders)
        total_loss += (age_w * l_age + gender_w * l_gender).item()

        # MAE in years
        pred_age = torch.clamp(out["age"].squeeze(1), 0.0, 100.0)
        mae += (pred_age - ages).abs().sum().item()

        # Gender accuracy
        pred_g  = (torch.sigmoid(out["gender"].squeeze(1)) >= 0.5).float()
        correct += (pred_g == genders).sum().item()
        total   += genders.size(0)

    n = len(loader)
    return (total_loss / n,
            mae / total,
            correct / total * 100)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    required=True,
                   help="Path to UTKFace image folder")
    p.add_argument("--out",     default="weights/mtl_model.pt",
                   help="Output weights path")
    p.add_argument("--epochs",  type=int,   default=30)
    p.add_argument("--batch",   type=int,   default=64)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--age-w",   type=float, default=1.0,
                   help="Age loss weight")
    p.add_argument("--gender-w",type=float, default=1.0,
                   help="Gender loss weight")
    p.add_argument("--patience",type=int,   default=5,
                   help="Early stopping patience (val loss)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Data   : {args.data}")
    print(f"Output : {args.out}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = UTKFaceDataset(args.data, split="train", augment=True)
    val_ds   = UTKFaceDataset(args.data, split="val",   augment=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MTLModel(device=str(device)).to(device)

    # ── Optimiser: two-phase learning rate ────────────────────────────────────
    # Phase 1 (first half of training): only heads are trainable
    # Phase 2 (second half): full model fine-tunes at a lower LR
    for p in model.features.parameters():
        p.requires_grad = False

    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    best_val_loss   = float("inf")
    patience_count  = 0
    unfreeze_epoch  = args.epochs // 2   # unfreeze backbone halfway through

    print(f"{'Epoch':>6} {'TrainL':>8} {'AgeL':>7} {'GendL':>7} "
          f"{'ValL':>7} {'MAE(yr)':>8} {'GndAcc%':>8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):

        # ── Phase 2: unfreeze backbone at midpoint ─────────────────────────
        if epoch == unfreeze_epoch:
            print(f"\n[Epoch {epoch}] Unfreezing backbone for fine-tuning.\n")
            for p in model.features.parameters():
                p.requires_grad = True
            # Lower LR for backbone
            optimiser = torch.optim.Adam([
                {"params": model.features.parameters(),  "lr": args.lr * 0.1},
                {"params": model.pool.parameters(),      "lr": args.lr * 0.1},
                {"params": model.age_head.parameters(),  "lr": args.lr},
                {"params": model.gender_head.parameters(),"lr": args.lr},
            ])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=args.epochs - unfreeze_epoch)

        t_loss, t_age, t_gender = train_one_epoch(
            model, train_ld, optimiser, device, args.age_w, args.gender_w)
        v_loss, mae, g_acc = evaluate(
            model, val_ld, device, args.age_w, args.gender_w)
        scheduler.step()

        print(f"{epoch:>6} {t_loss:>8.4f} {t_age:>7.4f} {t_gender:>7.4f} "
              f"{v_loss:>7.4f} {mae:>8.2f} {g_acc:>8.1f}")

        # ── Save best checkpoint ───────────────────────────────────────────
        if v_loss < best_val_loss:
            best_val_loss  = v_loss
            patience_count = 0
            torch.save(model.state_dict(), args.out)
            print(f"          ✔  Saved best model → {args.out}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to : {args.out}")


if __name__ == "__main__":
    main()