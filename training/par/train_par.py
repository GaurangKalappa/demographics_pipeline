# ============================================================
#  training/par/train_par.py
#
#  Trains the MobileNetV3-Large PAR model (gender + age group + orientation)
#  on PA-100K, then saves weights to:
#      demographics_pipeline/weights/par_model.pt
#
#  Usage:
#      python training/par/train_par.py \
#          --data   /path/to/release_data_parent \
#          --out    weights/par_model.pt \
#          --epochs 40 \
#          --batch  128 \
#          --lr     1e-3
#
#  PA-100K has no orientation labels, so the orient_head is NOT trained
#  here (its loss weight is 0.0 by default).  The head still initialises and
#  you can supply orientation labels later to fine-tune it.
# ============================================================
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.par_model import PARModel
from training.par.dataset_pa100k import PA100KDataset


# ── Loss helpers ──────────────────────────────────────────────────────────────

def gender_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logit.squeeze(1), target)


def age_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logit, target)


def orient_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logit, target)


# ── Training / eval loops ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, device,
                    gw, aw, ow):
    model.train()
    t_loss = t_g = t_a = 0.0

    for batch in loader:
        imgs    = batch["image"].to(device)
        genders = batch["gender"].to(device)
        age_idx = batch["age_idx"].to(device)
        ori_idx = batch["orient_idx"].to(device)

        optimiser.zero_grad()
        out = model(imgs)

        lg = gender_loss(out["gender"], genders)
        la = age_loss(out["age"],    age_idx)
        lo = orient_loss(out["orient"], ori_idx) if ow > 0 else torch.tensor(0.0)

        loss = gw * lg + aw * la + ow * lo
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()

        t_loss += loss.item()
        t_g    += lg.item()
        t_a    += la.item()

    n = len(loader)
    return t_loss / n, t_g / n, t_a / n


@torch.no_grad()
def evaluate(model, loader, device, gw, aw, ow):
    model.eval()
    t_loss = 0.0
    g_correct = a_correct = total = 0

    for batch in loader:
        imgs    = batch["image"].to(device)
        genders = batch["gender"].to(device)
        age_idx = batch["age_idx"].to(device)
        ori_idx = batch["orient_idx"].to(device)

        out = model(imgs)
        lg = gender_loss(out["gender"], genders)
        la = age_loss(out["age"], age_idx)
        lo = orient_loss(out["orient"], ori_idx) if ow > 0 else torch.tensor(0.0)

        t_loss += (gw * lg + aw * la + ow * lo).item()

        pred_g = (torch.sigmoid(out["gender"].squeeze(1)) >= 0.5).float()
        g_correct += (pred_g == genders).sum().item()

        pred_a = out["age"].argmax(dim=1)
        a_correct += (pred_a == age_idx).sum().item()

        total += genders.size(0)

    n = len(loader)
    return (t_loss / n,
            g_correct / total * 100,
            a_correct / total * 100)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True,
                   help="Path to PA-100K release_data parent folder")
    p.add_argument("--out",      default="weights/par_model.pt")
    p.add_argument("--epochs",   type=int,   default=40)
    p.add_argument("--batch",    type=int,   default=128)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--workers",  type=int,   default=4)
    p.add_argument("--gender-w", type=float, default=1.0)
    p.add_argument("--age-w",    type=float, default=1.0)
    p.add_argument("--orient-w", type=float, default=0.0,
                   help="Set >0 only if you have orientation labels")
    p.add_argument("--patience", type=int,   default=6)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Data   : {args.data}")
    print(f"Output : {args.out}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = PA100KDataset(args.data, split="train", augment=True)
    val_ds   = PA100KDataset(args.data, split="val",   augment=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PARModel(device=str(device)).to(device)

    # ── Two-phase training ────────────────────────────────────────────────────
    for p in model.features.parameters():
        p.requires_grad = False

    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    best_val = float("inf")
    patience = 0
    unfreeze = args.epochs // 2

    print(f"{'Ep':>4} {'TrLoss':>8} {'TrG':>6} {'TrA':>6} "
          f"{'ValLoss':>8} {'GndAcc%':>8} {'AgeAcc%':>8}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):

        if epoch == unfreeze:
            print(f"\n[Epoch {epoch}] Unfreezing backbone.\n")
            for p in model.features.parameters():
                p.requires_grad = True
            optimiser = torch.optim.Adam([
                {"params": model.features.parameters(),    "lr": args.lr * 0.1},
                {"params": model.pool.parameters(),         "lr": args.lr * 0.1},
                {"params": model.gender_head.parameters(), "lr": args.lr},
                {"params": model.age_head.parameters(),    "lr": args.lr},
                {"params": model.orient_head.parameters(), "lr": args.lr},
            ])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=args.epochs - unfreeze)

        tl, tg, ta = train_one_epoch(
            model, train_ld, optimiser, device,
            args.gender_w, args.age_w, args.orient_w)
        vl, g_acc, a_acc = evaluate(
            model, val_ld, device,
            args.gender_w, args.age_w, args.orient_w)
        scheduler.step()

        print(f"{epoch:>4} {tl:>8.4f} {tg:>6.4f} {ta:>6.4f} "
              f"{vl:>8.4f} {g_acc:>8.1f} {a_acc:>8.1f}")

        if vl < best_val:
            best_val = vl
            patience = 0
            torch.save(model.state_dict(), args.out)
            print(f"       ✔  Saved → {args.out}")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\nEarly stop at epoch {epoch}.")
                break

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Weights: {args.out}")


if __name__ == "__main__":
    main()