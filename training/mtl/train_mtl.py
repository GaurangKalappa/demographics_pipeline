# ============================================================
#  training/mtl/train_mtl.py
#
#  Trains the MobileNetV3-Small MTL model (age regression + gender
#  classification) on UTKFace, then saves weights to:
#      weights/mtl_model.pt
#
#  Checkpoint files (full training state, resumable):
#      weights/checkpoints/mtl_model/checkpoint_last.pt  ← every epoch
#      weights/checkpoints/mtl_model/checkpoint_best.pt  ← on val improvement
#
#  Training metrics log:
#      weights/checkpoints/mtl_model/train_log.csv
#
#  Usage – fresh start:
#      python training/mtl/train_mtl.py \
#          --data  /path/to/UTKFace \
#          --out   weights/mtl_model.pt \
#          --epochs 30 --batch 64 --lr 1e-3
#
#  Usage – resume after crash / interrupt:
#      python training/mtl/train_mtl.py \
#          --data  /path/to/UTKFace \
#          --out   weights/mtl_model.pt \
#          --resume
#          (all other flags are restored from the checkpoint automatically)
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

from models.mtl_model import MTLModel
from training.mtl.dataset_utkface import UTKFaceDataset
from training.checkpoint import (
    checkpoint_dir, save_checkpoint, save_best, load_checkpoint,
    TrainingLogger, setup_signal_handler, interrupt_requested,
)


# ── Loss helpers ──────────────────────────────────────────────────────────────

def age_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on age, both normalised to [0, 1]."""
    return nn.functional.mse_loss(pred.squeeze(1) / 100.0, target / 100.0)


def gender_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logit.squeeze(1), target)


# ── Optimiser builders ────────────────────────────────────────────────────────

def build_phase1_optimiser(model: MTLModel, lr: float):
    """Phase 1: backbone frozen, only task heads are updated."""
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def build_phase2_optimiser(model: MTLModel, lr: float):
    """Phase 2: full model, backbone at 10× lower LR."""
    return torch.optim.Adam([
        {"params": model.features.parameters(),   "lr": lr * 0.1},
        {"params": model.pool.parameters(),        "lr": lr * 0.1},
        {"params": model.age_head.parameters(),    "lr": lr},
        {"params": model.gender_head.parameters(), "lr": lr},
    ])


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, device, age_w, gender_w):
    model.train()
    total_loss = total_age = total_gender = 0.0

    for batch in loader:
        imgs    = batch["image"].to(device)
        ages    = batch["age"].to(device)
        genders = batch["gender"].to(device)

        optimiser.zero_grad()
        out = model(imgs)

        l_age    = age_loss(out["age"], ages)
        l_gender = gender_loss(out["gender"], genders)
        loss     = age_w * l_age + gender_w * l_gender

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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

        pred_age = torch.clamp(out["age"].squeeze(1), 0.0, 100.0)
        mae += (pred_age - ages).abs().sum().item()

        pred_g  = (torch.sigmoid(out["gender"].squeeze(1)) >= 0.5).float()
        correct += (pred_g == genders).sum().item()
        total   += genders.size(0)

    n = len(loader)
    return total_loss / n, mae / total, correct / total * 100


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True,
                   help="Path to UTKFace image folder")
    p.add_argument("--out",      default="weights/mtl_model.pt",
                   help="Final plug-in weights path")
    p.add_argument("--resume",   action="store_true",
                   help="Resume from checkpoint_last.pt in the checkpoint dir")
    p.add_argument("--epochs",   type=int,   default=30)
    p.add_argument("--batch",    type=int,   default=64)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--workers",  type=int,   default=4)
    p.add_argument("--age-w",    type=float, default=1.0)
    p.add_argument("--gender-w", type=float, default=1.0)
    p.add_argument("--patience", type=int,   default=5)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_d  = checkpoint_dir(args.out)
    logger  = TrainingLogger(ckpt_d)
    setup_signal_handler()

    print(f"\nDevice      : {device}")
    print(f"Data        : {args.data}")
    print(f"Output      : {args.out}")
    print(f"Checkpoints : {ckpt_d}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = UTKFaceDataset(args.data, split="train", augment=True)
    val_ds   = UTKFaceDataset(args.data, split="val",   augment=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # ── Model (always built fresh; weights loaded below if resuming) ───────────
    model = MTLModel(device=str(device)).to(device)

    # ── Training state defaults ────────────────────────────────────────────────
    start_epoch    = 1
    best_val       = float("inf")
    patience_count = 0
    history: list[dict] = []
    unfreeze_epoch = args.epochs // 2

    # ── Phase 1 setup (may be overridden on resume) ────────────────────────────
    for p in model.features.parameters():
        p.requires_grad = False
    phase     = 1
    sched_tmax = args.epochs
    optimiser  = build_phase1_optimiser(model, args.lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=sched_tmax)

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = load_checkpoint(ckpt_d)
        if ckpt is None:
            print("[Resume] No checkpoint found – starting from scratch.\n")
        else:
            model.load_state_dict(ckpt["model_state"])
            start_epoch    = ckpt["epoch"] + 1
            best_val       = ckpt["best_val"]
            patience_count = ckpt["patience"]
            phase          = ckpt["phase"]
            unfreeze_epoch = ckpt["unfreeze_epoch"]
            sched_tmax     = ckpt["sched_t_max"]
            history        = ckpt.get("history", [])

            # Reconstruct the correct optimiser for the saved phase
            if phase == 2:
                for p in model.features.parameters():
                    p.requires_grad = True
                optimiser = build_phase2_optimiser(model, args.lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=sched_tmax)
            else:
                optimiser = build_phase1_optimiser(model, args.lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=sched_tmax)

            optimiser.load_state_dict(ckpt["optim_state"])
            scheduler.load_state_dict(ckpt["sched_state"])
            print(f"[Resume] Resuming from epoch {start_epoch}  "
                  f"(phase {phase}, best_val={best_val:.4f})\n")

    if start_epoch > args.epochs:
        print("Training already complete (checkpoint epoch >= --epochs). Done.")
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"{'Epoch':>6} {'Phase':>5} {'TrainL':>8} {'AgeL':>7} "
          f"{'GendL':>7} {'ValL':>8} {'MAE(yr)':>8} {'GndAcc%':>8}")
    print("─" * 70)

    for epoch in range(start_epoch, args.epochs + 1):

        # ── Phase transition: unfreeze backbone at midpoint ────────────────
        if phase == 1 and epoch >= unfreeze_epoch:
            print(f"\n[Epoch {epoch}] Phase 2 – unfreezing backbone.\n")
            for p in model.features.parameters():
                p.requires_grad = True
            phase      = 2
            sched_tmax = args.epochs - unfreeze_epoch
            optimiser  = build_phase2_optimiser(model, args.lr)
            scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=max(sched_tmax, 1))

        t_loss, t_age, t_gender = train_one_epoch(
            model, train_ld, optimiser, device, args.age_w, args.gender_w)
        v_loss, mae, g_acc = evaluate(
            model, val_ld, device, args.age_w, args.gender_w)
        scheduler.step()

        improved = v_loss < best_val
        if improved:
            best_val       = v_loss
            patience_count = 0
        else:
            patience_count += 1

        # ── Log row ────────────────────────────────────────────────────────
        row = {
            "epoch": epoch, "phase": phase,
            "train_loss": round(t_loss, 5), "age_loss": round(t_age, 5),
            "gender_loss": round(t_gender, 5), "val_loss": round(v_loss, 5),
            "mae_years": round(mae, 3), "gender_acc": round(g_acc, 3),
            "best_val": round(best_val, 5), "patience": patience_count,
            "improved": improved,
        }
        history.append(row)
        logger.log(row)

        marker = " ✔" if improved else ""
        print(f"{epoch:>6} {phase:>5} {t_loss:>8.4f} {t_age:>7.4f} "
              f"{t_gender:>7.4f} {v_loss:>8.4f} {mae:>8.2f} {g_acc:>8.1f}{marker}")

        # ── Save best (full checkpoint + plug-in weights) ──────────────────
        if improved:
            save_best(ckpt_d, args.out, epoch, model, optimiser, scheduler,
                      best_val, patience_count, phase, unfreeze_epoch,
                      sched_tmax, history, vars(args))
            print(f"         ✔  Best checkpoint + weights saved → {args.out}")

        # ── Save last (always – for crash recovery) ────────────────────────
        save_checkpoint(ckpt_d, epoch, model, optimiser, scheduler,
                        best_val, patience_count, phase, unfreeze_epoch,
                        sched_tmax, history, vars(args))

        # ── Early stopping ─────────────────────────────────────────────────
        if patience_count >= args.patience:
            print(f"\nEarly stop – no improvement for {args.patience} epochs.")
            break

        # ── Graceful interrupt (SIGINT / SIGTERM) ──────────────────────────
        if interrupt_requested():
            print("\n[Signal] Checkpoint saved. Exiting cleanly.")
            sys.exit(0)

    print(f"\nTraining complete.  Best val loss : {best_val:.4f}")
    print(f"Plug-in weights   : {args.out}")
    print(f"Checkpoint dir    : {ckpt_d}")


if __name__ == "__main__":
    main()