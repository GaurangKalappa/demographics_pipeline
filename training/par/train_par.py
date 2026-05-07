# ============================================================
#  training/par/train_par.py
#
#  Trains the MobileNetV3-Large PAR model (gender + age group + orientation)
#  on PA-100K, then saves weights to:
#      weights/par_model.pt
#
#  Checkpoint files (full training state, resumable):
#      weights/checkpoints/par_model/checkpoint_last.pt  ← every epoch
#      weights/checkpoints/par_model/checkpoint_best.pt  ← on val improvement
#
#  Training metrics log:
#      weights/checkpoints/par_model/train_log.csv
#
#  Usage – fresh start:
#      python training/par/train_par.py \
#          --data  /path/to/PA-100K \
#          --out   weights/par_model.pt \
#          --epochs 40 --batch 128 --lr 1e-3
#
#  Usage – resume after crash / interrupt:
#      python training/par/train_par.py \
#          --data  /path/to/PA-100K \
#          --out   weights/par_model.pt \
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

from models.par_model import PARModel
from training.par.dataset_pa100k import PA100KDataset
from training.checkpoint import (
    checkpoint_dir, save_checkpoint, save_best, load_checkpoint,
    TrainingLogger, setup_signal_handler, interrupt_requested,
)


# ── Loss helpers ──────────────────────────────────────────────────────────────

def gender_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logit.squeeze(1), target)


def age_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logit, target)


def orient_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logit, target)


# ── Optimiser builders ────────────────────────────────────────────────────────

def build_phase1_optimiser(model: PARModel, lr: float):
    """Phase 1: backbone frozen."""
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def build_phase2_optimiser(model: PARModel, lr: float):
    """Phase 2: full model, backbone at 10× lower LR."""
    return torch.optim.Adam([
        {"params": model.features.parameters(),    "lr": lr * 0.1},
        {"params": model.pool.parameters(),         "lr": lr * 0.1},
        {"params": model.gender_head.parameters(),  "lr": lr},
        {"params": model.age_head.parameters(),     "lr": lr},
        {"params": model.orient_head.parameters(),  "lr": lr},
    ])


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, device, gw, aw, ow):
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
        la = age_loss(out["age"], age_idx)
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
    return t_loss / n, g_correct / total * 100, a_correct / total * 100


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      required=True,
                   help="Path to PA-100K root folder (contains train.csv, val.csv, test.csv and data/ subfolder)")
    p.add_argument("--out",       default="weights/par_model.pt",
                   help="Final plug-in weights path")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from checkpoint_last.pt in the checkpoint dir")
    p.add_argument("--epochs",    type=int,   default=40)
    p.add_argument("--batch",     type=int,   default=128)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--workers",   type=int,   default=4)
    p.add_argument("--gender-w",  type=float, default=1.0)
    p.add_argument("--age-w",     type=float, default=1.0)
    p.add_argument("--orient-w",  type=float, default=1.0,
                   help="Orientation loss weight — Front/Side/Back labels are present in the Kaggle CSV")
    p.add_argument("--patience",  type=int,   default=6)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_d = checkpoint_dir(args.out)
    logger = TrainingLogger(ckpt_d)
    setup_signal_handler()

    print(f"\nDevice      : {device}")
    print(f"Data        : {args.data}")
    print(f"Output      : {args.out}")
    print(f"Checkpoints : {ckpt_d}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = PA100KDataset(args.data, split="train", augment=True)
    val_ds   = PA100KDataset(args.data, split="val",   augment=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PARModel(device=str(device)).to(device)

    # ── Training state defaults ────────────────────────────────────────────────
    start_epoch    = 1
    best_val       = float("inf")
    patience_count = 0
    history: list[dict] = []
    unfreeze_epoch = args.epochs // 2

    # ── Phase 1 setup ─────────────────────────────────────────────────────────
    for p in model.features.parameters():
        p.requires_grad = False
    phase      = 1
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
    print(f"{'Ep':>4} {'Ph':>2} {'TrLoss':>8} {'TrG':>6} {'TrA':>6} "
          f"{'ValLoss':>8} {'GndAcc%':>8} {'AgeAcc%':>8}")
    print("─" * 66)

    for epoch in range(start_epoch, args.epochs + 1):

        # ── Phase transition ───────────────────────────────────────────────
        if phase == 1 and epoch >= unfreeze_epoch:
            print(f"\n[Epoch {epoch}] Phase 2 – unfreezing backbone.\n")
            for p in model.features.parameters():
                p.requires_grad = True
            phase      = 2
            sched_tmax = max(args.epochs - unfreeze_epoch, 1)
            optimiser  = build_phase2_optimiser(model, args.lr)
            scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=sched_tmax)

        tl, tg, ta = train_one_epoch(
            model, train_ld, optimiser, device,
            args.gender_w, args.age_w, args.orient_w)
        vl, g_acc, a_acc = evaluate(
            model, val_ld, device,
            args.gender_w, args.age_w, args.orient_w)
        scheduler.step()

        improved = vl < best_val
        if improved:
            best_val       = vl
            patience_count = 0
        else:
            patience_count += 1

        row = {
            "epoch": epoch, "phase": phase,
            "train_loss": round(tl, 5), "gender_loss": round(tg, 5),
            "age_loss": round(ta, 5), "val_loss": round(vl, 5),
            "gender_acc": round(g_acc, 3), "age_acc": round(a_acc, 3),
            "best_val": round(best_val, 5), "patience": patience_count,
            "improved": improved,
        }
        history.append(row)
        logger.log(row)

        marker = " ✔" if improved else ""
        print(f"{epoch:>4} {phase:>2} {tl:>8.4f} {tg:>6.4f} {ta:>6.4f} "
              f"{vl:>8.4f} {g_acc:>8.1f} {a_acc:>8.1f}{marker}")

        if improved:
            save_best(ckpt_d, args.out, epoch, model, optimiser, scheduler,
                      best_val, patience_count, phase, unfreeze_epoch,
                      sched_tmax, history, vars(args))
            print(f"     ✔  Best checkpoint + weights saved → {args.out}")

        save_checkpoint(ckpt_d, epoch, model, optimiser, scheduler,
                        best_val, patience_count, phase, unfreeze_epoch,
                        sched_tmax, history, vars(args))

        if patience_count >= args.patience:
            print(f"\nEarly stop – no improvement for {args.patience} epochs.")
            break

        if interrupt_requested():
            print("\n[Signal] Checkpoint saved. Exiting cleanly.")
            sys.exit(0)

    print(f"\nTraining complete.  Best val loss : {best_val:.4f}")
    print(f"Plug-in weights   : {args.out}")
    print(f"Checkpoint dir    : {ckpt_d}")


if __name__ == "__main__":
    main()