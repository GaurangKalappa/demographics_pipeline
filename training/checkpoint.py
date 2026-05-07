# ============================================================
#  training/checkpoint.py
#
#  Shared utilities for all training scripts:
#    - atomic_save()       : crash-safe file write
#    - save_checkpoint()   : full training state → checkpoint_last.pt
#    - save_best()         : best weights only  → checkpoint_best.pt + out
#    - load_checkpoint()   : restore full state from checkpoint_last.pt
#    - TrainingLogger      : append metrics to a CSV log per epoch
#    - setup_signal_handler: catch SIGINT/SIGTERM → save before exit
# ============================================================
from __future__ import annotations

import csv
import os
import shutil
import signal
import sys
import tempfile
from datetime import datetime
from typing import Any

import torch


# ═══════════════════════════════════════════════════════════════════════════════
#  Atomic file write
# ═══════════════════════════════════════════════════════════════════════════════

def atomic_save(obj: Any, path: str) -> None:
    """
    Write `obj` to `path` safely.

    Strategy: write to a sibling temp file first, then rename.
    On POSIX, rename() is atomic — a crash during save leaves the
    original file untouched.  On Windows, shutil.move() replaces
    atomically when src/dst are on the same filesystem.
    """
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(obj, tmp)
        shutil.move(tmp, path)
    except Exception:
        # Clean up the temp file if anything went wrong
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint save / load
# ═══════════════════════════════════════════════════════════════════════════════

def checkpoint_dir(out_path: str) -> str:
    """
    Derive the checkpoint directory from the final output weight path.
    Example: weights/mtl_model.pt  →  weights/checkpoints/mtl_model/
    """
    stem = os.path.splitext(os.path.basename(out_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(out_path)),
                        "checkpoints", stem)


def save_checkpoint(
    ckpt_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: Any,
    best_val: float,
    patience: int,
    phase: int,
    unfreeze_epoch: int,
    sched_t_max: int,
    history: list[dict],
    args_dict: dict,
) -> None:
    """
    Save a FULL training state to `checkpoint_last.pt`.
    Called at the end of every epoch regardless of whether val improved.
    Uses atomic_save so a crash during write cannot corrupt the file.
    """
    payload = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optim_state":    optimiser.state_dict(),
        "sched_state":    scheduler.state_dict(),
        "best_val":       best_val,
        "patience":       patience,
        "phase":          phase,
        "unfreeze_epoch": unfreeze_epoch,
        "sched_t_max":    sched_t_max,
        "history":        history,
        "args":           args_dict,
        "saved_at":       datetime.now().isoformat(timespec="seconds"),
    }
    path = os.path.join(ckpt_dir, "checkpoint_last.pt")
    atomic_save(payload, path)


def save_best(
    ckpt_dir: str,
    out_path: str,
    epoch: int,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: Any,
    best_val: float,
    patience: int,
    phase: int,
    unfreeze_epoch: int,
    sched_t_max: int,
    history: list[dict],
    args_dict: dict,
) -> None:
    """
    Save:
      1. Full training state → checkpoint_best.pt  (resumable)
      2. Model weights only  → {out_path}          (plug-in ready)

    Two separate atomic writes so both are always consistent.
    """
    payload = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optim_state":    optimiser.state_dict(),
        "sched_state":    scheduler.state_dict(),
        "best_val":       best_val,
        "patience":       patience,
        "phase":          phase,
        "unfreeze_epoch": unfreeze_epoch,
        "sched_t_max":    sched_t_max,
        "history":        history,
        "args":           args_dict,
        "saved_at":       datetime.now().isoformat(timespec="seconds"),
    }
    best_path = os.path.join(ckpt_dir, "checkpoint_best.pt")
    atomic_save(payload, best_path)

    # Plug-in weights (model state_dict only)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    atomic_save(model.state_dict(), out_path)


def load_checkpoint(ckpt_dir: str) -> dict | None:
    """
    Load `checkpoint_last.pt` from ckpt_dir.
    Returns None if no checkpoint exists (fresh run).
    """
    path = os.path.join(ckpt_dir, "checkpoint_last.pt")
    if not os.path.exists(path):
        return None
    print(f"[Resume] Loading checkpoint from:\n         {path}")
    return torch.load(path, map_location="cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV training log
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingLogger:
    """
    Appends one row per epoch to a CSV file.
    The file is created with a header on the first write.
    On resume, rows are appended — no data is lost or overwritten.
    """

    def __init__(self, ckpt_dir: str, filename: str = "train_log.csv"):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.path     = os.path.join(ckpt_dir, filename)
        self._written = os.path.exists(self.path)

    def log(self, row: dict) -> None:
        """Append a single epoch row dict to the CSV."""
        row["logged_at"] = datetime.now().isoformat(timespec="seconds")
        write_header = not self._written
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._written = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Signal handling
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level flag checked at the end of each epoch
_interrupt_requested: bool = False


def setup_signal_handler() -> None:
    """
    Register SIGINT (Ctrl+C) and SIGTERM handlers.
    Sets _interrupt_requested = True so the training loop can save
    a checkpoint and exit cleanly after the current epoch finishes.
    """
    def _handler(sig, frame):
        global _interrupt_requested
        sig_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
        print(f"\n[Signal] {sig_name} received – "
              "finishing current epoch then saving checkpoint …")
        _interrupt_requested = True

    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)


def interrupt_requested() -> bool:
    return _interrupt_requested