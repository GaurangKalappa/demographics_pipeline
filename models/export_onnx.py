# ============================================================
#  models/export_onnx.py  –  One-time ONNX export for all four models
#
#  Run ONCE after training is complete:
#      python models/export_onnx.py
#
#  Or selectively:
#      python models/export_onnx.py --skip-yolo
#      python models/export_onnx.py --skip-par --skip-mtl
#
#  Outputs (alongside the original .pt files):
#      weights/yolov8n_person.onnx
#      weights/yolov8n_face.onnx
#      weights/par_model.onnx
#      weights/mtl_model.onnx
#
#  Then set USE_ONNX = True in config.py and run the pipeline normally.
# ============================================================
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pt_to_onnx_path(pt_path: str) -> str:
    """Derive the .onnx path from a .pt path (same dir, same stem)."""
    stem = os.path.splitext(pt_path)[0]
    return stem + ".onnx"


def export_yolo(pt_path: str, out_path: str, label: str) -> None:
    """
    Export a YOLO model to ONNX using ultralytics' built-in exporter.
    The resulting .onnx can be passed directly to YOLO() in pipeline.py.
    """
    if not os.path.exists(pt_path):
        print(f"[Export] ⚠  {label} weights not found: {pt_path} — skipping.")
        return

    from ultralytics import YOLO
    print(f"[Export] Exporting {label}  {pt_path} → {out_path}")
    model = YOLO(pt_path)
    # ultralytics writes the file next to the .pt by default; we rename after
    model.export(format="onnx", dynamic=False, simplify=True)

    # ultralytics saves as <stem>.onnx in the same directory
    default_out = os.path.splitext(pt_path)[0] + ".onnx"
    if default_out != out_path and os.path.exists(default_out):
        os.rename(default_out, out_path)

    print(f"[Export] ✔  {label} → {out_path}")


def export_torch_model(model: torch.nn.Module,
                       dummy_input: torch.Tensor,
                       out_path: str,
                       input_names: list[str],
                       output_names: list[str],
                       label: str) -> None:
    """Export a PyTorch nn.Module to ONNX with explicit I/O names."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    model.eval()
    print(f"[Export] Exporting {label} → {out_path}")
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,   # fold constants → smaller, faster graph
    )
    print(f"[Export] ✔  {label} → {out_path}")


# ── Per-model exports ─────────────────────────────────────────────────────────

def export_par(args) -> None:
    pt_path  = config.PAR_WEIGHTS
    out_path = config.PAR_WEIGHTS_ONNX

    if not os.path.exists(pt_path):
        print(f"[Export] ⚠  PAR weights not found: {pt_path} — skipping.")
        return

    from models.par_model import PARModel
    model = PARModel(weights_path=pt_path, device="cpu")
    dummy = torch.zeros(1, 3, config.PAR_INPUT_SIZE[1], config.PAR_INPUT_SIZE[0])
    # Output names must match the order ONNXRunner reads them (index 0,1,2)
    export_torch_model(
        model, dummy, out_path,
        input_names  = ["input"],
        output_names = ["gender", "age", "orient"],
        label        = "PAR model",
    )


def export_mtl(args) -> None:
    pt_path  = config.MTL_WEIGHTS
    out_path = config.MTL_WEIGHTS_ONNX

    if not os.path.exists(pt_path):
        print(f"[Export] ⚠  MTL weights not found: {pt_path} — skipping.")
        return

    from models.mtl_model import MTLModel
    model = MTLModel(weights_path=pt_path, device="cpu")
    dummy = torch.zeros(1, 3, config.MTL_INPUT_SIZE[1], config.MTL_INPUT_SIZE[0])
    # Output names must match the order ONNXRunner reads them (index 0,1)
    export_torch_model(
        model, dummy, out_path,
        input_names  = ["input"],
        output_names = ["age", "gender"],
        label        = "MTL model",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Export all pipeline models to ONNX format")
    p.add_argument("--skip-yolo", action="store_true",
                   help="Skip YOLO person + face detector export")
    p.add_argument("--skip-par",  action="store_true",
                   help="Skip PAR body model export")
    p.add_argument("--skip-mtl",  action="store_true",
                   help="Skip MTL face model export")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 54)
    print("  ONNX Export — Demographics Pipeline")
    print("=" * 54 + "\n")

    if not args.skip_yolo:
        export_yolo(config.YOLO_PERSON_WEIGHTS,
                    config.YOLO_PERSON_WEIGHTS_ONNX,
                    "Person detector")
        if config.YOLO_FACE_WEIGHTS:
            export_yolo(config.YOLO_FACE_WEIGHTS,
                        config.YOLO_FACE_WEIGHTS_ONNX,
                        "Face detector")

    if not args.skip_par:
        export_par(args)

    if not args.skip_mtl:
        export_mtl(args)

    print("\n[Export] All done.  Set USE_ONNX = True in config.py to use them.")


if __name__ == "__main__":
    main()