#!/usr/bin/env python3
# ============================================================
#  main.py  –  Entry point for the Demographics Pipeline
#
#  Usage examples:
#    python main.py                                  # webcam, default ROI
#    python main.py --source video.mp4
#    python main.py --source rtsp://IP/stream
#
#    # Polygon ROI via CLI (flat x,y pairs — any number of vertices ≥ 3):
#    python main.py --roi 0.0,0.3,1.0,0.3,1.0,0.8,0.0,0.8   # rectangle
#    python main.py --roi 0.3,0.2,0.7,0.2,0.5,0.9            # triangle
#    python main.py --roi 0.1,0.1,0.9,0.1,0.9,0.5,0.5,0.9,0.1,0.5  # pentagon
#
#    # Interactive polygon drawing (click vertices on first frame):
#    python main.py --interactive-roi
#
#    # ONNX Runtime inference (run models/export_onnx.py first):
#    python main.py --use-onnx
# ============================================================

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import config
from pipeline import DemographicsPipeline


# ── Interactive polygon ROI selector ─────────────────────────────────────────

def select_polygon_interactively(
        source: int | str,
) -> list[tuple[float, float]]:
    """
    Open the video source, grab the first usable frame, and let the user
    define an N-sided polygon by clicking vertices.

    Controls
    --------
    Left-click          : add a vertex (shown as circle + connecting line)
    Right-click / 'u'   : undo last vertex
    'c' or Enter        : confirm polygon (needs ≥ 3 vertices)
    'r'                 : reset all vertices
    Esc                 : cancel — returns current config.ROI unchanged

    Returns
    -------
    List of (x_norm, y_norm) tuples, or config.ROI if cancelled / error.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ROI] Cannot open source for interactive selection. "
              "Using config.ROI.")
        return config.ROI

    for _ in range(5):      # let camera auto-exposure settle
        cap.read()
    ret, base_frame = cap.read()
    cap.release()

    if not ret:
        print("[ROI] Could not read frame. Using config.ROI.")
        return config.ROI

    h, w = base_frame.shape[:2]
    vertices: list[tuple[int, int]] = []   # pixel coords while drawing
    confirmed = False

    WIN = "Draw ROI polygon"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(w, 1280), min(h, 720))

    def _mouse_cb(event, mx, my, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            vertices.append((mx, my))
        elif event == cv2.EVENT_RBUTTONDOWN and vertices:
            vertices.pop()

    cv2.setMouseCallback(WIN, _mouse_cb)

    print("\n[ROI] Click to add polygon vertices.")
    print("      Right-click or 'u' → undo last vertex")
    print("      'c' or Enter       → confirm (needs ≥ 3 vertices)")
    print("      'r'                → reset")
    print("      Esc                → cancel\n")

    while True:
        frame = base_frame.copy()

        # Draw existing vertices and edges
        for i, pt in enumerate(vertices):
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(frame, vertices[i - 1], pt, (0, 255, 0), 2)

        # Closing edge preview (when ≥ 3 vertices)
        if len(vertices) >= 3:
            cv2.line(frame, vertices[-1], vertices[0], (0, 200, 0), 1,
                     cv2.LINE_AA)
            # Fill polygon semi-transparently
            overlay = frame.copy()
            pts_arr = np.array(vertices, dtype=np.int32)
            cv2.fillPoly(overlay, [pts_arr], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Status text
        status = (f"Vertices: {len(vertices)}"
                  + ("  |  Press 'c'/Enter to confirm" if len(vertices) >= 3
                     else "  |  Need ≥ 3 vertices"))
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord('c')):          # Enter or 'c' → confirm
            if len(vertices) >= 3:
                confirmed = True
                break
            else:
                print("[ROI] Need at least 3 vertices to confirm.")
        elif key == 27:                    # Esc → cancel
            break
        elif key == ord('r'):              # reset
            vertices.clear()
        elif key == ord('u') and vertices: # undo
            vertices.pop()

    cv2.destroyWindow(WIN)

    if not confirmed or len(vertices) < 3:
        print("[ROI] Selection cancelled. Using config.ROI.")
        return config.ROI

    # Convert pixel coords → normalised
    polygon = [(vx / w, vy / h) for vx, vy in vertices]
    print(f"[ROI] Polygon confirmed: {len(polygon)} vertices")
    for i, (px, py) in enumerate(polygon):
        print(f"      [{i}] ({px:.3f}, {py:.3f})")
    print()
    return polygon


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Demographics Pipeline — Age + Gender from video")

    parser.add_argument("--source", type=str, default=None,
                        help="0 (webcam) | path/to/video.mp4 | rtsp://…")
    parser.add_argument("--fps",    type=int, default=None,
                        help="Target capture FPS (live cameras only)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode")

    # ROI
    parser.add_argument(
        "--roi", type=str, default=None,
        help="Polygon ROI as flat normalised x,y pairs (≥ 3 vertices). "
             "Examples:\n"
             "  Rectangle : 0.0,0.3,1.0,0.3,1.0,0.8,0.0,0.8\n"
             "  Triangle  : 0.3,0.2,0.7,0.2,0.5,0.9")
    parser.add_argument(
        "--interactive-roi", action="store_true",
        help="Click polygon vertices on the first frame interactively")

    # ONNX
    parser.add_argument("--use-onnx", action="store_true",
                        help="Use ONNX Runtime inference "
                             "(run models/export_onnx.py first)")

    # Misc
    parser.add_argument("--person-conf", type=float, default=None,
                        help=f"Person detection threshold "
                             f"(default: {config.PERSON_CONF_THRESH})")
    parser.add_argument("--log", type=str, default=None,
                        help=f"JSONL log path (default: {config.LOG_PATH})")

    return parser.parse_args()


def _parse_roi_polygon(raw: str) -> list[tuple[float, float]] | None:
    """
    Parse '--roi x1,y1,x2,y2,...' into a list of (x, y) tuples.
    Returns None and prints an error if the format is invalid.
    """
    try:
        vals = [float(v) for v in raw.split(",")]
    except ValueError:
        print("[main] ✖  --roi values must be floats. Using config.ROI.")
        return None

    if len(vals) < 6 or len(vals) % 2 != 0:
        print("[main] ✖  --roi needs an even number of values and at least 3 pairs "
              "(6 values). Using config.ROI.")
        return None

    polygon = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]

    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in polygon):
        print("[main] ⚠  Some --roi coordinates are outside [0, 1]. "
              "They will be clamped by the ROI polygon test.")

    return polygon


def apply_cli_overrides(args) -> None:
    if args.source is not None:
        try:
            config.VIDEO_SOURCE = int(args.source)
        except ValueError:
            config.VIDEO_SOURCE = args.source

    if args.fps is not None:
        config.FPS_TARGET = args.fps

    if args.no_display:
        config.DISPLAY_WINDOW = False

    if args.roi is not None:
        polygon = _parse_roi_polygon(args.roi)
        if polygon is not None:
            config.ROI = polygon

    if args.use_onnx:
        config.USE_ONNX = True

    if args.person_conf is not None:
        config.PERSON_CONF_THRESH = args.person_conf

    if args.log is not None:
        config.LOG_PATH = args.log


def print_startup_banner() -> None:
    print("=" * 60)
    print("  Demographic Extraction Pipeline")
    print("  Age Estimation + Gender Identification")
    print("=" * 60)
    print(f"  Source       : {config.VIDEO_SOURCE}")
    print(f"  ROI          : {len(config.ROI)}-sided polygon")
    for i, (x, y) in enumerate(config.ROI):
        print(f"               [{i}] ({x:.3f}, {y:.3f})")
    print(f"  Buffer size  : {config.BUFFER_SIZE} frames")
    print(f"  Inference    : {'ONNX Runtime' if config.USE_ONNX else 'PyTorch'}")
    print(f"  Log path     : {config.LOG_PATH}")
    print(f"  Display      : {'ON' if config.DISPLAY_WINDOW else 'OFF (headless)'}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    apply_cli_overrides(args)

    # Interactive selection runs after source override is applied
    if args.interactive_roi:
        config.ROI = select_polygon_interactively(config.VIDEO_SOURCE)

    print_startup_banner()
    pipeline = DemographicsPipeline()
    pipeline.run(config.VIDEO_SOURCE)


if __name__ == "__main__":
    main()