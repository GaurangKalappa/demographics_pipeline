# ============================================================
#  utils/visualizer.py  –  Live inference window drawing helpers
# ============================================================
from __future__ import annotations

import cv2
import numpy as np

# Colour palette (BGR)
COL_TRACK   = (0,  200, 255)   # person bbox  – cyan
COL_FACE    = (0,  255, 120)   # face bbox    – green
COL_LABEL   = (255, 255, 255)
COL_MALE    = (255, 160,  60)  # orange
COL_FEMALE  = (255, 100, 200)  # pink
COL_BUFF    = (200, 200,   0)  # buffering    – yellow
COL_DONE    = (60,  220,  60)  # committed    – green

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
THICKNESS   = 2


def draw_person(frame: np.ndarray, bbox: tuple[int,int,int,int],
                track_id: int, state: str = "tracking") -> np.ndarray:
    """Draw person bounding box + track ID label above the box."""
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = frame.shape[:2]

    col = {"tracking": COL_TRACK, "buffering": COL_BUFF,
           "done": COL_DONE}.get(state, COL_TRACK)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, THICKNESS)

    label = f"ID:{track_id}"
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)

    # Clamp label x so it never runs off the right edge
    lx = min(x1, w_frame - tw - 6)
    lx = max(0, lx)

    # Prefer drawing above the box; if no room, draw inside the top edge
    if y1 - th - 6 >= 0:
        by1, by2 = y1 - th - 6, y1
    else:
        by1, by2 = y1, y1 + th + 6

    cv2.rectangle(frame, (lx, by1), (lx + tw + 4, by2), col, -1)
    cv2.putText(frame, label, (lx + 2, by2 - 4),
                FONT, FONT_SCALE, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def draw_face(frame: np.ndarray, bbox: tuple[int,int,int,int]) -> np.ndarray:
    """Draw face bounding box."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_FACE, 1)
    return frame


def draw_result(frame: np.ndarray, bbox: tuple[int,int,int,int],
                gender: str, age_range: str, source: str) -> np.ndarray:
    """
    Overlay the final demographic prediction on the bounding box.

    Issue 3 fix: label position is clamped so it always stays within
    frame bounds regardless of where the person is in the frame.

    Placement priority:
      1. Below the bbox  (preferred — doesn't obscure the person)
      2. Inside the bbox bottom  (fallback when bbox is near bottom edge)
    X position is also clamped so the label never runs off the right edge.
    """
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = frame.shape[:2]

    col  = COL_FEMALE if gender == "Female" else COL_MALE
    text = f"{gender} | {age_range} [{source[0].upper()}]"
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)

    pad = 4   # pixels of padding around the text

    # ── X: clamp so label never runs off the right edge ───────────────────
    lx = min(x1, w_frame - tw - pad * 2 - 2)
    lx = max(0, lx)

    # ── Y: prefer below bbox; fall back to inside bbox bottom ─────────────
    label_h = th + pad * 2
    if y2 + label_h <= h_frame:
        # Enough room below the bbox
        bg_y1 = y2
        bg_y2 = y2 + label_h
    else:
        # Draw inside the bottom of the bbox
        bg_y2 = min(y2, h_frame)
        bg_y1 = max(bg_y2 - label_h, y1)

    cv2.rectangle(frame, (lx, bg_y1), (lx + tw + pad * 2, bg_y2), col, -1)
    cv2.putText(frame, text, (lx + pad, bg_y2 - pad),
                FONT, FONT_SCALE, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, fps: float,
             buffer_stats: dict, total_committed: int) -> np.ndarray:
    """Draw a HUD overlay in the top-left corner."""
    lines = [
        f"FPS        : {fps:.1f}",
        f"Buffering  : {buffer_stats.get('buffering', 0)}",
        f"Committed  : {total_committed}",
    ]
    y = 20
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18
    return frame