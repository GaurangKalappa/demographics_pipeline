# ============================================================
#  utils/visualizer.py  –  Live inference window drawing helpers
# ============================================================

from typing import Dict, Optional, Tuple
import cv2
import numpy as np

# Colour palette (BGR)
COL_TRACK   = (0,  200, 255)   # person bbox
COL_FACE    = (0,  255, 120)   # face bbox
COL_LABEL   = (255, 255, 255)
COL_MALE    = (255, 160,  60)
COL_FEMALE  = (255, 100, 200)
COL_BUFF    = (200, 200,   0)  # buffering indicator
COL_DONE    = (60,  220,  60)  # committed track

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
THICKNESS   = 2


def draw_person(frame: np.ndarray, bbox: Tuple[int,int,int,int],
                track_id: int, state: str = "tracking") -> np.ndarray:
    """Draw person bounding box + track ID."""
    x1, y1, x2, y2 = bbox
    col = {"tracking": COL_TRACK, "buffering": COL_BUFF,
           "done": COL_DONE}.get(state, COL_TRACK)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, THICKNESS)
    label = f"ID:{track_id}"
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                FONT, FONT_SCALE, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def draw_face(frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    """Draw face bounding box."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_FACE, 1)
    return frame


def draw_result(frame: np.ndarray, bbox: Tuple[int,int,int,int],
                gender: str, age_range: str, source: str) -> np.ndarray:
    """Overlay the final demographic prediction below the person bbox."""
    x1, y1, x2, y2 = bbox
    col = COL_FEMALE if gender == "Female" else COL_MALE
    text = f"{gender} | {age_range} [{source[0].upper()}]"
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 6, y2 + th + 8), col, -1)
    cv2.putText(frame, text, (x1 + 3, y2 + th + 3),
                FONT, FONT_SCALE, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, fps: float,
             buffer_stats: Dict, total_committed: int) -> np.ndarray:
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