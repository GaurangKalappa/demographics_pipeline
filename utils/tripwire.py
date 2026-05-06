# ============================================================
#  utils/tripwire.py  –  Virtual tripwire / line-crossing detection
# ============================================================

from collections import defaultdict
from typing import Dict, Tuple
import numpy as np
from config import TRIPWIRE_START, TRIPWIRE_END


def _ccw(ax, ay, bx, by, cx, cy) -> bool:
    """Counter-clockwise orientation test."""
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    """Return True if line segment AB intersects CD."""
    return (
        _ccw(ax, ay, cx, cy, dx, dy) != _ccw(bx, by, cx, cy, dx, dy)
        and _ccw(ax, ay, bx, by, cx, cy) != _ccw(ax, ay, bx, by, dx, dy)
    )


class TripwireManager:
    """
    Maintains per-track centroid history and detects line crossings.

    The tripwire is defined by two normalised [0,1] coordinates and is
    scaled to actual pixel coords on each call (supports dynamic resolution).
    """

    def __init__(self,
                 start: Tuple[float, float] = TRIPWIRE_START,
                 end:   Tuple[float, float] = TRIPWIRE_END,
                 history_len: int = 4):
        self.start       = start          # (x_norm, y_norm)
        self.end         = end
        self.history_len = history_len
        # {track_id: deque of (cx_px, cy_px)}
        self._history: Dict[int, list] = defaultdict(list)
        self._triggered: set = set()      # IDs that have already crossed

    # ------------------------------------------------------------------
    def update(self, track_id: int, bbox: Tuple[int, int, int, int],
               frame_w: int, frame_h: int) -> bool:
        """
        Call once per track per frame.

        Parameters
        ----------
        track_id : int
        bbox     : (x1, y1, x2, y2) in pixel coords
        frame_w, frame_h : frame dimensions (for normalised tripwire scaling)

        Returns
        -------
        bool – True the first time this track crosses the wire.
        """
        if track_id in self._triggered:
            return False

        x1, y1, x2, y2 = bbox
        # Use bottom-centre of bounding box as per spec
        cx = (x1 + x2) // 2
        cy = y2

        hist = self._history[track_id]
        hist.append((cx, cy))
        if len(hist) > self.history_len:
            hist.pop(0)

        if len(hist) < 2:
            return False

        # Scale normalised wire to pixel coords
        wx1 = int(self.start[0] * frame_w)
        wy1 = int(self.start[1] * frame_h)
        wx2 = int(self.end[0]   * frame_w)
        wy2 = int(self.end[1]   * frame_h)

        prev_cx, prev_cy = hist[-2]
        curr_cx, curr_cy = hist[-1]

        if _segments_intersect(prev_cx, prev_cy, curr_cx, curr_cy,
                                wx1, wy1, wx2, wy2):
            self._triggered.add(track_id)
            return True

        return False

    def remove(self, track_id: int):
        """Clean up when a track disappears from the scene."""
        self._history.pop(track_id, None)
        self._triggered.discard(track_id)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the tripwire on the frame (in-place)."""
        import cv2
        h, w = frame.shape[:2]
        x1 = int(self.start[0] * w)
        y1 = int(self.start[1] * h)
        x2 = int(self.end[0]   * w)
        y2 = int(self.end[1]   * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "TRIPWIRE", (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return frame