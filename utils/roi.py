# ============================================================
#  utils/roi.py  –  Polygon ROI with entry/exit state tracking
#
#  State machine per track ID:
#
#    outside → entered  :  NOT prev AND curr  → pipeline buffers + infers once
#    entered → inside   :  prev AND curr      → pipeline keeps buffering
#    inside  → exited   :  prev AND NOT curr  → pipeline resets buffer + result
#    outside → outside  :  NOT prev AND NOT curr → nothing
#
#  Edge cases handled:
#    • New track          : prev defaults to False → entry detected correctly
#    • Tracker dropout    : remove() clears prev → reappearing track starts fresh
#    • Re-entry after exit: pipeline clears _results on exit → guard allows re-infer
#    • Degenerate polygon : < 3 vertices → falls back to full-frame (always inside)
#    • Concave polygons   : cv2.pointPolygonTest handles any simple polygon
# ============================================================
from __future__ import annotations

import cv2
import numpy as np

import config   # module import — values read at construction time, not import time


class ROIManager:
    """
    Manages per-track inside/outside state relative to an N-sided polygon ROI.

    Parameters
    ----------
    polygon : list of (x_norm, y_norm) tuples defining polygon vertices, or None.
              None → reads config.ROI at construction time.
              Must have at least 3 vertices.  Vertices are in normalised [0,1] coords.

    Returns from update()
    ---------------------
    "entered"  – first frame the person's bbox centre is inside the polygon
    "inside"   – subsequent frames still inside
    "exited"   – first frame outside after being inside
    "outside"  – outside and was already outside
    """

    def __init__(self, polygon: list[tuple[float, float]] | None = None):
        raw = polygon if polygon is not None else config.ROI
        # Validate: need at least 3 vertices for a polygon
        if len(raw) < 3:
            print(f"[ROI] ⚠  ROI has only {len(raw)} vertices (need ≥ 3). "
                  "Falling back to full-frame ROI.")
            raw = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.polygon: list[tuple[float, float]] = raw
        self._prev_inside: dict[int, bool] = {}

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, track_id: int,
               bbox: tuple[int, int, int, int],
               frame_w: int, frame_h: int) -> str:
        """
        Call once per detected track per frame.

        Returns one of: "entered" | "inside" | "exited" | "outside"
        """
        curr = self._centre_inside(bbox, frame_w, frame_h)
        prev = self._prev_inside.get(track_id, False)
        self._prev_inside[track_id] = curr

        if not prev and curr:   return "entered"
        if prev     and curr:   return "inside"
        if prev     and not curr: return "exited"
        return "outside"

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def remove(self, track_id: int) -> None:
        """
        Call when a track disappears from ByteTrack entirely.
        Clears prev state so a reappearing track starts from 'outside'.
        """
        self._prev_inside.pop(track_id, None)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the polygon ROI on the frame (in-place)."""
        h, w = frame.shape[:2]
        pts  = self._pixel_polygon(w, h)

        cv2.polylines(frame, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)

        # Label at the first vertex
        lx, ly = int(pts[0][0][0]) + 4, int(pts[0][0][1]) + 20
        cv2.putText(frame, f"ROI ({len(self.polygon)}-sided)",
                    (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    # ── Private helpers ───────────────────────────────────────────────────────

    def _centre_inside(self, bbox: tuple[int, int, int, int],
                       frame_w: int, frame_h: int) -> bool:
        """
        Return True if the bounding-box centre is inside the polygon.
        Uses cv2.pointPolygonTest — works for any simple (non-self-intersecting)
        polygon, both convex and concave.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        pts = self._pixel_polygon(frame_w, frame_h)
        # >= 0 means inside or on boundary; < 0 means outside
        return cv2.pointPolygonTest(pts, (cx, cy), measureDist=False) >= 0

    def _pixel_polygon(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Convert normalised polygon vertices to pixel coords for OpenCV."""
        pts = np.array(
            [[int(x * frame_w), int(y * frame_h)] for x, y in self.polygon],
            dtype=np.int32,
        )
        return pts.reshape((-1, 1, 2))   # shape required by cv2.polylines / pointPolygonTest