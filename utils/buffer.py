# ============================================================
#  utils/buffer.py  –  Per-track frame / crop buffer management
#
#  Design note — why _done was removed:
#
#  The old design had commit() call _done.add(track_id) to permanently
#  block re-inference.  This caused a race condition: the background
#  inference thread could call commit() (adding to _done) AFTER the
#  main thread had already called reset() (discarding from _done) on
#  ROI exit.  The result was a permanently blocked track_id that could
#  never be inferred again on re-entry.
#
#  The fix: _done is removed entirely.  commit() only frees crop memory.
#  Whether inference has completed is tracked by pipeline._results
#  (cleared on ROI exit).  The _in_progress set in the pipeline prevents
#  concurrent inference for the same track — no _done needed.
# ============================================================
from __future__ import annotations

import numpy as np
from collections import defaultdict

import config   # module import so BUFFER_SIZE is read at construction time


class TrackBuffer:
    """
    Accumulates full-body BGR crops for Track IDs that have entered the ROI.

    Lifecycle per visit:
        add_frame() × BUFFER_SIZE  →  is_ready() = True  →  pipeline fires inference
        commit()                   →  crops freed, track re-enterable (no _done block)

    On ROI exit:
        reset()  →  crops + ready flag cleared; re-entry starts a fresh buffer
    """

    def __init__(self, buffer_size: int | None = None):
        self.buffer_size = buffer_size if buffer_size is not None else config.BUFFER_SIZE
        self._buffers: dict[int, list[np.ndarray]] = defaultdict(list)
        self._ready:   set[int]                    = set()
        # NOTE: no _done set — see design note above

    # ── State queries ─────────────────────────────────────────────────────────

    def is_active(self, track_id: int) -> bool:
        """True while crops are being accumulated (buffer not yet committed)."""
        return track_id in self._buffers

    def is_ready(self, track_id: int) -> bool:
        """True once BUFFER_SIZE crops have been collected."""
        return track_id in self._ready

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_frame(self, track_id: int, body_crop: np.ndarray) -> None:
        """Append a body crop. Marks the ID as ready once BUFFER_SIZE crops collected."""
        buf = self._buffers[track_id]
        if len(buf) < self.buffer_size:
            buf.append(body_crop.copy())
            if len(buf) >= self.buffer_size:
                self._ready.add(track_id)

    def get_crops(self, track_id: int) -> list[np.ndarray] | None:
        """Return the accumulated crops for a track (or None if not buffering)."""
        return self._buffers.get(track_id, None)

    def commit(self, track_id: int) -> None:
        """
        Free crop memory after inference completes.

        Does NOT permanently block re-inference (no _done.add).
        Re-entry is naturally possible because pipeline._results is
        cleared on ROI exit — the guard `track_id not in self._results`
        allows inference to fire again on re-entry.
        """
        self._buffers.pop(track_id, None)
        self._ready.discard(track_id)

    def reset(self, track_id: int) -> None:
        """
        Clear buffer state on ROI exit so re-entry starts a fresh buffer.
        No _done to clear — race condition with commit() is eliminated.
        """
        self._buffers.pop(track_id, None)
        self._ready.discard(track_id)

    def remove(self, track_id: int) -> None:
        """
        Evict a track lost by ByteTrack before its buffer filled.
        Same as reset() but semantically distinct: used for tracker dropout,
        not ROI exit.
        """
        self._buffers.pop(track_id, None)
        self._ready.discard(track_id)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "buffering": len(self._buffers),
            "ready":     len(self._ready),
        }