# ============================================================
#  utils/buffer.py  –  Per-track frame / crop buffer management
# ============================================================

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
from config import BUFFER_SIZE


class TrackBuffer:
    """
    Accumulates full-body BGR crops for triggered Track IDs.

    When a buffer reaches BUFFER_SIZE frames it is marked as 'ready'
    for inference.  The buffer is cleared after results are committed.
    """

    def __init__(self, buffer_size: int = BUFFER_SIZE):
        self.buffer_size = buffer_size
        # {track_id: [bgr_crop, ...]}
        self._buffers: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._ready:   set = set()
        self._done:    set = set()   # IDs already processed → skip

    # ------------------------------------------------------------------
    def is_active(self, track_id: int) -> bool:
        return track_id in self._buffers and track_id not in self._done

    def is_ready(self, track_id: int) -> bool:
        return track_id in self._ready

    def is_done(self, track_id: int) -> bool:
        return track_id in self._done

    # ------------------------------------------------------------------
    def add_frame(self, track_id: int, body_crop: np.ndarray):
        """
        Append a body crop to the buffer.
        Marks the ID as 'ready' once BUFFER_SIZE crops are collected.
        """
        if track_id in self._done:
            return
        buf = self._buffers[track_id]
        if len(buf) < self.buffer_size:
            buf.append(body_crop.copy())
            if len(buf) >= self.buffer_size:
                self._ready.add(track_id)

    def get_crops(self, track_id: int) -> Optional[List[np.ndarray]]:
        """Return the accumulated crops for a track (or None)."""
        return self._buffers.get(track_id, None)

    def commit(self, track_id: int):
        """
        Mark this ID as fully processed and free its memory.
        Call this immediately after JSON serialisation per spec.
        """
        self._buffers.pop(track_id, None)
        self._ready.discard(track_id)
        self._done.add(track_id)

    def remove(self, track_id: int):
        """Evict a track that disappeared before its buffer was full."""
        self._buffers.pop(track_id, None)
        self._ready.discard(track_id)
        # Do NOT add to _done – the ID may re-appear

    def stats(self) -> Dict:
        return {
            "buffering": len(self._buffers),
            "ready":     len(self._ready),
            "done":      len(self._done),
        }