# ============================================================
#  pipeline.py  –  Demographics Pipeline Orchestrator
#
#  Stages:
#    1. Frame ingestion & person detection  (YOLOv8n + ByteTrack)
#    2. Tripwire triggering & buffer management
#    3. Hierarchical inference (Body PAR + Face MTL in parallel threads)
#    4. Decision ensemble & age-range mapping
#    5. JSON serialisation & live display
#

# ============================================================
from __future__ import annotations

import sys
import time
import threading

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
from models.par_model  import PARModel
from models.mtl_model  import MTLModel
from utils.preprocessing import (
    safe_crop, prepare_body_crop, prepare_face_crop,
)
from utils.tripwire    import TripwireManager
from utils.buffer      import TrackBuffer
from utils.decision    import resolve, DecisionResult
from utils.serializer  import build_record, write_record, print_record
from utils.visualizer  import draw_person, draw_face, draw_result, draw_hud


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference thread payload
# ═══════════════════════════════════════════════════════════════════════════════

class _InferenceJob:
    """Carries all data needed by the dual-stream inference thread."""
    __slots__ = ("track_id", "body_crops", "person_bbox",
                 "par_result", "mtl_result", "face_conf", "done")

    def __init__(self, track_id: int, body_crops: list[np.ndarray],
                 person_bbox: tuple[int, int, int, int]):
        self.track_id    = track_id
        self.body_crops  = body_crops
        self.person_bbox = person_bbox
        self.par_result: dict | None  = None
        self.mtl_result: dict | None  = None
        self.face_conf:  float | None = None
        self.done        = threading.Event()


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline class
# ═══════════════════════════════════════════════════════════════════════════════

class DemographicsPipeline:

    def __init__(self) -> None:
        print("\n[Pipeline] Initialising …")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Pipeline] Device : {device}")

        # ── Models ──────────────────────────────────────────────────────
        self.person_detector = YOLO(config.YOLO_PERSON_WEIGHTS)

        # Face detector is optional – if weights file is missing we skip
        # the face stream and fall back to body-only inference.
        self.face_detector: YOLO | None = None
        if config.YOLO_FACE_WEIGHTS:
            try:
                self.face_detector = YOLO(config.YOLO_FACE_WEIGHTS)
            except Exception as exc:
                print(f"[Pipeline] ⚠  Could not load face detector: {exc}")
                print("[Pipeline]    Face stream disabled – body-only mode.")

        self.par_model = PARModel(
            weights_path=config.PAR_WEIGHTS or None,
            device=device,
        )
        self.mtl_model = MTLModel(
            weights_path=config.MTL_WEIGHTS or None,
            device=device,
        )

        # ── State managers ───────────────────────────────────────────────
        self.tripwire = TripwireManager()
        self.buffer   = TrackBuffer()

        # track_id → final result / JSON record
        self._results: dict[int, DecisionResult] = {}
        self._records: dict[int, dict]           = {}
        self._committed_count = 0

        # FPS tracking
        self._fps      = 0.0
        self._frame_ts = time.time()

        print("[Pipeline] ✔  Ready.\n")

    # ────────────────────────────────────────────────────────────────────────────
    #  Public entry point
    # ────────────────────────────────────────────────────────────────────────────

    def run(self, source: int | str) -> None:
        """
        Main loop.

        Parameters
        ----------
        source : int | str
            0 for webcam, path string for video file, "rtsp://…" for RTSP stream.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[Pipeline] ✖  Cannot open source: {source}")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FPS, config.FPS_TARGET)

        print(f"[Pipeline] ▶  Streaming from : {source}")
        print("[Pipeline] Press 'q' to quit.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Pipeline] Stream ended or frame read failed.")
                    break

                self._update_fps()
                processed = self._process_frame(frame)

                if config.DISPLAY_WINDOW:
                    cv2.imshow(config.DISPLAY_WIN_NAME, processed)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[Pipeline] User requested quit.")
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[Pipeline] ■  Stopped.")

    # ────────────────────────────────────────────────────────────────────────────
    #  Stage 1 + 2: Detection, tracking, tripwire, buffering
    # ────────────────────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w    = frame.shape[:2]
        display = frame.copy()

        # Stage 1: YOLOv8n person detection + ByteTrack
        results = self.person_detector.track(
            frame,
            persist=True,
            conf=config.PERSON_CONF_THRESH,
            classes=[0],                  # COCO class 0 = person
            tracker="bytetrack.yaml",     # bundled in ultralytics >= 8.x
            verbose=False,
        )

        active_ids: set[int] = set()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                active_ids.add(track_id)

                # Stage 2: Tripwire check
                crossed = self.tripwire.update(
                    track_id, (x1, y1, x2, y2), w, h)

                if crossed:
                    print(f"[Tripwire] ▶  ID {track_id} crossed – buffering.")
                    body_crop = safe_crop(frame, x1, y1, x2, y2)
                    if body_crop is None:
                        body_crop = np.zeros((64, 32, 3), dtype=np.uint8)
                    self.buffer.add_frame(track_id, body_crop)

                elif (self.buffer.is_active(track_id)
                      and not self.buffer.is_done(track_id)):
                    body_crop = safe_crop(frame, x1, y1, x2, y2)
                    if body_crop is not None:
                        self.buffer.add_frame(track_id, body_crop)

                # Stage 3: Fire inference when buffer is full
                if (self.buffer.is_ready(track_id)
                        and track_id not in self._results):
                    crops = self.buffer.get_crops(track_id)
                    if crops:
                        self._run_inference(track_id, crops, (x1, y1, x2, y2))

                # Draw annotations
                state = (
                    "done"      if self.buffer.is_done(track_id)   else
                    "buffering" if self.buffer.is_active(track_id) else
                    "tracking"
                )
                draw_person(display, (x1, y1, x2, y2), track_id, state)

                if track_id in self._results:
                    res = self._results[track_id]
                    draw_result(display, (x1, y1, x2, y2),
                                res.gender, res.age_range, res.source)

        # Clean up tripwire state for tracks that left the scene
        dead_ids = set(self._results.keys()) - active_ids
        for tid in dead_ids:
            self.tripwire.remove(tid)

        self.tripwire.draw(display)
        draw_hud(display, self._fps, self.buffer.stats(), self._committed_count)
        return display

    # ────────────────────────────────────────────────────────────────────────────
    #  Stage 3: Dual-stream inference dispatcher
    # ────────────────────────────────────────────────────────────────────────────

    def _run_inference(self, track_id: int,
                       crops: list[np.ndarray],
                       person_bbox: tuple[int, int, int, int]) -> None:
        """Spawn a daemon thread to run both inference streams concurrently."""
        job = _InferenceJob(track_id, crops, person_bbox)
        t   = threading.Thread(
            target=self._inference_worker, args=(job,), daemon=True)
        t.start()

    def _inference_worker(self, job: _InferenceJob) -> None:
        """
        Dual-stream worker.
          Branch A – Body PAR on the first buffered crop
          Branch B – Face detection + MTL across all buffered crops
        Both branches run in child threads; we join before the decision step.
        """
        par_t = threading.Thread(
            target=self._body_stream, args=(job, job.body_crops[0]), daemon=True)
        mtl_t = threading.Thread(
            target=self._face_stream, args=(job, job.body_crops), daemon=True)

        par_t.start()
        mtl_t.start()
        par_t.join()
        mtl_t.join()

        # Stage 4: Decision ensemble
        par = job.par_result or {
            "gender_body_score": 0.5,
            "age_coarse":        "Adult",
            "orientation":       "Front",
            "confidence":        0.5,
        }

        result = resolve(
            c_face            = job.face_conf,
            gender_face_score = job.mtl_result.get("gender_face_score") if job.mtl_result else None,
            age_raw_face      = job.mtl_result.get("age_raw")           if job.mtl_result else None,
            c_body            = par["confidence"],
            gender_body_score = par["gender_body_score"],
            age_coarse        = par["age_coarse"],
        )

        # Stage 5: Serialisation
        record = build_record(job.track_id, result)
        write_record(record)
        print_record(record, result)

        self._results[job.track_id] = result
        self._records[job.track_id] = record
        self._committed_count      += 1

        # Free buffer immediately after commit (spec: prevent memory leaks)
        self.buffer.commit(job.track_id)

    # ── Body stream ──────────────────────────────────────────────────────────

    def _body_stream(self, job: _InferenceJob, body_crop: np.ndarray) -> None:
        """Run PAR inference on the first body crop."""
        try:
            prepared       = prepare_body_crop(body_crop)
            job.par_result = self.par_model.predict(prepared)
            print(f"[PAR  ] ID {job.track_id} → "
                  f"gender={job.par_result['gender_body_score']:.2f}  "
                  f"age={job.par_result['age_coarse']}  "
                  f"C_body={job.par_result['confidence']:.2f}")
        except Exception as exc:
            print(f"[PAR  ] ✖  Error for ID {job.track_id}: {exc}")
            job.par_result = None

    # ── Face stream ──────────────────────────────────────────────────────────

    def _face_stream(self, job: _InferenceJob,
                     crops: list[np.ndarray]) -> None:
        """
        1. Scan all buffered crops for faces (YOLOv8n-face).
        2. Keep the single highest-confidence face crop.
        3. Run MTL on that crop.
        """
        best_face_crop:  np.ndarray | None = None
        best_face_conf = 0.0

        for body_crop in crops:
            face_crop, face_conf = self._detect_best_face(body_crop)
            if face_crop is not None and face_conf > best_face_conf:
                best_face_crop = face_crop
                best_face_conf = face_conf

        if best_face_crop is None or best_face_conf < config.FACE_CONF_THRESH:
            print(f"[Face ] ID {job.track_id} → no usable face (best_conf="
                  f"{best_face_conf:.2f} < {config.FACE_CONF_THRESH}).")
            job.face_conf  = None
            job.mtl_result = None
            return

        job.face_conf = best_face_conf
        try:
            prepared       = prepare_face_crop(best_face_crop)
            job.mtl_result = self.mtl_model.predict(prepared)
            print(f"[MTL  ] ID {job.track_id} → "
                  f"gender={job.mtl_result['gender_face_score']:.2f}  "
                  f"age={job.mtl_result['age_raw']:.1f}  "
                  f"C_face={best_face_conf:.2f}")
        except Exception as exc:
            print(f"[MTL  ] ✖  Error for ID {job.track_id}: {exc}")
            job.mtl_result = None

    def _detect_best_face(
            self, body_crop: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """
        Run the face detector on a body crop.
        Returns (face_bgr_crop, confidence) of the best detection,
        or (None, 0.0) if no face found or detector unavailable.
        """
        if self.face_detector is None:
            return None, 0.0

        try:
            res = self.face_detector(
                body_crop, verbose=False, conf=config.FACE_CONF_THRESH)
            if not res or res[0].boxes is None or len(res[0].boxes) == 0:
                return None, 0.0

            boxes  = res[0].boxes
            confs  = boxes.conf.tolist()
            best_i = int(np.argmax(confs))
            best_c = float(confs[best_i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[best_i].tolist())
            face_crop = safe_crop(body_crop, x1, y1, x2, y2)
            return face_crop, best_c
        except Exception as exc:
            print(f"[Face ] ✖  Detector error: {exc}")
            return None, 0.0

    # ── FPS helper ───────────────────────────────────────────────────────────

    def _update_fps(self) -> None:
        now            = time.time()
        self._fps      = 1.0 / max(now - self._frame_ts, 1e-6)
        self._frame_ts = now