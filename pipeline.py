# ============================================================
#  pipeline.py  –  Demographics Pipeline Orchestrator
#
#  Stages:
#    1. Frame ingestion & person detection  (YOLOv8n + ByteTrack)
#    2. ROI state tracking (entry/exit triggers inference once per visit)
#    3. Hierarchical inference (Body PAR + Face MTL in parallel threads)
#       Face detection runs on every 2nd buffered crop (change 4)
#    4. Decision ensemble & age-range mapping
#    5. JSON serialisation & live display
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
from utils.roi         import ROIManager
from utils.buffer      import TrackBuffer
from utils.preprocessing import safe_crop, prepare_body_crop, prepare_face_crop
from utils.decision    import resolve, DecisionResult
from utils.serializer  import build_record, write_record, print_record
from utils.visualizer  import draw_person, draw_result, draw_hud


# ── Model imports: PyTorch or ONNX Runtime depending on config ────────────────
if config.USE_ONNX:
    from models.par_model_onnx import PARModelONNX as _PARModel
    from models.mtl_model_onnx import MTLModelONNX as _MTLModel
else:
    from models.par_model import PARModel  as _PARModel
    from models.mtl_model import MTLModel  as _MTLModel


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference thread payload
# ═══════════════════════════════════════════════════════════════════════════════

class _InferenceJob:
    __slots__ = ("track_id", "body_crops", "person_bbox",
                "par_result", "mtl_result", "face_conf")

    def __init__(self, track_id: int, body_crops: list[np.ndarray],
                person_bbox: tuple[int, int, int, int]):
        self.track_id    = track_id
        self.body_crops  = body_crops
        self.person_bbox = person_bbox
        self.par_result: dict | None  = None
        self.mtl_result: dict | None  = None
        self.face_conf:  float | None = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class DemographicsPipeline:

    def __init__(self) -> None:
        print("\n[Pipeline] Initialising …")
        print(f"[Pipeline] Mode   : {'ONNX Runtime' if config.USE_ONNX else 'PyTorch'}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Pipeline] Device : {device}")

        # ── Models ──────────────────────────────────────────────────────────
        if config.USE_ONNX:
            self.person_detector = YOLO(config.YOLO_PERSON_WEIGHTS_ONNX)
            self.face_detector: YOLO | None = None
            if config.YOLO_FACE_WEIGHTS_ONNX:
                try:
                    self.face_detector = YOLO(config.YOLO_FACE_WEIGHTS_ONNX)
                except Exception as exc:
                    print(f"[Pipeline] ⚠  Face detector (ONNX) failed: {exc}")
            self.par_model = _PARModel(config.PAR_WEIGHTS_ONNX)
            self.mtl_model = _MTLModel(config.MTL_WEIGHTS_ONNX)
        else:
            self.person_detector = YOLO(config.YOLO_PERSON_WEIGHTS)
            self.face_detector = None
            if config.YOLO_FACE_WEIGHTS:
                try:
                    self.face_detector = YOLO(config.YOLO_FACE_WEIGHTS)
                except Exception as exc:
                    print(f"[Pipeline] ⚠  Face detector failed: {exc}")
            self.par_model = _PARModel(
                weights_path=config.PAR_WEIGHTS or None, device=device)
            self.mtl_model = _MTLModel(
                weights_path=config.MTL_WEIGHTS or None, device=device)

        # ── State managers ───────────────────────────────────────────────────
        self.roi    = ROIManager()
        self.buffer = TrackBuffer()

        self._results:         dict[int, DecisionResult] = {}
        self._records:         dict[int, dict]           = {}
        self._committed_count: int                       = 0
        self._in_progress:     set[int]                  = set()

        self._fps      = 0.0
        self._frame_ts = time.time()

        print(f"[Pipeline] ROI    : {config.ROI}")
        print(f"[Pipeline] Buffer : {config.BUFFER_SIZE} frames "
              f"(face detection every 2nd crop)")
        print("[Pipeline] ✔  Ready.\n")

    # ────────────────────────────────────────────────────────────────────────────
    #  Public entry point
    # ────────────────────────────────────────────────────────────────────────────

    def run(self, source: int | str) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[Pipeline] ✖  Cannot open source: {source}")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FPS, config.FPS_TARGET)

        if config.DISPLAY_WINDOW:
            cv2.namedWindow(config.DISPLAY_WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(config.DISPLAY_WIN_NAME,
                             config.DISPLAY_MAX_WIDTH, config.DISPLAY_MAX_HEIGHT)

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
                    ph, pw = processed.shape[:2]
                    scale  = min(config.DISPLAY_MAX_WIDTH / pw,
                                 config.DISPLAY_MAX_HEIGHT / ph, 1.0)
                    if scale < 1.0:
                        display_frame = cv2.resize(
                            processed,
                            (int(pw * scale), int(ph * scale)),
                            interpolation=cv2.INTER_AREA)
                    else:
                        display_frame = processed

                    cv2.imshow(config.DISPLAY_WIN_NAME, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[Pipeline] ■  Stopped.")

    # ────────────────────────────────────────────────────────────────────────────
    #  Frame processing
    # ────────────────────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w    = frame.shape[:2]
        display = frame.copy()

        # ── Downscale for YOLO (bandwidth saving) ───────────────────────────
        inf_scale = 1.0
        if config.INFERENCE_WIDTH and w > config.INFERENCE_WIDTH:
            inf_scale = config.INFERENCE_WIDTH / w
            inf_frame = cv2.resize(frame,
                                   (config.INFERENCE_WIDTH, int(h * inf_scale)),
                                   interpolation=cv2.INTER_AREA)
        else:
            inf_frame = frame

        # ── Stage 1: Person detection + ByteTrack ───────────────────────────
        results = self.person_detector.track(
            inf_frame, persist=True,
            conf=config.PERSON_CONF_THRESH,
            classes=[0], tracker="bytetrack.yaml", verbose=False,
        )

        active_ids: set[int] = set()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(x1 / inf_scale)
                y1 = int(y1 / inf_scale)
                x2 = int(x2 / inf_scale)
                y2 = int(y2 / inf_scale)
                active_ids.add(track_id)

                # ── Stage 2: ROI state machine ───────────────────────────────
                roi_state = self.roi.update(track_id, (x1, y1, x2, y2), w, h)

                if roi_state == "entered":
                    # First frame inside ROI → start buffering
                    print(f"[ROI] ▶  ID {track_id} entered ROI – buffering.")
                    
                    body_crop = safe_crop(frame, x1, y1, x2, y2)

                    if body_crop is None:   
                        body_crop = np.zeros((64, 32, 3), dtype=np.uint8)

                    self.buffer.add_frame(track_id, body_crop)

                elif roi_state == "inside":
                    # Still inside → keep buffering until full
                    if self.buffer.is_active(track_id):
                        body_crop = safe_crop(frame, x1, y1, x2, y2)
                        if body_crop is not None:
                            self.buffer.add_frame(track_id, body_crop)

                elif roi_state == "exited":
                    # Left ROI → reset buffer + clear result so re-entry infers fresh.
                    # _in_progress is also cleared: if inference was still running
                    # when exit happened, commit() will still free crop memory, but
                    # the result will not be stored (track_id no longer in _results
                    # guard — the next entry starts cleanly).
                    print(f"[ROI] ◀  ID {track_id} exited ROI – resetting.")
                    self.buffer.reset(track_id)          # clear crops + ready flag
                    self._results.pop(track_id, None)    # clear last result
                    self._records.pop(track_id, None)    # clear last JSON record
                    self._in_progress.discard(track_id)  # _active_inference.remove()

                # "outside" → do nothing

                # ── Stage 3: Fire inference once buffer is full ──────────────
                if (self.buffer.is_ready(track_id)
                        and track_id not in self._results
                        and track_id not in self._in_progress):
                    crops = self.buffer.get_crops(track_id)
                    if crops:
                        self._in_progress.add(track_id)
                        self._run_inference(track_id, crops, (x1, y1, x2, y2))

                # ── Draw ─────────────────────────────────────────────────────
                # "done" state: inference completed → result exists in _results.
                # is_done() removed from buffer (no _done set); use _results instead.
                state = ("done"      if track_id in self._results else
                         "buffering" if self.buffer.is_active(track_id) else
                         "tracking")
                draw_person(display, (x1, y1, x2, y2), track_id, state)

                if track_id in self._results:
                    res = self._results[track_id]
                    draw_result(display, (x1, y1, x2, y2),
                                res.gender, res.age_range, res.source)

        # ── Clean up tracks the detector dropped entirely ────────────────────
        # (different from ROI exit: track lost by ByteTrack completely)
        known_ids = set(self.roi._prev_inside.keys())
        dead_ids  = known_ids - active_ids
        for tid in dead_ids:
            self.roi.remove(tid)
            # Do not reset buffer/results: the track may re-appear with same ID
            # and we want to keep showing the last result.

        self.roi.draw(display)
        draw_hud(display, self._fps, self.buffer.stats(), self._committed_count)
        return display

    # ────────────────────────────────────────────────────────────────────────────
    #  Stage 3: Dual-stream inference
    # ────────────────────────────────────────────────────────────────────────────

    def _run_inference(self, track_id: int, crops: list[np.ndarray],
                       person_bbox: tuple[int, int, int, int]) -> None:
        job = _InferenceJob(track_id, crops, person_bbox)
        threading.Thread(target=self._inference_worker,
                         args=(job,), daemon=True).start()

    def _inference_worker(self, job: _InferenceJob) -> None:
        try:
            par_t = threading.Thread(
                target=self._body_stream, args=(job, job.body_crops), daemon=True)
            mtl_t = threading.Thread(
                target=self._face_stream, args=(job, job.body_crops), daemon=True)
            par_t.start(); mtl_t.start()
            par_t.join();  mtl_t.join()

            par = job.par_result or {
                "gender_body_score": 0.5, "age_coarse": "Adult",
                "orientation": "Front",   "confidence": 0.5,
            }
            result = resolve(
                c_face            = job.face_conf,
                gender_face_score = job.mtl_result.get("gender_face_score") if job.mtl_result else None,
                age_raw_face      = job.mtl_result.get("age_raw")           if job.mtl_result else None,
                c_body            = par["confidence"],
                gender_body_score = par["gender_body_score"],
                age_coarse        = par["age_coarse"],
            )
            record = build_record(job.track_id, result)
            write_record(record)
            print_record(record, result)

            self._results[job.track_id] = result
            self._records[job.track_id] = record
            self._committed_count      += 1
            self.buffer.commit(job.track_id)

        except Exception as exc:
            print(f"[Worker] ✖  Error for ID {job.track_id}: {exc}")
        finally:
            self._in_progress.discard(job.track_id)

    # ── Body stream ───────────────────────────────────────────────────────────

    def _body_stream(self, job: _InferenceJob, body_crops: list[np.ndarray])-> None:
    
        try:
            gender_scores = []
            age_votes = []
            orient_votes = []
            confidences = []

            # Every 2nd crop
            for crop in body_crops[::2]:

                if crop is None or crop.size == 0:
                    continue

                prepared = prepare_body_crop(crop)

                result = self.par_model.predict(prepared)

                gender_scores.append(result["gender_body_score"])
                age_votes.append(result["age_coarse"])
                orient_votes.append(result["orientation"])
                confidences.append(result["confidence"])

            if len(gender_scores) == 0:
                print(f"[PAR  ] ✖  No valid body crops for ID {job.track_id}")
                job.par_result = None
                return

            avg_gender = float(np.mean(gender_scores))
            avg_conf   = float(np.mean(confidences))

            def majority_vote(items):
                return max(set(items), key=items.count)

            final_age    = majority_vote(age_votes)
            final_orient = majority_vote(orient_votes)

            job.par_result = {
                "gender_body_score": avg_gender,
                "age_coarse": final_age,
                "orientation": final_orient,
                "confidence": avg_conf,
            }

            print(
                f"[PAR  ] ID {job.track_id} → "
                f"gender={avg_gender:.2f}  "
                f"age={final_age}  "
                f"orient={final_orient}  "
                f"C_body={avg_conf:.2f}"
            )

        except Exception as exc:
                print(f"[PAR  ] ✖  Error for ID {job.track_id}: {exc}")
                job.par_result = None

    # ── Face stream — every 2nd crop (change 4) ───────────────────────────────

    def _face_stream(self, job: _InferenceJob,
                     crops: list[np.ndarray]) -> None:
        """
        Scan buffered crops for the best face.

        Change 4: runs face detection on every 2nd crop (crops[::2]).
        With BUFFER_SIZE=6 this means 3 YOLO face calls instead of 6,
        halving face-detection work per person while still scanning
        enough crops to find the best available face.
        """
        best_face_crop: np.ndarray | None = None
        best_face_conf = 0.0

        for body_crop in crops[::2]:    # every 2nd crop — change 4
            face_crop, face_conf = self._detect_best_face(body_crop)
            if face_crop is not None and face_conf > best_face_conf:
                best_face_crop = face_crop
                best_face_conf = face_conf

        if best_face_crop is None or best_face_conf < config.FACE_CONF_THRESH:
            print(f"[Face ] ID {job.track_id} → no usable face "
                  f"(best_conf={best_face_conf:.2f} < {config.FACE_CONF_THRESH}).")
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

    def _detect_best_face(self, body_crop: np.ndarray
                          ) -> tuple[np.ndarray | None, float]:
        if self.face_detector is None:
            return None, 0.0
        try:
            res = self.face_detector(
                body_crop, verbose=False, conf= 0.10) #config.FACE_CONF_THRESH)
            if not res or res[0].boxes is None or len(res[0].boxes) == 0:
                return None, 0.0
            boxes  = res[0].boxes
            confs  = boxes.conf.tolist()
            best_i = int(np.argmax(confs))
            x1, y1, x2, y2 = map(int, boxes.xyxy[best_i].tolist())
            return safe_crop(body_crop, x1, y1, x2, y2), float(confs[best_i])
        except Exception as exc:
            print(f"[Face ] ✖  Detector error: {exc}")
            return None, 0.0

    def _update_fps(self) -> None:
        now            = time.time()
        self._fps      = 1.0 / max(now - self._frame_ts, 1e-6)
        self._frame_ts = now