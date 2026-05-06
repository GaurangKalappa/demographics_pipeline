# ============================================================
#  config.py  –  Central configuration for the Demographics Pipeline
# ============================================================

# ---------- Video Input ----------
VIDEO_SOURCE = 0                   # 0 = webcam | "path/to/video.mp4" | "rtsp://..."
FPS_TARGET   = 20                  # Target capture FPS (changeable at runtime)

# ---------- Model Weights ----------
YOLO_PERSON_WEIGHTS = "weights/best_900.pt"             # Person detector (auto-downloads)
YOLO_FACE_WEIGHTS   = "weights/yolov8n-face.pt" # Face detector  (plug in your weights)
PAR_WEIGHTS         = "weights/par_model.pt"    # PP-Attribute / MobileNetV3 body model
MTL_WEIGHTS         = "weights/mtl_model.pt"    # MobileNetV3-Small face MTL model

# ---------- Detection Thresholds ----------
PERSON_CONF_THRESH = 0.50   # Minimum confidence to count as a person detection
FACE_CONF_THRESH   = 0.40   # Minimum face detection confidence to accept a crop

# ---------- Decision Logic Thresholds ----------
FACE_DOMINANCE_THRESH = 0.85   # C_face > this → use Face MTL exclusively (Case A)
BODY_DOMINANCE_THRESH = 0.60   # C_face < this → use Body PAR exclusively (Case B)
                                # Between the two          → weighted ensemble  (Case C)

# ---------- Tripwire (Line Crossing) ----------
# Defined as two (x, y) points in normalised [0,1] image coords.
# Default: a horizontal line 55% down the frame.
TRIPWIRE_START = (0.0, 0.55)
TRIPWIRE_END   = (1.0, 0.55)

# ---------- Buffer ----------
BUFFER_SIZE = 10    # Number of frames to accumulate per triggered Track ID

# ---------- Input Sizes ----------
PAR_INPUT_SIZE = (128, 256)    # (width, height)  – body crop for PAR model
MTL_INPUT_SIZE = (224, 224)    # (width, height)  – face crop for MTL model

# ---------- CLAHE Pre-processing ----------
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)

# ---------- Age Range Bins ----------
AGE_BINS = [
    (0,  12,  "Child"),
    (13, 18,  "Teen"),
    (19, 30,  "Young Adult"),
    (31, 45,  "Adult"),
    (46, 60,  "Middle-Aged"),
    (61, 999, "Senior"),
]

# ---------- Output ----------
LOG_PATH         = "logs/demographics.jsonl"   # One JSON record per line
DISPLAY_WINDOW   = True                        # Show live annotated frame
DISPLAY_WIN_NAME = "Demographics Pipeline"