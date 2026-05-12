# ============================================================
#  config.py  –  Central configuration for the Demographics Pipeline
# ============================================================

# ---------- Video Input ----------
VIDEO_SOURCE = "C:/Users/gaura/Documents/Internship/DE_project/dataset_videos/channel_1801_mrng.mp4"                   # 0 = webcam | "path/to/video.mp4" | "rtsp://..."
FPS_TARGET   = 20                  # Target capture FPS (changeable at runtime)

# ---------- Inference Resolution ----------
# Frame is downscaled to this width before being passed to YOLO.
# YOLO internally resizes to 640 anyway — passing a large CCTV frame
# wastes memory bandwidth every call.  Set to None to disable downscaling.
INFERENCE_WIDTH = 640  
# ---------- Display ----------
# Maximum display window dimensions.  The annotated frame is scaled DOWN
# to fit within these bounds before imshow — processing always runs at
# full resolution.  Increase if your monitor is larger than 1280×720.
DISPLAY_MAX_WIDTH  = 1280
DISPLAY_MAX_HEIGHT = 720

ROI = [(0.00078125, 0.4638888888888889), (0.27421875, 0.10138888888888889), (0.52734375, 0.09861111111111111), (0.58515625, 0.9930555555555556), (0.00234375, 0.9930555555555556)]  

# ---------- Model Weights ----------
YOLO_PERSON_WEIGHTS = "weights/best_900.pt"             # Person detector (auto-downloads)
YOLO_FACE_WEIGHTS   = "weights/yolov8n-face.pt" # Face detector  (plug in your weights)
PAR_WEIGHTS         = "weights/par_model.pt"    # PP-Attribute / MobileNetV3 body model
MTL_WEIGHTS         = "weights/mtl_model.pt"    # MobileNetV3-Small face MTL model

# ---------- ONNX Inference ----------
# Set USE_ONNX = True AFTER running:  python models/export_onnx.py
# ONNX Runtime is 2–4× faster than PyTorch on CPU for these model sizes.
USE_ONNX = True


YOLO_PERSON_WEIGHTS_ONNX = "weights/yolov8n_person.onnx"
YOLO_FACE_WEIGHTS_ONNX   = "weights/yolov8n_face.onnx"
PAR_WEIGHTS_ONNX         = "weights/par_model.onnx"
MTL_WEIGHTS_ONNX         = "weights/mtl_model.onnx"

# ---------- Detection Thresholds ----------
PERSON_CONF_THRESH = 0.50   # Minimum confidence to count as a person detection
FACE_CONF_THRESH   = 0.40   # Minimum face detection confidence to accept a crop

# ---------- Decision Logic Thresholds ----------
FACE_DOMINANCE_THRESH = 0.85   # C_face > this → use Face MTL exclusively (Case A)
BODY_DOMINANCE_THRESH = 0.60   # C_face < this → use Body PAR exclusively (Case B)
                                # Between the two          → weighted ensemble  (Case C)

# ---------- Buffer ----------
BUFFER_SIZE = 6    # Number of frames to accumulate per triggered Track ID

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