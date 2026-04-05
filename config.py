import os

# =========================
# 📂 PATH SETTINGS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

MODEL_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")


# =========================
# 🎥 VIDEO SETTINGS
# =========================
CAMERA_SOURCE = 0  
# Replace with CCTV:
# CAMERA_SOURCE = "rtsp://username:password@ip:port/stream"

FPS = 20
CLIP_SECONDS = 10


# =========================
# 🧠 AI SETTINGS
# =========================

# Face recognition threshold (OpenCV template match)
FACE_MATCH_THRESHOLD = 0.6

# Head movement thresholds
LOOK_LEFT_THRESHOLD = 0.3
LOOK_RIGHT_THRESHOLD = 0.7

# Violation sensitivity
MAX_LOOK_AWAY = 15   # increase to reduce false positives


# =========================
# ⚠️ VIOLATION RULES
# =========================
ENABLE_PHONE_DETECTION = True
ENABLE_HEAD_MOVEMENT = True
ENABLE_FACE_MISSING = True


# =========================
# 📝 REPORT SETTINGS
# =========================
DEFAULT_STATUS = "Pending"

# Cooldown to avoid spam reports (seconds)
REPORT_COOLDOWN = 5