import os
from ultralytics import YOLO
from config import YOLO_MODEL_PATH


def load_yolo_model():
    # Check if model exists
    if not os.path.exists(YOLO_MODEL_PATH):
        print("⬇️ YOLO model not found. Downloading...")

        # This will auto-download model
        model = YOLO("yolov8n.pt")

        # Save it locally
        os.makedirs(os.path.dirname(YOLO_MODEL_PATH), exist_ok=True)
        model.save(YOLO_MODEL_PATH)

        print("✅ Model downloaded and saved")

    else:
        print("✅ YOLO model found")

    # Load model
    model = YOLO(YOLO_MODEL_PATH)
    return model