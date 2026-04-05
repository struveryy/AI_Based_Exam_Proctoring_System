from ultralytics import YOLO
from models.model_loader import load_yolo_model
class ObjectDetector:
    def __init__(self):
        from models.model_loader import load_yolo_model

class ObjectDetector:
    def __init__(self):
        self.model = load_yolo_model()

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]

            if label in ["cell phone"]:
                detections.append(label)

        return detections

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]

            if label in ["cell phone"]:
                detections.append(label)

        return detections