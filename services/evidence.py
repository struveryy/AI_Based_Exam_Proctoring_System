import cv2
import os
from datetime import datetime

class EvidenceManager:
    def __init__(self):
        os.makedirs("data/evidence", exist_ok=True)

    def save_image(self, frame):
        filename = f"data/evidence/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        return filename