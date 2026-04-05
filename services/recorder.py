import cv2
from collections import deque
from datetime import datetime
import os

class ClipRecorder:
    def __init__(self, fps=20, seconds=10):
        self.fps = fps
        self.buffer_size = fps * seconds
        self.buffer = deque(maxlen=self.buffer_size)

        os.makedirs("data/clips", exist_ok=True)

    def update(self, frame):
        self.buffer.append(frame)

    def save_clip(self):
        if len(self.buffer) == 0:
            return None

        filename = f"data/clips/{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"

        h, w, _ = self.buffer[0].shape

        out = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'XVID'),
            self.fps,
            (w, h)
        )

        for frame in self.buffer:
            out.write(frame)

        out.release()

        return filename