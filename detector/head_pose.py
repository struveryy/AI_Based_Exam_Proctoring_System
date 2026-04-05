import cv2
import mediapipe as mp

class HeadPoseEstimator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()

    def get_direction(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return "unknown"

        face = result.multi_face_landmarks[0]
        nose = face.landmark[1]

        if nose.x < 0.4:
            return "left"
        elif nose.x > 0.6:
            return "right"
        elif nose.y < 0.4:
            return "up"

        return "center"