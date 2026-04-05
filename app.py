import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp

from detector.face_recognition import FaceRecognizer

# ---------------- INIT ----------------
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection()

face_rec = FaceRecognizer()

cap = cv2.VideoCapture(0)

print("🚀 AI Proctor with Tracking Started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ---------------- YOLO DETECTION ----------------
    results = model(frame, verbose=False)[0]

    detections = []
    phone_detected = False

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, "person"))

        if label == "cell phone":
            phone_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "PHONE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ---------------- TRACKING ----------------
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w_box, h_box = map(int, track.to_ltrb())

        x1, y1, x2, y2 = l, t, l + w_box, t + h_box

        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        # Face detection inside person
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        faces = face_detector.process(rgb)

        name = "Unknown"
        roll = "N/A"

        if faces.detections:
            student = face_rec.recognize(person_crop)
            name = student["name"]
            roll = student["roll_no"]

        # Draw tracked box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"ID {track_id} | {name} ({roll})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------------- ALERT ----------------
    if phone_detected:
        cv2.putText(frame, "🚨 CHEATING DETECTED",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # ---------------- DISPLAY ----------------
    cv2.imshow("AI Proctor Tracking System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()