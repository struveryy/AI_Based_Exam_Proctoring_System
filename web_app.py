from flask import Flask, render_template, Response, request, redirect, send_from_directory
import cv2, os, json, time
from detector.head_pose import HeadPoseEstimator
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
from detector.face_recognition import FaceRecognizer
from dotenv import load_dotenv
import os
from collections import deque
load_dotenv()

# ---------------- GLOBAL ----------------
app = Flask(__name__)

student_tracker = {}
COOLDOWN = 5  # seconds
FPS = int(os.getenv("FPS", 20))
BUFFER_SECONDS = int(os.getenv("BUFFER_SECONDS", 10))

frame_buffer = deque(maxlen=FPS * BUFFER_SECONDS)

# -------- BUFFER FOR VIDEO CLIP --------
COOLDOWN = int(os.getenv("COOLDOWN", 5))
FPS = int(os.getenv("FPS", 20))
BUFFER_SECONDS = int(os.getenv("BUFFER_SECONDS", 10))

# ---------------- INIT ----------------
model = YOLO(os.getenv("MODEL_PATH", "yolov8n.pt"))
tracker = DeepSort(max_age=30)
face_rec = FaceRecognizer()
head_pose = HeadPoseEstimator()
cap = cv2.VideoCapture(int(os.getenv("CAMERA_SOURCE", 0)))

# create folders
os.makedirs("data/reports", exist_ok=True)
os.makedirs("data/evidence/images", exist_ok=True)
os.makedirs("data/evidence/clips", exist_ok=True)

# ---------------- SAVE EVIDENCE ----------------
def save_evidence(name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    img_path = f"data/evidence/images/{name}_{timestamp}.jpg"
    video_path = f"data/evidence/clips/{name}_{timestamp}.avi"

    if len(frame_buffer) > 0:
        # save image
        cv2.imwrite(img_path, frame_buffer[-1])

        # save video
        h, w, _ = frame_buffer[0].shape
        out = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            FPS,
            (w, h)
        )

        for f in frame_buffer:
            out.write(f)

        out.release()

    return img_path, video_path

# ---------------- VIDEO STREAM ----------------
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # store frame in buffer
        frame_buffer.append(frame.copy())
        

        results = model(frame, verbose=False)[0]

        detections = []
        violations = []
        alert = False

        # ---------------- DETECTION ----------------
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "person":
                detections.append(([x1, y1, x2-x1, y2-y1], 0.9, "person"))

            if label in ["cell phone", "book"]:
                alert = True
                violations.append(label)
            direction = head_pose.get_direction(frame)

            if direction in ["left", "right", "up"]:
                alert = True
                violations.append(f"looking_{direction}")

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, label.upper(), (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv2.putText(frame, f"Direction: {direction}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255,255,0), 2)

        # ---------------- TRACKING ----------------
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())
            x1, y1, x2, y2 = l, t, l+w, t+h

            crop = frame[y1:y2, x1:x2]

            name = "Unknown"
            roll = "N/A"

            if crop.size > 0:
                student = face_rec.recognize(crop)
                name = student["name"]
                roll = student["roll_no"]

            # ---------------- DRAW ----------------
            color = (0,255,0)
            if alert:
                color = (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            label_text = f"ID {track_id} | {name} ({roll})"
            cv2.putText(frame, label_text, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # ---------------- WARNING + UFM ----------------
            current_time = time.time()

            if alert and name != "Unknown" and len(violations) > 0:

                if name not in student_tracker:
                    student_tracker[name] = {
                        "warnings": 0,
                        "last_time": 0
                    }

                data = student_tracker[name]

                # cooldown check
                if current_time - data["last_time"] >= COOLDOWN:

                    data["warnings"] += 1
                    data["last_time"] = current_time

                    print(f"⚠️ {name} Warning {data['warnings']}")

                    # -------- WARNING --------
                    if data["warnings"] < 3:
                        cv2.putText(frame,
                                    f"WARNING {data['warnings']}",
                                    (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 255), 2)

                    # -------- UFM --------
                    elif data["warnings"] >= 3:
                        print(f"🚨 UFM FILED FOR {name}")

                        img_path, video_path = save_evidence(name)

                        report = {
                            "name": name,
                            "roll_no": roll,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "violations": violations,
                            "warnings": data["warnings"],
                            "evidence_image": img_path,
                            "evidence_video": video_path,
                            "status": "Pending"
                        }

                        filename = f"data/reports/{name}_{int(current_time)}.json"

                        with open(filename, "w") as f:
                            json.dump(report, f, indent=4)

                        data["warnings"] = 0

                # show warning count
                cv2.putText(frame,
                            f"Warnings: {data['warnings']}",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255,255,0), 2)

        # global alert
        if alert:
            cv2.putText(frame, "🚨 CHEATING ALERT",
                        (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 3)

        # stream frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reports')
def reports():
    reports = []

    for file in os.listdir("data/reports"):
        with open(f"data/reports/{file}") as f:
            r = json.load(f)
            r["file"] = file
            reports.append(r)

    return render_template("reports.html", reports=reports)

@app.route('/report/<file>', methods=["GET","POST"])
def report(file):
    path = f"data/reports/{file}"

    with open(path) as f:
        data = json.load(f)

    if request.method == "POST":
        data["status"] = request.form["action"]

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        return redirect("/reports")

    return render_template("report.html", report=data)

# serve evidence files
@app.route('/data/<path:filename>')
def data_files(filename):
    return send_from_directory("data", filename)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)