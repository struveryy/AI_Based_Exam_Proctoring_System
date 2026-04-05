import cv2
import os
import csv

class FaceRecognizer:
    def __init__(self):
        self.known_faces = {}
        self.student_db = {}
        self.load_faces()
        self.load_students()

    def load_faces(self):
        path = "data/faces"
        os.makedirs(path, exist_ok=True)

        for file in os.listdir(path):
            full_path = os.path.join(path, file)

            if not os.path.isfile(full_path):
                continue

            img = cv2.imread(full_path, 0)

            if img is not None:
                name = file.split(".")[0]
                self.known_faces[name] = img

    def load_students(self):
        try:
            with open("data/students.csv", newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.student_db[row["name"]] = row
        except:
            print("⚠️ students.csv not found")

    def recognize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for name, ref in self.known_faces.items():
            try:
                res = cv2.matchTemplate(gray, ref, cv2.TM_CCOEFF_NORMED)

                if res.max() > 0.6:
                    student = self.student_db.get(name, {})

                    return {
                        "name": name,
                        "roll_no": student.get("roll_no", "N/A"),
                        "course": student.get("course", "N/A")
                    }

            except:
                continue

        return {
            "name": "Unknown",
            "roll_no": "N/A",
            "course": "N/A"
        }