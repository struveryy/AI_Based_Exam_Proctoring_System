import json
import os
from datetime import datetime

class ReportManager:
    def __init__(self):
        os.makedirs("data/reports", exist_ok=True)

    def create(self, student_info, violations, evidence_path, clip_path):
        report = {
            "name": student_info["name"],
            "roll_no": student_info["roll_no"],
            "course": student_info["course"],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "violations": violations,
            "evidence_image": evidence_path,
            "evidence_clip": clip_path,
            "status": "Pending"
        }

        filename = f"data/reports/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, "w") as f:
            json.dump(report, f, indent=4)

        return filename

    def load_all_reports(self):
        reports = []

        for file in os.listdir("data/reports"):
            path = f"data/reports/{file}"

            if not os.path.isfile(path):
                continue

            with open(path, "r") as f:
                data = json.load(f)
                data["file"] = file
                reports.append(data)

        return reports

    def update_status(self, file, new_status):
        path = f"data/reports/{file}"

        with open(path, "r") as f:
            data = json.load(f)

        data["status"] = new_status

        with open(path, "w") as f:
            json.dump(data, f, indent=4)