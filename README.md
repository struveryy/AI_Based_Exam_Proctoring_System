# 🎓 AI Exam Proctoring System

An AI-powered proctoring system that monitors students during exams
using CCTV/webcam.\
It detects suspicious activities like mobile usage, books, and looking
away, and automatically generates UFM (Unfair Means) reports with
evidence.

------------------------------------------------------------------------

## 🚀 Features

-   🧑‍🎓 Face Recognition (Name + Roll No)
-   🎥 Live Monitoring (Flask Web App)
-   📱 Cheating Detection (Phone, Book, Looking Away)
-   ⚠️ Warning System (1 → 2 → UFM)
-   🎥 Evidence Capture (Image + 10 sec video)
-   📋 Report Dashboard

------------------------------------------------------------------------

## 🧠 Tech Stack

-   Python
-   OpenCV
-   YOLOv8
-   Deep SORT
-   MediaPipe
-   Flask

------------------------------------------------------------------------

## ▶️ Run

``` bash
python web_app.py
```

Open: http://127.0.0.1:5000

------------------------------------------------------------------------

## 📁 Output

-   Reports → data/reports/
-   Images → data/evidence/images/
-   Videos → data/evidence/clips/

------------------------------------------------------------------------
🧪 How It Works
Camera captures live video
YOLO detects objects (phone/book)
DeepSORT tracks students
Face recognition identifies student
Head pose detects looking direction
Warning system triggers
After 3 violations → UFM generated
Evidence (image + video) saved
------------------------------------------------------------------------
## 👨‍💻 Author

Anubhaw vaish
