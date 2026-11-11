# 👁️ VisionPass AI – Automated Face Recognition Attendance System

An enterprise-grade, real-time Automated Attendance System built with **OpenCV Deep Learning CNNs (YuNet & SFace)**, **SQLite**, and an interactive **Streamlit Dashboard** and standalone **OpenCV Kiosk**.

---

## 🌟 Key Features

- 🧠 **Dual Deep Learning CNN Architecture**:
  - **YuNet (`FaceDetectorYN`)**: Ultra-fast, edge-optimized CNN face detector with 5 facial landmark predictions.
  - **SFace (`FaceRecognizerSF`)**: Deep CNN representation extractor generating 128-dimensional biometric embeddings.
- ⚡ **Real-Time Webcam Recognition**: Live face detection and automated punch-in with high FPS.
- 🛡️ **Anti-Duplicate Cooldown**: Prevents multiple attendance records for the same person within a configurable window (e.g. 5 minutes).
- 🕒 **Smart Time Classification**: Classifies punches into **Present** or **Late** according to configurable cutoff thresholds (e.g. 09:30 AM).
- 📸 **Audit Snapshots**: Saves timestamped photo captures of every punch event for compliance and verification.
- 📊 **Rich Analytics Dashboard**: Filter attendance records by Date Range, Department, or Status, with interactive bar charts, pie charts, and hourly trends.
- 📥 **Export to CSV & JSON**: One-click download of attendance logs for payroll and HR systems.
- 👤 **Biometric Enrollment Portal**: Register users via live webcam capture or photo upload with face quality and sharpness validation.
- 🖥️ **Standalone Kiosk Mode**: High-performance desktop HUD window (`kiosk.py`) for dedicated hardware terminals or turnstiles.

---

## 📁 Project Structure

```
face_attendance_system/
├── models/
│   ├── download_models.py                # Auto-downloader for YuNet & SFace ONNX models
│   ├── face_detection_yunet_2023mar.onnx # YuNet Face Detector
│   └── face_recognition_sface_2021dec.onnx # SFace Face Recognizer
├── engine/
│   ├── __init__.py
│   ├── detector.py                       # YuNet FaceDetectorYN wrapper
│   ├── recognizer.py                     # SFace FaceRecognizerSF wrapper
│   ├── liveness.py                       # Sharpness & lighting quality checks
│   └── pipeline.py                       # Unified face recognition pipeline
├── db/
│   ├── __init__.py
│   ├── database.py                       # SQLite connection & schema initialization
│   └── repository.py                     # High-level CRUD operations
├── ui/
│   ├── __init__.py
│   ├── views_attendance.py               # Live camera attendance terminal view
│   ├── views_register.py                 # User face enrollment view
│   ├── views_analytics.py                # Visual charts & export view
│   └── views_settings.py                 # System parameters & user management
├── data/
│   ├── attendance.db                     # SQLite database
│   └── snapshots/                        # Punch snapshots & profile crops
├── app.py                                # Main Streamlit Web Application
├── kiosk.py                              # Standalone OpenCV Desktop Kiosk
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Models Download
The ONNX models download automatically upon first launch, or you can run:
```bash
python models/download_models.py
```

### 3. Launch Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Launch Standalone Kiosk Mode
For dedicated physical terminals or turnstiles with an OpenCV desktop HUD:
```bash
python kiosk.py
```
- Press **Q** or **ESC** to exit.
- Press **R** to refresh user embeddings from the database.

---

## 🗄️ Database Schema

The system uses SQLite (`data/attendance.db`) with Write-Ahead Logging (WAL) enabled:

1. **`users`**:
   - `id` (INTEGER PRIMARY KEY)
   - `user_code` (TEXT UNIQUE)
   - `name` (TEXT)
   - `email` (TEXT)
   - `department` (TEXT)
   - `role` (TEXT)
   - `embedding` (BLOB, 128-float32 vector)
   - `photo_path` (TEXT)
   - `is_active` (INTEGER)
   - `created_at` (TIMESTAMP)

2. **`attendance`**:
   - `id` (INTEGER PRIMARY KEY)
   - `user_id` (INTEGER, FK to `users.id`)
   - `date` (TEXT, YYYY-MM-DD)
   - `time` (TEXT, HH:MM:SS)
   - `status` (TEXT: 'Present', 'Late', 'Half-Day')
   - `confidence` (REAL)
   - `snapshot_path` (TEXT)
   - `device` (TEXT)

3. **`system_settings`**:
   - Key-value configuration for thresholds, cooldowns, and schedule cutoffs.
