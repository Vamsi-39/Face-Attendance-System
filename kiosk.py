import cv2
import time
import os
import sys
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db.database import Database
from db.repository import AttendanceRepository
from engine.pipeline import FaceAttendancePipeline

def run_kiosk(camera_id: int = 0):
    print("=" * 60)
    print("      AUTOMATED ATTENDANCE SYSTEM - KIOSK HUD MODE     ")
    print("=" * 60)
    print("[INFO] Initializing SQLite database and Deep CNN Models...")
    
    db = Database()
    repo = AttendanceRepository(db=db)
    pipeline = FaceAttendancePipeline(repo=repo)

    print(f"[OK] Pipeline loaded. Active registered users: {len(pipeline.user_ids)}")
    print("[INFO] Opening camera ID:", camera_id)
    print("[INFO] Controls:")
    print("       - Press 'Q' or 'ESC' to Quit")
    print("       - Press 'R' to Reload User Embeddings from Database")
    print("=" * 60)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "AI Automated Face Attendance System [KIOSK MODE]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps_counter = 0
    fps = 0.0
    start_time = time.time()
    last_notification = ""
    notif_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera frame read failed.")
            break

        # Process frame
        annotated, detections, punches = pipeline.process_frame(frame, auto_mark=True, device_name="Kiosk-Terminal-1")

        if punches:
            p = punches[0]
            last_notification = f"SUCCESS: {p['name']} ({p['status']} at {p['time']})"
            notif_timer = time.time() + 3.0  # Show for 3 seconds
            print(f"[ATTENDANCE RECORDED] {p['name']} | Status: {p['status']} | Time: {p['time']}")

        # Calculate FPS
        fps_counter += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            start_time = time.time()

        # Top Header Bar
        h, w = annotated.shape[:2]
        header_bg = annotated.copy()
        cv2.rectangle(header_bg, (0, 0), (w, 50), (15, 23, 42), -1)
        cv2.addWeighted(header_bg, 0.85, annotated, 0.15, 0, annotated)

        # Title & Clock
        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(annotated, "VISIONPASS AI - ATTENDANCE KIOSK", (15, 32), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"{now_str}  |  FPS: {fps:.1f}", (w - 360, 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, (56, 189, 248), 1, cv2.LINE_AA)

        # Bottom Toast Banner if active
        if time.time() < notif_timer and last_notification:
            cv2.rectangle(annotated, (0, h - 50), (w, h), (34, 197, 94), -1)
            cv2.putText(annotated, last_notification, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            print("[INFO] Exiting Kiosk Mode...")
            break
        elif key in [ord('r'), ord('R')]:
            pipeline.refresh_database_cache()
            print(f"[INFO] Database cache refreshed. Active users: {len(pipeline.user_ids)}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_kiosk(cam_index)
