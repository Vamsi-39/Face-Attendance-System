import streamlit as st
import cv2
import time
import numpy as np
from datetime import datetime
from PIL import Image
from engine.pipeline import FaceAttendancePipeline
from db.repository import AttendanceRepository

def render_attendance_view(pipeline: FaceAttendancePipeline, repo: AttendanceRepository):
    st.markdown("## 📷 Live Attendance Terminal")
    st.markdown("Position your face towards the webcam to automatically record attendance.")

    # Refresh embedding cache from DB
    pipeline.refresh_database_cache()
    active_count = len(pipeline.user_ids)

    if active_count == 0:
        st.warning("⚠️ No active users registered in the system yet. Please go to **Register User** tab to enroll faces.")

    # Top metrics bar
    summary = repo.get_daily_summary()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Enrolled", f"{summary['total_users']} users")
    with col2:
        st.metric("Present Today", f"{summary['present_count']} ({summary['attendance_rate']}%)")
    with col3:
        st.metric("On-Time", f"{summary['on_time']}", delta="Normal", delta_color="normal")
    with col4:
        st.metric("Late Arrivals", f"{summary['late']}", delta="Delayed", delta_color="inverse")

    st.markdown("---")

    # Mode selector: Continuous Video Stream vs Snapshot Capture
    capture_mode = st.radio(
        "Select Camera Input Mode:",
        ["📹 Continuous Live Webcam Stream", "📸 Snapshot / Upload Check-in"],
        horizontal=True
    )

    col_cam, col_logs = st.columns([3, 2])

    with col_cam:
        if "Continuous" in capture_mode:
            st.markdown("### 🔴 Live Scanner Feed")
            
            run_camera = st.checkbox("▶️ Start Live Webcam Stream", value=False, key="run_webcam_stream")
            video_placeholder = st.empty()
            status_placeholder = st.empty()

            if run_camera:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Could not access webcam. Please ensure a camera is connected and not in use by another app.")
                else:
                    status_placeholder.info("🟢 Camera Active. Detecting faces in real time...")
                    stop_stream = False
                    
                    # Live streaming loop
                    while run_camera and not stop_stream:
                        ret, frame = cap.read()
                        if not ret:
                            status_placeholder.error("Failed to read frame from camera.")
                            break

                        # Process frame through deep CNN pipeline
                        annotated, detections, punches = pipeline.process_frame(frame, auto_mark=True)

                        # Convert BGR to RGB for Streamlit display
                        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        # Handle new attendance event
                        if punches:
                            for punch in punches:
                                st.toast(f"✅ {punch['message']} for **{punch['name']}**!", icon="🎉")

                        time.sleep(0.03)

                    cap.release()
                    video_placeholder.empty()
                    status_placeholder.info("⏹️ Camera Stream stopped.")
            else:
                video_placeholder.info("Click '▶️ Start Live Webcam Stream' above to launch the real-time AI recognition scanner.")

        else:
            st.markdown("### 📸 Snapshot Attendance")
            camera_photo = st.camera_input("Take a photo to mark attendance:")
            
            if camera_photo is not None:
                # Convert PIL to BGR OpenCV format
                image = Image.open(camera_photo)
                img_array = np.array(image.convert('RGB'))
                bgr_frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                with st.spinner("Analyzing face with CNN engine..."):
                    annotated, detections, punches = pipeline.process_frame(bgr_frame, auto_mark=True)

                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Processed Recognition Result", use_container_width=True)

                if punches:
                    for p in punches:
                        st.success(f"🎉 **{p['name']}**: {p['message']}")
                elif detections:
                    for d in detections:
                        if not d["is_match"]:
                            st.error(f"❌ Unknown Face (Match score: {d['score']:.2f} < threshold). User not registered.")
                else:
                    st.warning("⚠️ No face detected in the photo. Please ensure face is centered and well-lit.")

    with col_logs:
        st.markdown("### 📋 Today's Punch Feed")
        today_str = datetime.now().strftime("%Y-%m-%d")
        records = repo.get_attendance_records(start_date=today_str, end_date=today_str)

        if not records:
            st.info("No attendance records logged today yet.")
        else:
            for r in records[:10]:
                status_color = "green" if r["status"] == "Present" else "orange"
                st.markdown(f"""
                <div style="padding: 10px; margin-bottom: 8px; border-radius: 8px; border-left: 5px solid {status_color}; background-color: #f8fafc; color: #1e293b;">
                    <div style="font-weight: 600; font-size: 15px;">{r['name']} <span style="font-size: 12px; color: #64748b;">({r['user_code']})</span></div>
                    <div style="font-size: 13px; color: #475569;">🕒 Time: <b>{r['time']}</b> &nbsp;|&nbsp; Dept: <b>{r['department']}</b></div>
                    <div style="font-size: 12px; margin-top: 4px;">Status: <b style="color: {status_color};">{r['status']}</b> &bull; Confidence: {r['confidence']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
