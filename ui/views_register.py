import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from engine.pipeline import FaceAttendancePipeline
from db.repository import AttendanceRepository

def render_register_view(pipeline: FaceAttendancePipeline, repo: AttendanceRepository):
    st.markdown("## 👤 User Enrollment & Biometric Registration")
    st.markdown("Register a new student or employee by providing their details and capturing their facial biometrics.")

    col_form, col_cam = st.columns([1, 1])

    with col_form:
        st.markdown("### 📝 Profile Information")
        user_code = st.text_input("User / Employee ID *", placeholder="e.g. EMP001 or STU202601").strip()
        name = st.text_input("Full Name *", placeholder="e.g. John Doe").strip()
        email = st.text_input("Email Address", placeholder="e.g. john.doe@example.com").strip()
        
        # Department with option to add new
        existing_depts = repo.get_departments()
        dept_options = ["Engineering", "Computer Science", "Human Resources", "Finance", "Marketing", "Operations"]
        for d in existing_depts:
            if d not in dept_options:
                dept_options.append(d)
        dept_options.append("+ Add New Department")

        selected_dept = st.selectbox("Department", dept_options)
        if selected_dept == "+ Add New Department":
            department = st.text_input("Enter Custom Department Name").strip()
        else:
            department = selected_dept

        role = st.selectbox("Role", ["Employee", "Student", "Faculty", "Staff", "Contractor"])

    with col_cam:
        st.markdown("### 📷 Face Capture & Validation")
        upload_choice = st.radio("Capture Method:", ["Take Photo via Webcam", "Upload Face Image File"], horizontal=True)
        
        bgr_image = None
        if upload_choice == "Take Photo via Webcam":
            cam_pic = st.camera_input("Capture Face Snapshot")
            if cam_pic:
                img = Image.open(cam_pic)
                bgr_image = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
        else:
            uploaded_file = st.file_uploader("Upload Portrait Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                img = Image.open(uploaded_file)
                bgr_image = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)

        # Validation and preview
        face_valid = False
        embedding = None
        aligned_crop = None

        if bgr_image is not None:
            # Detect faces
            detections = pipeline.detector.detect(bgr_image)
            
            if len(detections) == 0:
                st.error("❌ No face detected. Please ensure good lighting and face the camera directly.")
            elif len(detections) > 1:
                st.warning(f"⚠️ Multiple faces detected ({len(detections)}). Please ensure only ONE person is in the frame.")
            else:
                det = detections[0]
                quality_ok, q_msg, stats = pipeline.liveness.check_quality(bgr_image, det)
                
                # Extract aligned crop and embedding
                raw_face = det["raw_face"]
                aligned_crop = pipeline.recognizer.align_crop(bgr_image, raw_face)
                embedding = pipeline.recognizer.extract_features(bgr_image, raw_face)

                # Draw landmark overlay for user preview
                preview_img = bgr_image.copy()
                x, y, w, h = det["bbox"]
                cv2.rectangle(preview_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for pt in det["landmarks"]:
                    cv2.circle(preview_img, (pt[0], pt[1]), 3, (0, 255, 255), -1)

                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption="Face Detection & Landmarks", use_container_width=True)
                with col_p2:
                    if aligned_crop.size > 0:
                        st.image(cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2RGB), caption="Aligned 112x112 Crop", use_container_width=True)

                if quality_ok:
                    st.success(f"✅ Biometrics Validated! (Sharpness: {stats.get('sharpness', 0):.0f})")
                    face_valid = True
                else:
                    st.warning(f"⚠️ Warning: {q_msg}")
                    face_valid = True  # Still allow user to proceed if they insist

    st.markdown("---")
    submit_btn = st.button("💾 Complete Registration & Save Biometrics", type="primary", use_container_width=True)

    if submit_btn:
        if not user_code or not name:
            st.error("❌ Please provide both User / Employee ID and Full Name.")
            return

        if not face_valid or embedding is None:
            st.error("❌ Please capture/upload a valid face photo before saving.")
            return

        # Check for existing user code
        existing = repo.get_user_by_code(user_code)
        if existing:
            st.error(f"❌ A user with ID '{user_code}' already exists ({existing['name']}). Please use a unique ID.")
            return

        # Save profile photo
        photos_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "snapshots"
        )
        os.makedirs(photos_dir, exist_ok=True)
        photo_filename = f"profile_{user_code}.jpg"
        photo_path = os.path.join(photos_dir, photo_filename)
        try:
            cv2.imwrite(photo_path, aligned_crop if aligned_crop is not None else bgr_image)
        except Exception:
            photo_filename = ""

        # Insert user to database
        user_id = repo.add_user(
            user_code=user_code,
            name=name,
            email=email,
            department=department if department else "General",
            role=role,
            embedding=embedding,
            photo_path=photo_filename
        )

        # Refresh in-memory embeddings cache
        pipeline.refresh_database_cache()

        st.balloons()
        st.success(f"🎉 Successfully enrolled **{name}** (ID: `{user_code}`) into the Attendance System!")
