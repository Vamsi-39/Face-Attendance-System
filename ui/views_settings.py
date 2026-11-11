import os
import streamlit as st
import pandas as pd
from engine.pipeline import FaceAttendancePipeline
from db.repository import AttendanceRepository

def render_settings_view(pipeline: FaceAttendancePipeline, repo: AttendanceRepository):
    st.markdown("## ⚙️ System Settings & User Directory")
    st.markdown("Manage system recognition parameters, late arrival thresholds, and enrolled users.")

    tab_users, tab_config = st.tabs(["👥 Enrolled Users Directory", "🛠️ AI & Threshold Configuration"])

    with tab_users:
        st.markdown("### 📋 Registered Individuals")
        users = repo.get_all_users()

        if not users:
            st.info("No registered users found. Use the **Register User** tab to enroll individuals.")
        else:
            df_users = pd.DataFrame(users)
            display_cols = ["id", "user_code", "name", "email", "department", "role", "is_active", "created_at"]
            st.dataframe(
                df_users[display_cols].rename(columns={
                    "id": "DB ID",
                    "user_code": "User Code",
                    "name": "Full Name",
                    "email": "Email",
                    "department": "Department",
                    "role": "Role",
                    "is_active": "Active",
                    "created_at": "Enrolled Date"
                }),
                use_container_width=True
            )

            st.markdown("---")
            st.markdown("#### 🗑️ Manage Individual User")
            user_map = {f"{u['name']} ({u['user_code']}) - ID #{u['id']}": u['id'] for u in users}
            selected_user_str = st.selectbox("Select User to Manage:", list(user_map.keys()))
            target_user_id = user_map[selected_user_str]
            target_user = repo.get_user_by_id(target_user_id)

            c1, c2 = st.columns(2)
            with c1:
                new_active = st.checkbox("Active Status (Permitted to scan)", value=bool(target_user.get("is_active", 1)))
                if st.button("Update Status"):
                    repo.update_user(
                        user_id=target_user_id,
                        name=target_user["name"],
                        user_code=target_user["user_code"],
                        email=target_user.get("email", ""),
                        department=target_user.get("department", "General"),
                        role=target_user.get("role", "Member"),
                        is_active=1 if new_active else 0
                    )
                    pipeline.refresh_database_cache()
                    st.success("User status updated successfully.")
                    st.rerun()

            with c2:
                if st.button("🚨 Delete User & Biometrics", type="secondary"):
                    deleted = repo.delete_user(target_user_id)
                    if deleted:
                        pipeline.refresh_database_cache()
                        st.warning(f"User #{target_user_id} and corresponding records deleted.")
                        st.rerun()

    with tab_config:
        st.markdown("### 🔧 AI Engine & Business Rules Configuration")
        
        # Load current values
        current_thresh = float(repo.get_setting("similarity_threshold", "0.363"))
        current_cutoff = repo.get_setting("late_cutoff_time", "09:30:00")
        current_cooldown = int(repo.get_setting("cooldown_seconds", "300"))
        current_min_face = float(repo.get_setting("min_face_confidence", "0.85"))
        current_org = repo.get_setting("org_name", "AI Vision Attendance Portal")

        with st.form("settings_form"):
            org_name = st.text_input("Organization / Institution Name", value=current_org)
            
            st.markdown("#### 🎯 Face Recognition Sensitivity")
            st.markdown("Cosine Similarity Threshold for SFace CNN (Higher = stricter match, Lower = more lenient):")
            sim_threshold = st.slider("Similarity Threshold", min_value=0.20, max_value=0.70, value=current_thresh, step=0.01)

            st.markdown("#### 🕒 Attendance Time Rules")
            cutoff_time_str = st.text_input("Late Arrival Cutoff Time (HH:MM:SS)", value=current_cutoff)

            cooldown_mins = st.slider("Anti-Duplicate Cooldown Window (Minutes)", min_value=1, max_value=60, value=current_cooldown // 60)

            min_face_score = st.slider("YuNet Face Detection Confidence", min_value=0.50, max_value=0.99, value=current_min_face, step=0.05)

            saved = st.form_submit_button("💾 Save System Configuration", type="primary")

            if saved:
                repo.set_setting("org_name", org_name, "Organization Name")
                repo.set_setting("similarity_threshold", str(sim_threshold), "SFace Cosine Threshold")
                repo.set_setting("late_cutoff_time", cutoff_time_str, "Late cutoff time")
                repo.set_setting("cooldown_seconds", str(cooldown_mins * 60), "Duplicate cooldown seconds")
                repo.set_setting("min_face_confidence", str(min_face_score), "Detection confidence")

                # Update running pipeline instance
                pipeline.similarity_threshold = sim_threshold
                pipeline.recognizer.cosine_threshold = sim_threshold
                pipeline.detector.set_score_threshold(min_face_score)

                st.success("✅ Configuration settings saved and applied immediately!")
