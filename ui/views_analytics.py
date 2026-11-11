import os
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from db.repository import AttendanceRepository

def render_analytics_view(repo: AttendanceRepository):
    st.markdown("## 📊 Attendance Analytics & Reporting")
    st.markdown("Query, visualize, and export comprehensive attendance records.")

    # Filter Bar
    with st.expander("🔍 Search & Filter Options", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            today = datetime.now().date()
            default_start = today - timedelta(days=7)
            date_range = st.date_input("Date Range", (default_start, today))
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range[0].strftime("%Y-%m-%d"), date_range[1].strftime("%Y-%m-%d")
            elif isinstance(date_range, tuple) and len(date_range) == 1:
                start_date = end_date = date_range[0].strftime("%Y-%m-%d")
            else:
                start_date = end_date = today.strftime("%Y-%m-%d")

        with col2:
            depts = ["All"] + repo.get_departments()
            selected_dept = st.selectbox("Department", depts)

        with col3:
            selected_status = st.selectbox("Status", ["All", "Present", "Late", "Half-Day"])

        with col4:
            search_query = st.text_input("Search Name / ID", placeholder="Type name or code...").strip()

    # Query records from database
    records = repo.get_attendance_records(
        start_date=start_date,
        end_date=end_date,
        department=selected_dept,
        status=selected_status,
        search=search_query
    )

    df = pd.DataFrame(records)

    # Executive Summary Metrics
    summary = repo.get_daily_summary()
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Records Found", f"{len(df)}")
    with m2:
        st.metric("Total Registered", f"{summary['total_users']}")
    with m3:
        st.metric("Present Today", f"{summary['present_count']}")
    with m4:
        st.metric("Late Today", f"{summary['late']}")
    with m5:
        st.metric("Today Attendance", f"{summary['attendance_rate']}%")

    st.markdown("---")

    # Visual Analytics Charts
    tab_charts, tab_table, tab_export = st.tabs(["📈 Visual Insights", "📋 Records Table", "📥 Downloads & Exports"])

    with tab_charts:
        if df.empty:
            st.info("No attendance records found matching the selected criteria.")
        else:
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### 🏢 Department-Wise Attendance")
                dept_counts = df.groupby("department")["id"].count().reset_index()
                dept_counts.columns = ["Department", "Punches"]
                
                chart_dept = alt.Chart(dept_counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("Department:N", sort="-y", title="Department"),
                    y=alt.Y("Punches:Q", title="Total Attendance Punches"),
                    color=alt.Color("Department:N", legend=None),
                    tooltip=["Department", "Punches"]
                ).properties(height=280)
                st.altair_chart(chart_dept, use_container_width=True)

            with c2:
                st.markdown("#### ⏱️ Status Breakdown (On-Time vs Late)")
                status_counts = df["status"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]

                chart_status = alt.Chart(status_counts).mark_arc(innerRadius=45).encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(field="Status", type="nominal", scale=alt.Scale(
                        domain=["Present", "Late", "Half-Day"],
                        range=["#22c55e", "#f97316", "#eab308"]
                    )),
                    tooltip=["Status", "Count"]
                ).properties(height=280)
                st.altair_chart(chart_status, use_container_width=True)

            # Hourly punch distribution
            st.markdown("#### 🕒 Hourly Punch Activity")
            if "time" in df.columns:
                df["Hour"] = df["time"].apply(lambda t: t.split(":")[0] + ":00" if isinstance(t, str) and ":" in t else "00:00")
                hourly_counts = df.groupby("Hour")["id"].count().reset_index()
                hourly_counts.columns = ["Hour", "Punches"]

                chart_hour = alt.Chart(hourly_counts).mark_line(point=True, color="#3b82f6").encode(
                    x=alt.X("Hour:N", title="Hour of Day"),
                    y=alt.Y("Punches:Q", title="Number of Punches"),
                    tooltip=["Hour", "Punches"]
                ).properties(height=220)
                st.altair_chart(chart_hour, use_container_width=True)

    with tab_table:
        if df.empty:
            st.info("No records to display.")
        else:
            st.markdown("#### 📜 Attendance Records")
            display_cols = ["date", "time", "user_code", "name", "department", "role", "status", "confidence", "device"]
            existing_display_cols = [c for c in display_cols if c in df.columns]
            
            st.dataframe(
                df[existing_display_cols].rename(columns={
                    "date": "Date",
                    "time": "Time",
                    "user_code": "User ID",
                    "name": "Full Name",
                    "department": "Department",
                    "role": "Role",
                    "status": "Status",
                    "confidence": "Match Score",
                    "device": "Device"
                }),
                use_container_width=True,
                height=380
            )

            # Expandable Snapshot Viewer
            with st.expander("🖼️ View Verification Punch Snapshots"):
                snapshots_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "snapshots"
                )
                records_with_photos = [r for r in records if r.get("snapshot_path")]
                if not records_with_photos:
                    st.info("No snapshots available for the current selection.")
                else:
                    grid_cols = st.columns(4)
                    for i, r in enumerate(records_with_photos[:8]):
                        photo_path = os.path.join(snapshots_dir, r["snapshot_path"])
                        if os.path.exists(photo_path):
                            with grid_cols[i % 4]:
                                st.image(photo_path, caption=f"{r['name']} ({r['date']} {r['time']})", use_container_width=True)

    with tab_export:
        st.markdown("### 📥 Download Options & Data Export")
        st.markdown("Download reports, raw database, or the complete project source code archive directly:")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("#### 📄 Export Attendance Data")
            if not df.empty:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV Report (.csv)",
                    data=csv_data,
                    file_name=f"attendance_report_{start_date}_to_{end_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                json_data = df.to_json(orient="records", indent=2).encode('utf-8')
                st.download_button(
                    label="📋 Download JSON Report (.json)",
                    data=json_data,
                    file_name=f"attendance_report_{start_date}_to_{end_date}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("No records available to export for this date filter.")

        with col_ex2:
            st.markdown("#### 📦 Download System Backups & Project")
            
            # Database File Download
            db_path = repo.db.db_path
            if os.path.exists(db_path):
                with open(db_path, "rb") as db_file:
                    db_bytes = db_file.read()
                st.download_button(
                    label="🗄️ Download SQLite Database (attendance.db)",
                    data=db_bytes,
                    file_name="attendance.db",
                    mime="application/x-sqlite3",
                    use_container_width=True
                )

            # Full Project Zip Archive Download
            zip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "face_attendance_system.zip")
            if not os.path.exists(zip_path):
                zip_path = r"C:\Users\shaik\Downloads\face_attendance_system.zip"
            
            if os.path.exists(zip_path):
                with open(zip_path, "rb") as zf:
                    zip_bytes = zf.read()
                st.download_button(
                    label="📦 Download Complete Project (.zip)",
                    data=zip_bytes,
                    file_name="face_attendance_system.zip",
                    mime="application/zip",
                    use_container_width=True
                )
