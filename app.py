import streamlit as st
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db.database import Database
from db.repository import AttendanceRepository
from engine.pipeline import FaceAttendancePipeline
from ui.views_attendance import render_attendance_view
from ui.views_register import render_register_view
from ui.views_analytics import render_analytics_view
from ui.views_settings import render_settings_view

# Page Config
st.set_page_config(
    page_title="AI Automated Attendance System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Modern Dashboard Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }
    .badge-status-present {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-status-late {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_system_components():
    """Initializes and caches database repository and AI pipeline."""
    db = Database()
    repo = AttendanceRepository(db=db)
    pipeline = FaceAttendancePipeline(repo=repo)
    return repo, pipeline

def main():
    repo, pipeline = get_system_components()
    org_name = repo.get_setting("org_name", "AI Vision Attendance Portal")

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(f"## 👁️ **VisionPass AI**")
        st.caption(f"📍 {org_name}")
        st.markdown("---")

        menu = st.radio(
            "Navigation Menu",
            [
                "📷 Live Attendance",
                "👤 Register New User",
                "📊 Analytics & Reports",
                "⚙️ Settings & Directory"
            ],
            index=0
        )

        st.markdown("---")
        
        # System Health info in sidebar
        st.markdown("#### ℹ️ System Status")
        st.success("🟢 CNN Engine: YuNet + SFace")
        st.info(f"👥 Enrolled Faces: **{len(pipeline.user_ids)}**")
        st.caption("Built with OpenCV 5.0 CNN & SQLite")

    # Route Views
    if menu == "📷 Live Attendance":
        render_attendance_view(pipeline, repo)
    elif menu == "👤 Register New User":
        render_register_view(pipeline, repo)
    elif menu == "📊 Analytics & Reports":
        render_analytics_view(repo)
    elif menu == "⚙️ Settings & Directory":
        render_settings_view(pipeline, repo)

if __name__ == "__main__":
    main()
