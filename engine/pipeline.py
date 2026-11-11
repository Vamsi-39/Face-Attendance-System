import os
import cv2
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from engine.detector import YuNetFaceDetector
from engine.recognizer import SFaceRecognizer
from engine.liveness import LivenessDetector
from db.repository import AttendanceRepository

class FaceAttendancePipeline:
    """
    Unified end-to-end pipeline combining YuNet Face Detection, 
    SFace Deep Embedding Extraction, Quality Check, and Attendance Recording.
    """

    def __init__(self, repo: Optional[AttendanceRepository] = None,
                 detector_score_thresh: float = 0.85,
                 similarity_threshold: Optional[float] = None):
        self.repo = repo or AttendanceRepository()
        
        # Load configurable thresholds from database
        db_thresh = float(self.repo.get_setting("similarity_threshold", "0.363"))
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else db_thresh
        self.metric = self.repo.get_setting("match_metric", "cosine")

        self.detector = YuNetFaceDetector(score_threshold=detector_score_thresh)
        self.recognizer = SFaceRecognizer(cosine_threshold=self.similarity_threshold)
        self.liveness = LivenessDetector()

        self.snapshots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "snapshots"
        )
        os.makedirs(self.snapshots_dir, exist_ok=True)

        # In-memory cached embeddings
        self.user_ids: List[int] = []
        self.user_names: List[str] = []
        self.embeddings_matrix: np.ndarray = np.empty((0, 128), dtype=np.float32)
        self.refresh_database_cache()

    def refresh_database_cache(self):
        """Loads/updates the active user face embeddings from SQLite."""
        self.user_ids, self.user_names, self.embeddings_matrix = self.repo.get_registered_embeddings()

    def process_frame(self, frame: np.ndarray, auto_mark: bool = True, 
                      device_name: str = "Webcam") -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Processes a single video frame.
        
        Returns:
            - annotated_frame: Frame with HUD bounding boxes, names, and landmarks.
            - detections_info: List of all detected faces and their match status.
            - punch_events: List of newly marked attendance records in this frame.
        """
        if frame is None or frame.size == 0:
            return frame, [], []

        annotated = frame.copy()
        raw_detections = self.detector.detect(frame)
        detections_info = []
        punch_events = []

        for det in raw_detections:
            x, y, w, h = det["bbox"]
            landmarks = det["landmarks"]
            raw_face = det["raw_face"]

            # Quality & Liveness assessment
            quality_ok, quality_msg, quality_stats = self.liveness.check_quality(frame, det)

            # Feature extraction
            feat = self.recognizer.extract_features(frame, raw_face)

            # 1:N Match against registered user embeddings
            user_id, user_name, score, is_match = self.recognizer.match_best(
                query_feat=feat,
                user_ids=self.user_ids,
                user_names=self.user_names,
                embeddings_matrix=self.embeddings_matrix,
                metric=self.metric,
                threshold=self.similarity_threshold
            )

            det_info = {
                "bbox": [x, y, w, h],
                "user_id": user_id,
                "name": user_name if is_match else "Unknown",
                "score": score,
                "is_match": is_match,
                "quality_ok": quality_ok,
                "quality_msg": quality_msg,
                "landmarks": landmarks
            }
            detections_info.append(det_info)

            # Attendance Recording
            if is_match and auto_mark and user_id is not None:
                # Save snapshot photo of the punch event
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snapshot_name = f"punch_{user_id}_{timestamp_str}.jpg"
                snapshot_path = os.path.join(self.snapshots_dir, snapshot_name)
                
                # Try marking attendance
                marked, msg, rec = self.repo.mark_attendance(
                    user_id=user_id,
                    confidence=score,
                    snapshot_path=snapshot_name,
                    device=device_name
                )

                if marked:
                    # Save cropped/full snapshot
                    try:
                        cv2.imwrite(snapshot_path, frame)
                    except Exception:
                        pass
                    punch_events.append({
                        "user_id": user_id,
                        "name": user_name,
                        "status": rec.get("status", "Present"),
                        "time": rec.get("time", ""),
                        "message": msg,
                        "snapshot": snapshot_name
                    })

            # Draw Visual HUD on annotated frame
            self._draw_overlay(annotated, det_info)

        return annotated, detections_info, punch_events

    def _draw_overlay(self, img: np.ndarray, info: Dict[str, Any]):
        """Draws a modern HUD box, landmark dots, and name card over the face."""
        x, y, w, h = info["bbox"]
        is_match = info["is_match"]
        name = info["name"]
        score = info["score"]
        quality_ok = info["quality_ok"]

        # Color: Emerald Green for Recognized, Crimson/Orange for Unknown, Amber for Quality Warning
        if is_match:
            color = (34, 197, 94)      # Vibrant Green
            tag = f"VERIFIED: {name} ({score:.2f})"
        elif not quality_ok:
            color = (0, 165, 255)      # Amber Warning
            tag = f"{info['quality_msg']}"
        else:
            color = (59, 130, 246)     # Sky Blue / Unknown
            tag = f"UNKNOWN ({score:.2f})"

        # Sleek Corner Box
        line_len = max(15, int(min(w, h) * 0.2))
        thickness = 2
        
        # Bounding box border
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

        # Stylized Corners
        # Top-Left
        cv2.line(img, (x, y), (x + line_len, y), color, thickness + 1)
        cv2.line(img, (x, y), (x, y + line_len), color, thickness + 1)
        # Top-Right
        cv2.line(img, (x + w, y), (x + w - line_len, y), color, thickness + 1)
        cv2.line(img, (x + w, y), (x + w, y + line_len), color, thickness + 1)
        # Bottom-Left
        cv2.line(img, (x, y + h), (x + line_len, y + h), color, thickness + 1)
        cv2.line(img, (x, y + h), (x, y + h - line_len), color, thickness + 1)
        # Bottom-Right
        cv2.line(img, (x + w, y + h), (x + w - line_len, y + h), color, thickness + 1)
        cv2.line(img, (x + w, y + h), (x + w, y + h - line_len), color, thickness + 1)

        # Draw 5 Facial Landmark Dots
        for pt in info.get("landmarks", []):
            cv2.circle(img, (pt[0], pt[1]), 2, (0, 255, 255), -1)

        # Label Pill
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        pill_y = max(24, y - 8)
        cv2.rectangle(img, (x, pill_y - th - 6), (x + tw + 12, pill_y + 4), color, -1)
        cv2.putText(img, tag, (x + 6, pill_y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
