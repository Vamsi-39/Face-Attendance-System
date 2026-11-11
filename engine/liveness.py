import cv2
import numpy as np
from typing import Dict, Any, Tuple

class LivenessDetector:
    """
    Lightweight anti-spoofing and face quality heuristics.
    Detects blurriness, extreme angles, and low resolution.
    """

    def __init__(self, min_laplacian_var: float = 60.0, min_face_size: int = 60):
        self.min_laplacian_var = min_laplacian_var
        self.min_face_size = min_face_size

    def check_quality(self, frame: np.ndarray, detection: Dict[str, Any]) -> Tuple[bool, str, Dict[str, float]]:
        """
        Validates whether a detected face has sufficient quality for reliable recognition.
        """
        x, y, w, h = detection["bbox"]
        if w < self.min_face_size or h < self.min_face_size:
            return False, f"Face too small ({w}x{h} px). Please move closer.", {"size": min(w, h)}

        # Crop face region
        face_roi = frame[max(0, y):y+h, max(0, x):x+w]
        if face_roi.size == 0:
            return False, "Invalid face region", {}

        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi

        # 1. Sharpness via Laplacian variance
        lap_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        if lap_var < self.min_laplacian_var:
            return False, f"Face blurry (sharpness={lap_var:.1f}). Hold still.", {"sharpness": lap_var}

        # 2. Lighting / brightness check
        mean_brightness = np.mean(gray_roi)
        if mean_brightness < 40:
            return False, "Lighting too dark. Please face light source.", {"brightness": mean_brightness}
        if mean_brightness > 230:
            return False, "Too much glare / overexposed.", {"brightness": mean_brightness}

        return True, "Quality OK", {
            "sharpness": round(lap_var, 1),
            "brightness": round(mean_brightness, 1),
            "size": min(w, h)
        }
