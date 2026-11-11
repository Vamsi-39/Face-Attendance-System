import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

DEFAULT_YUNET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_detection_yunet_2023mar.onnx"
)

class YuNetFaceDetector:
    """
    CNN-based YuNet face detector using OpenCV FaceDetectorYN.
    Detects face bounding boxes and 5 facial landmarks (right eye, left eye, nose tip, right mouth, left mouth).
    """

    def __init__(self, model_path: str = DEFAULT_YUNET_PATH, 
                 score_threshold: float = 0.85, 
                 nms_threshold: float = 0.3, 
                 top_k: int = 5000):
        self.model_path = model_path
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.detector = None
        self._input_size = (320, 320)
        self.init_detector()

    def init_detector(self):
        """Initializes FaceDetectorYN model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YuNet model not found at '{self.model_path}'. Run models/download_models.py first.")

        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )

    def set_score_threshold(self, threshold: float):
        self.score_threshold = max(0.1, min(0.99, float(threshold)))
        if self.detector:
            self.detector.setScoreThreshold(self.score_threshold)

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in an RGB/BGR image.
        Returns a list of dicts containing:
          - 'bbox': [x, y, w, h] (integers)
          - 'score': confidence score (float 0.0 - 1.0)
          - 'landmarks': 5 landmark (x, y) coordinates
          - 'raw_face': full 15-element detection array for SFace alignment
        """
        if image is None or image.size == 0 or self.detector is None:
            return []

        h, w = image.shape[:2]
        if (w, h) != self._input_size:
            self._input_size = (w, h)
            self.detector.setInputSize(self._input_size)

        # Detect faces
        _, faces = self.detector.detect(image)
        if faces is None:
            return []

        results = []
        for face in faces:
            # Face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, fw, fh = map(int, face[0:4])
            score = float(face[-1])
            landmarks = face[4:14].reshape((5, 2)).astype(int).tolist()

            # Ensure bounding box is within frame boundaries
            x = max(0, x)
            y = max(0, y)
            fw = min(w - x, fw)
            fh = min(h - y, fh)

            if fw > 10 and fh > 10:
                results.append({
                    "bbox": [x, y, fw, fh],
                    "score": score,
                    "landmarks": landmarks,
                    "raw_face": face
                })

        return results
