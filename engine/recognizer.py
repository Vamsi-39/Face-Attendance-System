import os
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

DEFAULT_SFACE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_recognition_sface_2021dec.onnx"
)

DEFAULT_COSINE_THRESHOLD = 0.363
DEFAULT_L2_THRESHOLD = 1.128

class SFaceRecognizer:
    """
    CNN-based SFace feature extractor and matcher using OpenCV FaceRecognizerSF.
    Extracts 128-dimensional deep biometric embeddings from aligned facial crops.
    """

    def __init__(self, model_path: str = DEFAULT_SFACE_PATH, 
                 cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
                 l2_threshold: float = DEFAULT_L2_THRESHOLD):
        self.model_path = model_path
        self.cosine_threshold = float(cosine_threshold)
        self.l2_threshold = float(l2_threshold)
        self.recognizer = None
        self.init_recognizer()

    def init_recognizer(self):
        """Initializes FaceRecognizerSF model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SFace model not found at '{self.model_path}'. Run models/download_models.py first.")

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config=""
        )

    def align_crop(self, image: np.ndarray, raw_face: np.ndarray) -> np.ndarray:
        """
        Warp and align face to standardized 112x112 image using 5 landmarks.
        """
        if self.recognizer is None or image is None or raw_face is None:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return self.recognizer.alignCrop(image, raw_face)

    def extract_features(self, image: np.ndarray, raw_face: np.ndarray) -> np.ndarray:
        """
        Extracts 128-D normalized biometric embedding vector from face.
        """
        aligned_face = self.align_crop(image, raw_face)
        if aligned_face.size == 0:
            return np.zeros((128,), dtype=np.float32)
        
        feature = self.recognizer.feature(aligned_face)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature.flatten().astype(np.float32)

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray, metric: str = "cosine") -> float:
        f1 = feat1.reshape(1, -1).astype(np.float32)
        f2 = feat2.reshape(1, -1).astype(np.float32)

        if metric.lower() == "cosine":
            sim = float(np.dot(f1, f2.T)[0, 0])
            return sim
        else:
            dist = float(np.linalg.norm(f1 - f2))
            return dist

    def match_best(self, query_feat: np.ndarray, 
                   user_ids: List[int], 
                   user_names: List[str], 
                   embeddings_matrix: np.ndarray, 
                   metric: str = "cosine",
                   threshold: Optional[float] = None) -> Tuple[Optional[int], Optional[str], float, bool]:
        """
        Performs ultra-fast vectorized 1:N face identification against database matrix.
        Returns (best_user_id, best_user_name, best_score, is_match).
        """
        if query_feat is None or len(user_ids) == 0 or embeddings_matrix.shape[0] == 0:
            return None, None, 0.0, False

        thresh = threshold if threshold is not None else (self.cosine_threshold if metric == "cosine" else self.l2_threshold)
        q = query_feat.flatten().astype(np.float32)

        if metric.lower() == "cosine":
            scores = np.dot(embeddings_matrix, q)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            is_match = best_score >= thresh
        else:
            diffs = embeddings_matrix - q
            distances = np.linalg.norm(diffs, axis=1)
            best_idx = int(np.argmin(distances))
            best_score = float(distances[best_idx])
            is_match = best_score <= thresh

        if is_match:
            return user_ids[best_idx], user_names[best_idx], best_score, True
        else:
            return None, None, best_score, False
