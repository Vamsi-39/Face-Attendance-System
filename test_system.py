import os
import sys
import unittest
import numpy as np
import cv2
from datetime import datetime, timedelta

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db.database import Database
from db.repository import AttendanceRepository
from engine.detector import YuNetFaceDetector
from engine.recognizer import SFaceRecognizer
from engine.liveness import LivenessDetector
from engine.pipeline import FaceAttendancePipeline

class TestAttendanceSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create an isolated temporary test database
        cls.test_db_path = os.path.join(PROJECT_ROOT, "data", "test_attendance.db")
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)
        cls.db = Database(db_path=cls.test_db_path)
        cls.repo = AttendanceRepository(db=cls.db)
        cls.pipeline = FaceAttendancePipeline(repo=cls.repo)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
            except Exception:
                pass

    def test_01_models_and_engine_initialization(self):
        """Test YuNet and SFace models are properly initialized."""
        self.assertIsNotNone(self.pipeline.detector)
        self.assertIsNotNone(self.pipeline.recognizer)
        self.assertIsNotNone(self.pipeline.liveness)
        print("[TEST 1 PASSED] AI Models & Deep Learning Engine initialized successfully.")

    def test_02_user_registration_and_db(self):
        """Test adding users and storing 128-D embeddings."""
        # Generate 2 distinct normalized mock embeddings
        emb1 = np.random.randn(128).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)

        emb2 = np.random.randn(128).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)

        user1_id = self.repo.add_user(
            user_code="TEST001",
            name="Alice Smith",
            email="alice@example.com",
            department="Computer Science",
            role="Student",
            embedding=emb1
        )
        self.assertGreater(user1_id, 0)

        user2_id = self.repo.add_user(
            user_code="TEST002",
            name="Bob Jones",
            email="bob@example.com",
            department="Engineering",
            role="Faculty",
            embedding=emb2
        )
        self.assertGreater(user2_id, 0)

        # Retrieve users
        user1 = self.repo.get_user_by_code("TEST001")
        self.assertIsNotNone(user1)
        self.assertEqual(user1["name"], "Alice Smith")

        # Verify embeddings retrieval
        user_ids, names, matrix = self.repo.get_registered_embeddings()
        self.assertEqual(len(user_ids), 2)
        self.assertEqual(matrix.shape, (2, 128))
        print("[TEST 2 PASSED] User registration & embedding storage verified.")

    def test_03_vectorized_matching(self):
        """Test 1:N cosine similarity matching accuracy."""
        self.pipeline.refresh_database_cache()
        
        # Query with exact embedding of Alice (with slight noise)
        _, _, matrix = self.repo.get_registered_embeddings()
        alice_emb = matrix[0]
        noisy_query = alice_emb + np.random.normal(0, 0.01, size=128).astype(np.float32)
        noisy_query /= np.linalg.norm(noisy_query)

        uid, name, score, is_match = self.pipeline.recognizer.match_best(
            query_feat=noisy_query,
            user_ids=self.pipeline.user_ids,
            user_names=self.pipeline.user_names,
            embeddings_matrix=self.pipeline.embeddings_matrix,
            metric="cosine"
        )
        self.assertTrue(is_match)
        self.assertEqual(name, "Alice Smith")
        self.assertGreater(score, 0.90)
        print(f"[TEST 3 PASSED] Vectorized 1:N matching identified '{name}' with score {score:.4f}.")

    def test_04_attendance_marking_and_cooldown(self):
        """Test attendance punch recording and duplicate prevention cooldown."""
        user1 = self.repo.get_user_by_code("TEST001")
        uid = user1["id"]

        # First punch -> Should succeed
        success1, msg1, rec1 = self.repo.mark_attendance(user_id=uid, confidence=0.95, device="TestUnit")
        self.assertTrue(success1)
        self.assertIn("Attendance marked", msg1)

        # Second immediate punch -> Should be blocked by cooldown
        success2, msg2, rec2 = self.repo.mark_attendance(user_id=uid, confidence=0.95, device="TestUnit")
        self.assertFalse(success2)
        self.assertIn("already recorded recently", msg2)
        print("[TEST 4 PASSED] Attendance recording & anti-duplicate cooldown verified.")

    def test_05_analytics_and_summary(self):
        """Test daily summary and record retrieval."""
        today = datetime.now().strftime("%Y-%m-%d")
        summary = self.repo.get_daily_summary(today)
        self.assertEqual(summary["total_users"], 2)
        self.assertEqual(summary["present_count"], 1)
        self.assertEqual(summary["absent_count"], 1)

        records = self.repo.get_attendance_records(start_date=today, end_date=today)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["name"], "Alice Smith")
        print("[TEST 5 PASSED] Daily summary metrics and analytics queries verified.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
