import os
import sys
import numpy as np
from datetime import datetime, timedelta
import random

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db.database import Database
from db.repository import AttendanceRepository

def seed_demo_records():
    print("[INFO] Seeding demo users and attendance records...")
    db = Database()
    repo = AttendanceRepository(db=db)

    sample_users = [
        ("EMP101", "Alex Morgan", "alex.m@example.com", "Computer Science", "Faculty"),
        ("EMP102", "Sarah Connor", "sarah.c@example.com", "Engineering", "Employee"),
        ("EMP103", "David Miller", "david.m@example.com", "Human Resources", "Staff"),
        ("EMP104", "Elena Rostova", "elena.r@example.com", "Computer Science", "Student"),
        ("EMP105", "Marcus Vance", "marcus.v@example.com", "Finance", "Employee"),
        ("EMP106", "Priya Sharma", "priya.s@example.com", "Marketing", "Employee"),
    ]

    created_user_ids = []
    for code, name, email, dept, role in sample_users:
        existing = repo.get_user_by_code(code)
        if not existing:
            # Generate random unit vector for embedding
            emb = np.random.randn(128).astype(np.float32)
            emb /= np.linalg.norm(emb)
            uid = repo.add_user(
                user_code=code,
                name=name,
                email=email,
                department=dept,
                role=role,
                embedding=emb
            )
            created_user_ids.append(uid)
            print(f"  [+] Enrolled user: {name} ({code})")
        else:
            created_user_ids.append(existing["id"])

    # Seed past 7 days of attendance
    today = datetime.now()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        for day_offset in range(7, -1, -1):
            date_obj = today - timedelta(days=day_offset)
            date_str = date_obj.strftime("%Y-%m-%d")

            for uid in created_user_ids:
                # 85% chance of showing up
                if random.random() < 0.85:
                    # Random arrival time between 08:45 and 09:50
                    hour = random.choice([8, 9])
                    minute = random.randint(30, 59) if hour == 8 else random.randint(0, 50)
                    time_str = f"{hour:02d}:{minute:02d}:{random.randint(0, 59):02d}"

                    status = "Present" if (hour == 8 or (hour == 9 and minute <= 30)) else "Late"
                    confidence = round(random.uniform(0.75, 0.98), 2)

                    cursor.execute("""
                        INSERT OR IGNORE INTO attendance (user_id, date, time, status, confidence, device)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (uid, date_str, time_str, status, confidence, "Webcam-Gate-1"))

    print("[OK] Demo seed complete! You can now view populated charts in the 'Analytics & Reports' tab.")

if __name__ == "__main__":
    seed_demo_records()
