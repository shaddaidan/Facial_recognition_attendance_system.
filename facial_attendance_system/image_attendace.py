# image_attendance.py

import face_recognition
import os
import pickle
import pandas as pd
from datetime import datetime

# === Base path setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === File and folder paths ===
ENCODING_FILE = os.path.join(BASE_DIR, 'data', 'face_encodings.pkl')
ATTENDANCE_FILE = os.path.join(BASE_DIR, 'data', 'attendance.csv')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'attendance_uploads')

# === Load known encodings ===
with open(ENCODING_FILE, 'rb') as f:
    known_encodings = pickle.load(f)

known_names = list(known_encodings.keys())
known_faces = list(known_encodings.values())

# === Prepare attendance list ===
attendance = []

# === Loop through uploaded images ===
for filename in os.listdir(UPLOAD_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image = face_recognition.load_image_file(image_path)

        # Detect and encode faces in image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                attendance.append({'Name': name, 'Timestamp': time_now, 'Image': filename})
                print(f"✅ Marked Present: {name}")
            else:
                print("⚠️ Face not recognized in:", filename)

# === Save attendance to CSV ===
if attendance:
    df = pd.DataFrame(attendance)
    if os.path.exists(ATTENDANCE_FILE):
        df.to_csv(ATTENDANCE_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(ATTENDANCE_FILE, index=False)
    print(f"✅ Attendance saved to {ATTENDANCE_FILE}")
else:
    print("ℹ️ No recognized faces to log.")
