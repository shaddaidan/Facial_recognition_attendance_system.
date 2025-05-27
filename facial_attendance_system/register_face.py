# register_face.py

import face_recognition
import os
import pickle

# === Base path setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === File and folder paths ===
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'register')
ENCODING_FILE = os.path.join(BASE_DIR, 'data', 'face_encodings.pkl')

# === Create encoding dictionary ===
known_encodings = {}

# === Loop through all images in register/ ===
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            student_name = os.path.splitext(filename)[0]
            known_encodings[student_name] = encodings[0]
            print(f"✅ Registered: {student_name}")
        else:
            print(f"⚠️ No face found in: {filename}")

# === Save to pickle file ===
with open(ENCODING_FILE, 'wb') as f:
    pickle.dump(known_encodings, f)

print(f"✅ All encodings saved to {ENCODING_FILE}")
