from flask import Flask, render_template, request
import os
import base64
import time
import pickle
import face_recognition
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# === File and Folder Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTER_FOLDER = os.path.join(BASE_DIR, 'register')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'attendance_uploads')
ENCODING_FILE = os.path.join(BASE_DIR, 'data', 'face_encodings.pkl')
ATTENDANCE_LOG = os.path.join(BASE_DIR, 'data', 'attendance.csv')

# === Ensure folders exist ===
os.makedirs(REGISTER_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        photo = request.files['photo']
        filename = f"{name}.jpg"
        filepath = os.path.join(REGISTER_FOLDER, filename)
        photo.save(filepath)

        # === Load image and detect face with locations ===
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, face_locations)

        if not encodings:
            return "⚠️ No face detected in the image. Please try again with a clearer face photo."

        encoding = encodings[0]

        # === Load or create encoding database ===
        if os.path.exists(ENCODING_FILE):
            with open(ENCODING_FILE, 'rb') as f:
                known_encodings = pickle.load(f)
        else:
            known_encodings = {}

        known_encodings[name] = encoding

        # === Save updated encodings ===
        with open(ENCODING_FILE, 'wb') as f:
            pickle.dump(known_encodings, f)

        return f"✅ {name} registered and added to face database!"

    return render_template('register.html')



@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        photo = request.files['photo']
        filename = photo.filename or "attendance.jpg"
        unique_filename = f"{int(time.time())}_{filename}"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        photo.save(image_path)

        # === Check if encoding file exists ===
        if not os.path.exists(ENCODING_FILE):
            return "⚠️ Face database not found. Please register someone first."

        # === Load known encodings ===
        try:
            with open(ENCODING_FILE, 'rb') as f:
                known_encodings = pickle.load(f)
        except Exception as e:
            return f"❌ Failed to load encodings: {e}"

        known_names = list(known_encodings.keys())
        known_faces = list(known_encodings.values())

        # === Load uploaded image and detect faces ===
        try:
            unknown_image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(unknown_image)
            face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        except Exception as e:
            return f"❌ Error processing image: {e}"

        if not face_encodings:
            return "⚠️ No face detected. Please try again."

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # === Log attendance ===
                log = pd.DataFrame([{
                    'Name': name,
                    'Timestamp': timestamp,
                    'Source': 'MobileCamera',
                    'Image': unique_filename
                }])

                if os.path.exists(ATTENDANCE_LOG):
                    log.to_csv(ATTENDANCE_LOG, mode='a', header=False, index=False)
                else:
                    log.to_csv(ATTENDANCE_LOG, index=False)

                return f"✅ Attendance marked for {name}"

        return "❌ Face not recognized. Please try again."

    return render_template('attendance.html')

if __name__ == '__main__':
    print("✅ Flask is starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
