from flask import Flask, render_template, request
import os
import time
import pickle
import json
import face_recognition
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTER_FOLDER = os.path.join(BASE_DIR, 'register')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'attendance_uploads')
ATTENDANCE_LOG = os.path.join(BASE_DIR, 'data', 'attendance.csv')

os.makedirs(REGISTER_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

def clean_image(image_path):
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        try:
            for orientation in ExifTags.TAGS:
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()
            if exif:
                orientation_value = exif.get(orientation, None)
                if orientation_value == 3:
                    image = image.rotate(180, expand=True)
                elif orientation_value == 6:
                    image = image.rotate(270, expand=True)
                elif orientation_value == 8:
                    image = image.rotate(90, expand=True)
        except:
            pass

        image.thumbnail((1024, 1024))
        image.save(image_path, format='JPEG')
    except Exception as e:
        print(f"❌ Error in clean_image: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form['student_id'].strip()
        student_name = request.form['student_name'].strip()
        photo = request.files['photo']

        if not student_id or not student_name:
            return "❌ Both ID and Name are required."

        # Explicitly disallow slashes in ID
        if '/' in student_id or '\\' in student_id:
            return "❌ Invalid student ID format. Please avoid using '/' or '\\'."

        try:
            student_folder = os.path.join(REGISTER_FOLDER, student_id)
            os.makedirs(student_folder, exist_ok=True)

            temp_path = os.path.join(student_folder, f"{student_id}_temp.jpg")
            photo.save(temp_path)

            final_image_path = os.path.join(student_folder, f"{student_id}.jpeg")
            clean_image(temp_path)
            os.rename(temp_path, final_image_path)

            image = face_recognition.load_image_file(final_image_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if not face_encodings:
                os.remove(final_image_path)
                return "⚠️ No face found. Try again with a clearer image."

            encoding = face_encodings[0]
            with open(os.path.join(student_folder, "face_encoding.pkl"), 'wb') as f:
                pickle.dump(encoding, f)

            meta_data = {"name": student_name}
            with open(os.path.join(student_folder, "meta.json"), 'w') as f:
                json.dump(meta_data, f)

            return f"✅ Registered {student_name} ({student_id}) successfully!"

        except Exception as e:
            return f"❌ Registration failed: {e}"

    return render_template('register.html')



@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        student_id_input = request.form['student_id'].strip()
        photo = request.files['photo']

        filename = photo.filename or "attendance.jpg"
        unique_filename = f"{int(time.time())}_{student_id_input}_{filename}"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        photo.save(image_path)

        clean_image(image_path)

        try:
            unknown_image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(unknown_image)
            face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        except Exception as e:
            return f"❌ Error processing image: {e}"

        if not face_encodings:
            return "⚠️ No face detected. Please try again."

        unknown_encoding = face_encodings[0]
        known_encodings = []
        known_ids = []
        known_names = []

        for student_folder in os.listdir(REGISTER_FOLDER):
            folder_path = os.path.join(REGISTER_FOLDER, student_folder)
            encoding_path = os.path.join(folder_path, "face_encoding.pkl")
            meta_path = os.path.join(folder_path, "meta.json")
            if os.path.exists(encoding_path) and os.path.exists(meta_path):
                with open(encoding_path, 'rb') as f:
                    known_encodings.append(pickle.load(f))
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    known_names.append(meta.get("name", "Unknown"))
                known_ids.append(student_folder)

        distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        min_distance = min(distances)
        threshold = 0.45

        if min_distance < threshold:
            best_match_index = distances.tolist().index(min_distance)
            matched_id = known_ids[best_match_index]
            matched_name = known_names[best_match_index]
        else:
            return "❌ Face not recognized. Please try again."

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log = pd.DataFrame([{
            'Student ID': matched_id,
            'Name': matched_name,
            'Timestamp': timestamp,
            'Source': 'MobileCamera',
            'Image': unique_filename
        }])

        if os.path.exists(ATTENDANCE_LOG):
            log.to_csv(ATTENDANCE_LOG, mode='a', header=False, index=False)
        else:
            log.to_csv(ATTENDANCE_LOG, index=False)

        return f"✅ Attendance marked for {matched_name} ({matched_id})"

    return render_template('attendance.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
