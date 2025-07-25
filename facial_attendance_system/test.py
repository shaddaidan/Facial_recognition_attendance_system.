from flask import Flask, render_template, request
import os
import base64
import time
import pickle
import face_recognition
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# === File and Folder Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTER_FOLDER = os.path.join(BASE_DIR, 'register')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'attendance_uploads')
ATTENDANCE_LOG = os.path.join(BASE_DIR, 'data', 'attendance.csv')

# === Ensure folders exist ===
os.makedirs(REGISTER_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

# === Image Cleaning Function ===
def clean_image(image_path):
    try:
        image = Image.open(image_path)
        print(f"🖼 Original image mode: {image.mode}, size: {image.size}")

        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Fix EXIF orientation
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
        except Exception as e:
            print("⚠️ No EXIF or failed to rotate:", e)

        # Resize
        image.thumbnail((1024, 1024))

        # Force overwrite
        image.save(image_path, format='JPEG')
        print(f"✅ Cleaned image saved: {image_path}")
    except Exception as e:
        print(f"❌ Error in clean_image: {e}")


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form['student_id']
        photo = request.files['photo']

        student_folder = os.path.join(REGISTER_FOLDER, student_id)
        os.makedirs(student_folder, exist_ok=True)

        # Save photo temporarily
        temp_path = os.path.join(student_folder, f"{student_id}_temp.jpg")
        photo.save(temp_path)

        try:
            # Clean and save final image
            final_image_path = os.path.join(student_folder, f"{student_id}.jpeg")
            clean_image(temp_path)
            os.rename(temp_path, final_image_path)
        except Exception as e:
            return f"❌ Error processing image: {e}"

        # === Face encoding ===
        try:
            image = face_recognition.load_image_file(final_image_path)
            print(f"🧠 [REGISTER] Image shape for encoding: {image.shape}")

            face_locations = face_recognition.face_locations(image)
            print(f"🔍 [REGISTER] Face locations: {face_locations}")

            face_encodings = face_recognition.face_encodings(image, face_locations)

            if not face_encodings:
                return f"⚠️ No face found in the image. Try a clearer front-facing photo."

            encoding = face_encodings[0]
            encoding_path = os.path.join(student_folder, "face_encoding.pkl")
            with open(encoding_path, 'wb') as f:
                pickle.dump(encoding, f)

            return f"✅ {student_id} registered successfully!"
        except Exception as e:
            return f"❌ Error encoding face: {e}"

    return render_template('register.html')

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        student_id = request.form['student_id']
        photo = request.files['photo']

        # === Save with timestamped name ===
        filename = photo.filename or "attendance.jpg"
        unique_filename = f"{int(time.time())}_{filename}"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        photo.save(image_path)

        # === Clean the uploaded image ===
        clean_image(image_path)

        # === Load cleaned image and try to detect face ===
        try:
            unknown_image = face_recognition.load_image_file(image_path)
            print(f"🧠 [ATTENDANCE] Image shape: {unknown_image.shape}")

            face_locations = face_recognition.face_locations(unknown_image)
            print(f"🔍 [ATTENDANCE] Face locations: {face_locations}")

            face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        except Exception as e:
            return f"❌ Error processing image: {e}"

        if not face_encodings:
            return "⚠️ No face detected. Please try again with a clearer photo."

        unknown_encoding = face_encodings[0]

        # === Locate student’s saved encoding ===
        encoding_path = os.path.join(REGISTER_FOLDER, student_id, "face_encoding.pkl")
        if not os.path.exists(encoding_path):
            return f"❌ No registration found for student ID: {student_id}"

        with open(encoding_path, 'rb') as f:
            known_encoding = pickle.load(f)

        match = face_recognition.compare_faces([known_encoding], unknown_encoding)[0]
        print(f"🧪 [ATTENDANCE] Match result for {student_id}: {match}")

        if match:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log = pd.DataFrame([{
                'Name': student_id,
                'Timestamp': timestamp,
                'Source': 'MobileCamera',
                'Image': unique_filename
            }])

            if os.path.exists(ATTENDANCE_LOG):
                log.to_csv(ATTENDANCE_LOG, mode='a', header=False, index=False)
            else:
                log.to_csv(ATTENDANCE_LOG, index=False)

            return f"✅ Attendance marked for {student_id}"

        return "❌ Face did not match the registered image. Please try again."

    return render_template('attendance.html')


if __name__ == '__main__':
    print("✅ Flask is starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
