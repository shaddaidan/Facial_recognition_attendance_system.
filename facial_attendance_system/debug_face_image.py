import face_recognition
from PIL import Image, ExifTags
import os
import pickle
import sys

# ====== CONFIG ======
student_id = "TEST001"  # Replace with input()
input_image_path = "/Users/shaddaiadeniran/Documents/python projects/NCAIR /DATA SCIENCE/FINAL_PROJECT/facial_attendance_system/N2.jpeg"  # Replace with your image path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTER_DIR = os.path.join(BASE_DIR, 'register')
student_folder = os.path.join(REGISTER_DIR, student_id)
os.makedirs(student_folder, exist_ok=True)

# ====== FUNCTIONS ======

def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(image._getexif().items())
        orientation_value = exif.get(orientation, None)

        if orientation_value == 3:
            image = image.rotate(180, expand=True)
        elif orientation_value == 6:
            image = image.rotate(270, expand=True)
        elif orientation_value == 8:
            image = image.rotate(90, expand=True)

    except Exception as e:
        print("⚠️ No EXIF orientation data found or error:", e)

    return image


# ====== LOAD IMAGE ======

try:
    pil_image = Image.open(input_image_path).convert('RGB')
    pil_image = correct_image_orientation(pil_image)

    # Resize to a manageable size to aid detection
    pil_image.thumbnail((1024, 1024))
except Exception as e:
    print(f"❌ Failed to open image: {e}")
    sys.exit(1)

# Save the corrected and resized image
final_image_path = os.path.join(student_folder, f"{student_id}.jpeg")
pil_image.save(final_image_path, format='JPEG')
print(f"✅ Saved corrected image to {final_image_path}")

# ====== FACE ENCODING ======

image_np = face_recognition.load_image_file(final_image_path)
face_locations = face_recognition.face_locations(image_np, model='cnn')
face_encodings = face_recognition.face_encodings(image_np, face_locations)

if not face_encodings:
    print("❌ No face detected. Try again with a better image.")
    sys.exit(1)

# Save encoding
encoding = face_encodings[0]
encoding_path = os.path.join(student_folder, "face_encoding.pkl")

with open(encoding_path, 'wb') as f:
    pickle.dump(encoding, f)

print(f"✅ Face encoding saved to {encoding_path}")
