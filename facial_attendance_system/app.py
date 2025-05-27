from flask import Flask, render_template, request, redirect
import os
import base64

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTER_FOLDER = os.path.join(BASE_DIR, 'register')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_name = request.form['name']
        image_data = request.form['captured_image']
        
        # Extract base64 and decode
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Save as JPEG in the register folder
        file_path = os.path.join(REGISTER_FOLDER, f"{student_name}.jpg")
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        return f"✅ {student_name} registered successfully!"
    
    return render_template('register.html')
