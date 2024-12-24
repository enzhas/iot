import datetime
import shutil
import sys
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
import cv2
import threading
import numpy as np
import os
import face_recognition
import subprocess
import time

app = Flask(__name__)

# Global variables to store the frame and recognition result
latest_frame = None
colored_frame = None
frame_lock = threading.Lock()
face_recognition_result = ""

def get_cam_ip():
    cmd = 'arp -a | grep "d0:ef:76:ef:67:4"'
    returned_output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    parse = str(returned_output).split(' ', 1)
    ip = parse[1].split(' ')
    cam = ip[0][1:-1]
    print(cam)
    return cam

# Path to your folder with labeled photos
FOLDER_PATH = "./known_faces"

# ESP32 camera stream URL
cam_ip = get_cam_ip()
ESP32_STREAM_URL = f"http://{cam_ip}:81/stream"

last_save_time = 0  # Global variable to track the last save time

def save_detected_face(face_image, name):
    """Save the detected face with a timestamp at a 5-second interval."""
    global last_save_time
    current_time = time.time()

    # Check if 5 seconds have passed since the last save
    if current_time - last_save_time >= 2:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{name}_{timestamp}.jpg"
        save_path = os.path.join("./detected_faces", file_name)
        cv2.imwrite(save_path, face_image)
        print(f"Saved: {file_name}")
        last_save_time = current_time  

# Prepare training data
def prepare_training_data(folder_path):
    known_encodings = []
    known_names = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            label = os.path.splitext(file_name)[0]
            img_path = os.path.join(folder_path, file_name)

            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(label)

    return known_encodings, known_names

known_encodings, known_names = prepare_training_data(FOLDER_PATH)

# Open ESP32 video stream
cap = cv2.VideoCapture(ESP32_STREAM_URL)

def capture_frames():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from the stream. Retrying...")
            cap.open(ESP32_STREAM_URL)
            continue

        with frame_lock:
            latest_frame = frame

def perform_face_recognition():
    global latest_frame, face_recognition_result, colored_frame
    os.makedirs("./detected_faces", exist_ok=True) 

    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_recognition_result = "No faces detected"  # Default message
        name = "Unknown"
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            # name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            face_image = frame[top:bottom, left:right]
            if face_image.size > 0:  # Ensure face image is valid
                save_detected_face(face_image, name)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        colored_frame = frame

        # Update the global face recognition result
        face_recognition_result = f"Detected: {name}"

def generate_frames():
    global colored_frame
    while True:
        with frame_lock:
            if colored_frame is None:
                continue
            frame = colored_frame

        # _, buffer = cv2.imencode('.jpg', frame)
        # frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition_result')
def recognition_result():
    global face_recognition_result
    return f"Current Recognition: {face_recognition_result}"


@app.route('/static/detected_faces/<path:filename>')
def static_faces(filename):
    return send_from_directory('./detected_faces', filename)

@app.route('/detected_faces')
def detected_faces():
    detected_faces_path = "./detected_faces"
    faces = [f for f in os.listdir(detected_faces_path) if os.path.isfile(os.path.join(detected_faces_path, f))]
    return jsonify(faces)

@app.route('/save_face', methods=['POST'])
def save_face():
    global  known_encodings, known_names
    # Check if there is a face image in the detected_faces folder
    detected_faces_path = "./detected_faces"
    detected_faces = [f for f in os.listdir(detected_faces_path) if os.path.isfile(os.path.join(detected_faces_path, f))]

    if not detected_faces:
        return "No face detected", 400  # Return error if no faces are detected

    latest_face = detected_faces[-1]  # Get the most recent face image
    source_path = os.path.join(detected_faces_path, latest_face)

    # Move the face image to known_faces folder (ensure it's unique)
    known_faces_path = "./known_faces"
    if not os.path.exists(known_faces_path):
        os.makedirs(known_faces_path)

    # Save with a new name (you can use any naming convention, here using current timestamp)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target_path = os.path.join(known_faces_path, f"{timestamp}_{latest_face}")

    shutil.copy(source_path, target_path)
    # known_encodings, known_names = prepare_training_data(FOLDER_PATH)
    # raise RuntimeError('Not running with the Werkzeug Server')
    os._exit(0)

    return f"Face saved as {target_path}", 200
   

if __name__ == "__main__":
    # Start background threads
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    
    recognition_thread = threading.Thread(target=perform_face_recognition, daemon=True)
    recognition_thread.start()

    app.run(host='0.0.0.0', port=5001)
