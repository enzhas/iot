from flask import Flask, Response, render_template
import cv2
import threading
import numpy as np
import os
import face_recognition
import subprocess

app = Flask(__name__)

# Global variables to store the frame and recognition result
latest_frame = None
frame_lock = threading.Lock()
face_recognition_result = ""

# Function to get camera IP
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

# Capture frames from the ESP32 stream
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

# Start background thread to capture frames
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Perform face recognition on captured frames
def perform_face_recognition():
    global latest_frame, face_recognition_result
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        name = "Unknown"
        face_recognition_result = "No faces detected"  # Default message

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Update recognition result
        face_recognition_result = f"Detected: {name}"

        # Convert frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Video feed endpoint
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Recognition result endpoint
@app.route('/recognition_result')
def recognition_result():
    global face_recognition_result
    return f"Current Recognition: {face_recognition_result}"

# Main route
@app.route('/')
def index():
    return render_template('index.html')

# Generate frames for video feed
def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # Perform face recognition and frame encoding
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
