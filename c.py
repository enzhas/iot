from flask import Flask, render_template, jsonify, Response
import requests

app = Flask(__name__)

BACKEND_URL = "http://192.168.58.120:5001/"  # URL of the backend app

@app.route('/')
def index():
    """Render the frontend page."""
    return render_template('index.html')  # Create an HTML page for the frontend

@app.route('/video_feed')
def video_feed():
    """Fetch the video stream from the backend."""
    def generate():
        response = requests.get(f"{BACKEND_URL}/video_feed", stream=True)
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                yield chunk
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition_status')
def recognition_status():
    """Fetch and display recognition status from backend."""
    response = requests.get(f"{BACKEND_URL}/recognition_result")
    # Handling plain text response from backend
    return response.text

@app.route('/detected_faces')
def detected_faces():
    """Fetch and display the list of detected faces."""
    response = requests.get(f"{BACKEND_URL}/detected_faces")
    return jsonify(response.json())

@app.route('/save_face', methods=['POST'])
def save_face():
    """Save the detected face by forwarding the POST request to the backend."""
    response = requests.post(f"{BACKEND_URL}/save_face")
    return response.text  # Return the backend response to the frontend

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)  # Run frontend on a separate port
