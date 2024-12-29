# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import cvzone
import os
import base64
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Load face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Configuration
SHIRT_FOLDER = "shirts"
if not os.path.exists(SHIRT_FOLDER):
    os.makedirs(SHIRT_FOLDER)

class VirtualTryOn:
    def __init__(self):
        self.current_shirt_index = 0
        self.shirts = self.load_shirts()
        
    def load_shirts(self):
        shirts = []
        for file in os.listdir(SHIRT_FOLDER):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shirt_path = os.path.join(SHIRT_FOLDER, file)
                shirts.append(shirt_path)
        return shirts
    
    def get_current_shirt(self):
        if not self.shirts:
            return None
        return self.shirts[self.current_shirt_index]
    
    def next_shirt(self):
        if self.shirts:
            self.current_shirt_index = (self.current_shirt_index + 1) % len(self.shirts)
            
    def previous_shirt(self):
        if self.shirts:
            self.current_shirt_index = (self.current_shirt_index - 1) % len(self.shirts)

# Initialize virtual try-on
tryon = VirtualTryOn()

@app.route('/')
def index():
    return render_template('index.html')

def apply_shirt(frame, shirt_path, body):
    try:
        x, y, w, h = body
        
        # Read and resize shirt
        shirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
        if shirt is None:
            return frame
            
        # Calculate shirt dimensions while maintaining aspect ratio
        shirt_h = int(h * 1.2)  # Make shirt slightly larger than body
        aspect_ratio = shirt.shape[1] / shirt.shape[0]
        shirt_w = int(shirt_h * aspect_ratio)
        
        # Resize shirt
        shirt = cv2.resize(shirt, (shirt_w, shirt_h))
        
        # Calculate position to center shirt on body
        shirt_x = max(x - (shirt_w - w) // 2, 0)
        shirt_y = max(y - (shirt_h - h) // 4, 0)  # Adjust vertical position
        
        # Apply shirt overlay
        frame = cvzone.overlayPNG(frame, shirt, [shirt_x, shirt_y])
        
    except Exception as e:
        print(f"Error applying shirt: {e}")
    
    return frame

@socketio.on('request_frame')
def handle_frame_request():
    success, frame = cap.read()
    if not success:
        return
    
    # Mirror the frame horizontally for a more natural view
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect body
    bodies = body_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    
    # Apply shirt if body detected
    current_shirt = tryon.get_current_shirt()
    if len(bodies) > 0 and current_shirt:
        frame = apply_shirt(frame, current_shirt, bodies[0])
    
    # Convert frame to base64 for sending to client
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    emit('frame', {'image': frame_base64})

@socketio.on('change_shirt')
def handle_shirt_change(data):
    direction = data.get('direction', 'next')
    if direction == 'next':
        tryon.next_shirt()
    else:
        tryon.previous_shirt()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True)