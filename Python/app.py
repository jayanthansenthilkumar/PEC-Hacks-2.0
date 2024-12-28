import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load the dress image (ensure it's a PNG with transparency)
dress = cv2.imread("dress.png", cv2.IMREAD_UNCHANGED)  # Replace with your dress image path

# Function to overlay transparent image
def overlay_transparent(background, overlay, x, y, width=None, height=None):
    if width and height:
        overlay = cv2.resize(overlay, (width, height))

    h, w, _ = overlay.shape
    for i in range(h):
        for j in range(w):
            if y + i >= background.shape[0] or x + j >= background.shape[1]:
                continue
            alpha = overlay[i, j, 3] / 255.0
            background[y + i, x + j] = (1 - alpha) * background[y + i, x + j] + alpha * overlay[i, j, :3]
    return background

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # If landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key points for dress alignment
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate dress width and height based on body key points
        dress_width = int(abs(right_shoulder.x - left_shoulder.x) * frame.shape[1] * 1.5)
        dress_height = int(abs(right_hip.y - left_shoulder.y) * frame.shape[0] * 1.2)

        # Calculate position to place the dress
        x = int(left_shoulder.x * frame.shape[1] - dress_width // 4)
        y = int(left_shoulder.y * frame.shape[0])

        # Overlay the dress on the frame
        frame = overlay_transparent(frame, dress, x, y, width=dress_width, height=dress_height)

    # Display the result
    cv2.imshow("Virtual Dress Fitting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
