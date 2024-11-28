import cv2
import time
import numpy as np
import mediapipe as mp
from tensorflow import keras
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from predictor import Predictor
from collections import deque

# Mediapipe and TensorFlow setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Variables for drawing and predictions
output_image = np.zeros((640, 480, 3), np.uint8)
black_image = np.zeros((400, 400, 3), np.uint8)
landmarks = np.zeros((63,), np.float32)
current_text = ""
rolling_buffer = deque(maxlen=10)

# Function to process hand detection and draw on the frame
def displayHandOnFrame(results, frame, timestamp):
    global output_image, black_image, landmarks

    image_copy = np.copy(frame.numpy_view())
    black_image_new = np.zeros((400, 400, 3), np.uint8)
    landmarks_new = np.zeros((63,), np.float32)

    for hand_landmarks in results.hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        mp_drawing.draw_landmarks(
            image_copy,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Adjust bounding box and scale
        x_min = min([lm.x for lm in hand_landmarks])
        x_max = max([lm.x for lm in hand_landmarks])
        y_min = min([lm.y for lm in hand_landmarks])
        y_max = max([lm.y for lm in hand_landmarks])
        box_width, box_height = x_max - x_min, y_max - y_min
        scale_factor = 400 * 0.9 / max(box_width, box_height)

        translated_landmarks = []
        for lm in hand_landmarks:
            x = (lm.x - x_min) * scale_factor
            y = (lm.y - y_min) * scale_factor
            x += (400 - (box_width * scale_factor)) / 2
            y += (400 - (box_height * scale_factor)) / 2
            translated_landmarks.append(landmark_pb2.NormalizedLandmark(x=x / 400, y=y / 400, z=lm.z))

        landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=translated_landmarks)
        mp_drawing.draw_landmarks(
            black_image_new,
            landmark_list,
            mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

        landmarks_new = np.array([value for landmark in landmark_list.landmark for value in (landmark.x, landmark.y, landmark.z)])

    output_image = image_copy
    black_image = black_image_new
    landmarks = landmarks_new

# Set up hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=displayHandOnFrame,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = HandLandmarker.create_from_options(options)

# Load model and predictor
loaded_model = keras.models.load_model("mp-landmarks-to-asl-nn.keras")
predictor = Predictor(loaded_model, "landmark_label_classes.npy")

# Start video capture
cap = cv2.VideoCapture(0)
start_time = time.time()

# Main loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # Predict the current ASL letter
        predicted_asl = predictor.predict_landmarks(landmarks.reshape(1, -1))
        rolling_buffer.append(predicted_asl)
        most_frequent_letter = max(set(rolling_buffer), key=rolling_buffer.count)

        # Draw the GUI
        display_frame = output_image.copy()
        cv2.rectangle(display_frame, (10,60), (200, 25), (0,0,0), -1)
        cv2.putText(display_frame, f"Detected: {most_frequent_letter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(display_frame, (10,410), (200, 375), (0,0,0), -1)
        cv2.putText(display_frame, f"Text: {current_text}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(display_frame, (10,460), (500, 430), (0,0,0), -1)
        cv2.putText(display_frame, "Press SPACE to add letter, 'C' to clear, 'Q' to quit", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show the frame
        cv2.imshow("ASL Recognition", display_frame)

        # Handle keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Add letter to text box
            current_text += most_frequent_letter
        elif key == ord('c'):  # Clear text box
            current_text = ""

finally:
    cap.release()
    cv2.destroyAllWindows()