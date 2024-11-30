# This is the main program. Please only run this file if you wish to demo the project.
#
# This code is responsible for:
# 1. Showing the camera feed with the GUI
# 2. Capture a frame from the camera and detect hands with MediaPipe Hands
# 3. Feed captured data in image format or landmark format to a model
# 4. Display the output of the model on the GUI
# 5. Afford user-interactivity such as building/clearing a word, changing model

import cv2
import time
import numpy as np
import mediapipe as mp
from tensorflow import keras
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from predictor import Predictor
from collections import deque

# Set up MediaPipe task API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

# MediaPipe Hands drawing classes
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load models and predictors
# Dense Model
dense_model = keras.models.load_model("MP-to-ASL-dense-model.keras")
dense_predictor = Predictor(dense_model, "label_classes_dense.npy")
# CNN Model
cnn_model = keras.models.load_model("MP-to-ASL-CNN-model.keras")
cnn_predictor = Predictor(cnn_model, "label_classes_cnn.npy")

# Variables for drawing and predictions
output_image = np.zeros((640, 480, 3), np.uint8) # Output image for GUI rendering
black_image = np.zeros((400,400,3), np.uint8) # Hidden image for CNN input
landmarks = np.zeros((63,), np.float32) # Landmarks array for Dense NN input
current_text = "" # Text buffer for GUI
model_type = "dense" # Keep track of which model user is using
rolling_buffer = deque(maxlen=10) # Buffer for most common prediction in the last 10 frames

# Function to process hand detection and draw on the frame
def displayHandOnFrame(results, frame, timestamp):
    # Refer to variables established outside of this function
    global output_image, black_image, landmarks

    # New buffers that are copies of what will be written to
    image_copy = np.copy(frame.numpy_view())
    black_image_new = np.zeros((400,400,3), np.uint8)
    landmarks_flattened = np.zeros((63,), np.float32)

    # For each detected hand (always 1)...
    for hand_landmarks in results.hand_landmarks:
        # Normalized Landmarks for MediaPipe drawing utils to work
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        # Draw landmarks on webcam frame
        mp_drawing.draw_landmarks(
            image_copy,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Get bounding box and scale 
        x_min = min([lm.x for lm in hand_landmarks])
        x_max = max([lm.x for lm in hand_landmarks])
        y_min = min([lm.y for lm in hand_landmarks])
        y_max = max([lm.y for lm in hand_landmarks])
        box_width, box_height = x_max - x_min, y_max - y_min
        scale_factor = 400 * 0.9 / max(box_width, box_height)

        # Use bounding box + scale to bind landmarks within a 400x400 xy plane
        bound_landmarks = []
        for lm in hand_landmarks:
            x = (lm.x - x_min) * scale_factor
            y = (lm.y - y_min) * scale_factor
            x += (400 - (box_width * scale_factor)) / 2
            y += (400 - (box_height * scale_factor)) / 2
            bound_landmarks.append(landmark_pb2.NormalizedLandmark(x=x / 400, y=y / 400, z=lm.z))

        landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=bound_landmarks)

        # Draw bound landmarks on the 400x400 black image, for input to the CNN
        mp_drawing.draw_landmarks(
            black_image_new,
            landmark_list,
            mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )
        
        # Flatten landmarks, for input to the Dense NN
        landmarks_flattened = np.array([value for landmark in landmark_list.landmark for value in (landmark.x, landmark.y, landmark.z)])

    # Update global variables with the buffers generated in this function
    output_image = image_copy
    black_image = black_image_new
    landmarks = landmarks_flattened

# Set up hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM, # Set up landmarker for LIVE_STREAM (camera feed)
    result_callback=displayHandOnFrame, # Callback function asynchronously called after detecting hand
    num_hands=1, # Only ever detect 1 hand 
    min_hand_detection_confidence=0.5, # Recommended values
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = HandLandmarker.create_from_options(options)

# Start video capture
cap = cv2.VideoCapture(0)
start_time = time.time()

# Main loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame & timestamp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # Predict the current ASL letter
        predicted_asl = ''
        if model_type == 'dense':
            predicted_asl = dense_predictor.predict_landmarks(landmarks.reshape(1, -1))
        else:
            predicted_asl = cnn_predictor.predict_image_array(black_image)
        # Add predicted letter to rolling buffer, take most frequent letter in buffer to mitigate noise
        rolling_buffer.append(predicted_asl)
        most_frequent_letter = max(set(rolling_buffer), key=rolling_buffer.count)

        # Draw the GUI
        display_frame = output_image.copy()
        # Detected letter
        cv2.rectangle(display_frame, (10,60), (200, 25), (0,0,0), -1)
        cv2.putText(display_frame, f"Detected: {most_frequent_letter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Current text user is building
        cv2.rectangle(display_frame, (10,410), (200, 375), (0,0,0), -1)
        cv2.putText(display_frame, f"Text: {current_text}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Shows which model is in use, CNN or Dense
        cv2.rectangle(display_frame, (400, 60), (640, 25), (0,0,0), -1)
        cv2.putText(display_frame, f"Model: {model_type.upper()}", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,127,255), 2)

        # Instructions
        cv2.rectangle(display_frame, (10,450), (400, 420), (0,0,0), -1)
        cv2.putText(display_frame, "Press SPACE to add letter, 'C' to clear", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.rectangle(display_frame, (10,480), (400, 450), (0,0,0), -1)
        cv2.putText(display_frame, "'M' to change models, or 'Q' to quit", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show the frame
        cv2.imshow("ASL Recognition", display_frame)

        # Handle keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit with Q
            break
        elif key == ord(' '):  # Add letter to text box with SPACE
            current_text += most_frequent_letter
        elif key == ord('c'):  # Clear text box with C
            current_text = ""
        elif key == ord('m'): # Toggle models with M
            if model_type == "dense":
                model_type = "cnn"
            else:
                model_type = "dense"
        
        if cv2.getWindowProperty("ASL Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()