import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

output_image = np.zeros((640,480,3), np.uint8)
black_image = np.zeros((400,400,3), np.uint8)

def displayHandOnFrame(results, frame, timestamp):

    image_copy = np.copy(frame.numpy_view())
    black_image_new = np.zeros((400,400,3), np.uint8)

    if not results.hand_landmarks:
        print("No hands detected.")
    else:
        print(len(results.hand_landmarks))
    
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

        mp_drawing.draw_landmarks(
            black_image_new,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

    global output_image
    output_image = image_copy
    global black_image
    black_image = black_image_new

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=displayHandOnFrame,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp_ms = int((time.time() - start_time) * 1000)

    landmarker.detect_async(mp_image, timestamp_ms)

    cv2.imshow("hand landmarker", output_image)
    cv2.imshow("black landmarker", black_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()