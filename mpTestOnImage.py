# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("vignonantoine/mediapipe-processed-asl-dataset")
# print("Path to dataset files:", path)

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

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

black_image = np.zeros((400,400,3), np.uint8)
img = cv2.imread("./processed_combine_asl_dataset/0/hand1_0_bot_seg_1_cropped.jpeg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
results = landmarker.detect(mp_image)

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
        black_image,
        hand_landmarks_proto,
        mp_hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style()
    )

cv2.imshow("original",img)

cv2.imshow("new", black_image)

cv2.waitKey(0)
cv2.destroyAllWindows()