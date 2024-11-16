import os
import cv2
import csv
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

def bind_landmarks_to_image_size(hand_landmarks, image_width, image_height):
    x_min = min([lm[0] for lm in hand_landmarks])
    x_max = max([lm[0] for lm in hand_landmarks])
    y_min = min([lm[1] for lm in hand_landmarks])
    y_max = max([lm[1] for lm in hand_landmarks])

    box_width, box_height = x_max - x_min, y_max - y_min
    scale_factor = min(image_width, image_height) * 0.9 / max(box_width, box_height)

    translated_landmarks = []
    for lm in hand_landmarks:
        x = (lm[0] - x_min) * scale_factor
        y = (lm[1] - y_min) * scale_factor

        x += (image_width - (box_width * scale_factor)) / 2
        y += (image_height - (box_height * scale_factor)) / 2
        translated_landmarks.append([x/image_width, y/image_height, lm[2]])

    return translated_landmarks

def compute_centroid(landmarks):
    return np.mean(landmarks, axis=0)

def landmarks_to_array(landmark_objects):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark_objects])

def rotate_around_center(landmarks, angle_x=0, angle_y=0, angle_z=0):
    centroid = compute_centroid(landmarks)
    
    centered_landmarks = landmarks - centroid
    
    rotation_x = np.array([
        [1, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [0, math.sin(angle_x), math.cos(angle_x)]
    ])
    
    rotation_y = np.array([
        [math.cos(angle_y), 0, math.sin(angle_y)],
        [0, 1, 0],
        [-math.sin(angle_y), 0, math.cos(angle_y)]
    ])
    
    rotation_z = np.array([
        [math.cos(angle_z), -math.sin(angle_z), 0],
        [math.sin(angle_z), math.cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    rotated_landmarks = np.dot(centered_landmarks, rotation_x.T)
    rotated_landmarks = np.dot(rotated_landmarks, rotation_y.T)
    rotated_landmarks = np.dot(rotated_landmarks, rotation_z.T)
    
    rotated_landmarks += centroid
    return rotated_landmarks

def scale(landmarks, scale_factors):
    return landmarks * scale_factors

def add_noise(landmarks, noise_level=0.02):
    noise = np.random.normal(0, noise_level, landmarks.shape)
    return landmarks + noise

def augment_landmarks(landmark_objects, label):
    augmented_data = []

    landmarks = landmarks_to_array(landmark_objects)

    rotation_angles_x = np.linspace(-0.6, 0.6, 5)
    rotation_angles_y = np.linspace(-0.6, 0.6, 5)
    rotation_angles_z = np.linspace(-0.2, 0.2, 5)
    # scale_factors_x = np.linspace(0.8, 1.2, 3)
    # scale_factors_y = np.linspace(0.8, 1.2, 3)
    # scale_factors_z = np.linspace(0.8, 1.2, 3)
    mirror_x_options = [-1, 1]
    
    # 2 * 5 * 5 * 5 = 250 augments for each instance
    for mirror_x in mirror_x_options:
        for angle_x in rotation_angles_x:
            for angle_y in rotation_angles_y:
                for angle_z in rotation_angles_z:
                    transformed_landmarks = rotate_around_center(landmarks, angle_x, angle_y, angle_z)
                    transformed_landmarks = scale(transformed_landmarks, np.array([mirror_x, 1, 1]))
                    transformed_landmarks = add_noise(transformed_landmarks, noise_level=0.005)

                    centered_landmarks = bind_landmarks_to_image_size(transformed_landmarks, 400, 400)

                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in centered_landmarks
                    ])

                    flattened_proto = [value for landmark in hand_landmarks_proto.landmark for value in (landmark.x, landmark.y, landmark.z)]

                    augmented_data.append(flattened_proto + [label])
    
    return augmented_data

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

landmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(landmarker_options)

directory = "./asl_dataset"

with open('mp-landmarks-to-asl.csv', newline='', mode='w+') as data:
    writer = csv.writer(data, quotechar='"', quoting=csv.QUOTE_STRINGS)
    writer.writerow([f"{axis}{idx}" for idx in range(1,22) for axis in ["x", "y", "z"]] + ["label"])
    for path, folders, files in os.walk(directory):
        for file in files:
            print(file)

            image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
            image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0,0,0))
            image = cv2.resize(image, (400,400))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            results = landmarker.detect(mp_frame)

            # if no hand was detected from the image, skip image
            if (len(results.hand_landmarks) == 0):
                continue

            augmented_landmarks = augment_landmarks(
                results.hand_landmarks[0], 
                path.split('\\' if os.name == 'nt' else '/')[-1].upper()
            )

            for augmented_landmark in augmented_landmarks:
                writer.writerow(augmented_landmark)
