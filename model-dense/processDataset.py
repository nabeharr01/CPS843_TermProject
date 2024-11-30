# This file runs preprocessing on an image dataset of hands representing ASL alphabet letters.
# First, MediaPipe Hands detects a hand from an image, then augmentations are applied
# (Scaling, rotating, noise, and binding to 400x400 xy coordinate space).
# The augmented MediaPipe Hand landmarks are appended to a CSV file with their label.
# 
# If you are not training a Dense neural network, no need to run this file and obtain a CSV dataset.

import os
import cv2
import csv
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Download hand_landmarker.task from https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
# Place it in this directory
landmarker_model_path = "hand_landmarker.task"
# Directory of image dataset to process
dataset_directory = "./asl_dataset"

# Scale landmarks to fit within an image_width x image_height image
def bind_landmarks_to_image_size(hand_landmarks, image_width, image_height):
    x_min = min([lm[0] for lm in hand_landmarks]) # smallest x coord
    x_max = max([lm[0] for lm in hand_landmarks]) # largest x coord
    y_min = min([lm[1] for lm in hand_landmarks]) # smallest y coord
    y_max = max([lm[1] for lm in hand_landmarks]) # largest y coord

    box_width, box_height = x_max - x_min, y_max - y_min
    # Scale down to 90% of smaller dimension, for padding
    scale_factor = min(image_width, image_height) * 0.9 / max(box_width, box_height)

    # For each unscaled landmark, scale and fit, then append to bound_landmarks
    bound_landmarks = []
    for lm in hand_landmarks:
        x = (lm[0] - x_min) * scale_factor
        y = (lm[1] - y_min) * scale_factor

        x += (image_width - (box_width * scale_factor)) / 2
        y += (image_height - (box_height * scale_factor)) / 2
        bound_landmarks.append([x/image_width, y/image_height, lm[2]])

    return bound_landmarks

# Compute centre coordinate (centroid) of landmarks by taking the mean
def compute_centroid(landmarks):
    return np.mean(landmarks, axis=0)

# Convert landmarks as objects [(x, y, z)] to array format [[x, y, z]]
def landmarks_to_array(landmark_objects):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark_objects])

# Rotate a landmark set in three axes around its own centroid
def rotate_around_center(landmarks, angle_x=0, angle_y=0, angle_z=0):
    centroid = compute_centroid(landmarks) # get centre coordinate
    
    centered_landmarks = landmarks - centroid # centre around origin
    
    # x-axis rotation matrix
    rotation_x = np.array([
        [1, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [0, math.sin(angle_x), math.cos(angle_x)]
    ])
    
    # y-axis rotation matrix
    rotation_y = np.array([
        [math.cos(angle_y), 0, math.sin(angle_y)],
        [0, 1, 0],
        [-math.sin(angle_y), 0, math.cos(angle_y)]
    ])
    
    # z-axis rotation matrix
    rotation_z = np.array([
        [math.cos(angle_z), -math.sin(angle_z), 0],
        [math.sin(angle_z), math.cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    # transform landmarks with all three axis rotations
    rotated_landmarks = np.dot(centered_landmarks, rotation_x.T)
    rotated_landmarks = np.dot(rotated_landmarks, rotation_y.T)
    rotated_landmarks = np.dot(rotated_landmarks, rotation_z.T)
    
    rotated_landmarks += centroid # place back at original centroid
    return rotated_landmarks

# Scale a landmark by [x, y, z] scale factors
def scale(landmarks, scale_factors):
    return landmarks * scale_factors

# Add noise to each landmark
def add_noise(landmarks, noise_level=0.02):
    noise = np.random.normal(0, noise_level, landmarks.shape)
    return landmarks + noise

# Combine all augmentations to augment a landmark
# Returns a list of 250 augmentations, flattened for csv append
def augment_landmarks(landmark_objects, label):
    augmented_data = [] # buffer to hold each augmented landmark set

    landmarks = landmarks_to_array(landmark_objects)

    # Set bounds + steps for rotations and scale
    rotation_angles_x = np.linspace(-0.6, 0.6, 5)
    rotation_angles_y = np.linspace(-0.6, 0.6, 5)
    rotation_angles_z = np.linspace(-0.2, 0.2, 5)
    mirror_x_options = [-1, 1] # covers left-handedness
    
    # 2 * 5 * 5 * 5 = 250 augments for each instance
    for mirror_x in mirror_x_options:
        for angle_x in rotation_angles_x:
            for angle_y in rotation_angles_y:
                for angle_z in rotation_angles_z:
                    transformed_landmarks = rotate_around_center(landmarks, angle_x, angle_y, angle_z)
                    transformed_landmarks = scale(transformed_landmarks, np.array([mirror_x, 1, 1]))
                    transformed_landmarks = add_noise(transformed_landmarks, noise_level=0.005)

                    centered_landmarks = bind_landmarks_to_image_size(transformed_landmarks, 400, 400)

                    # Convert to NormalizedLandmarkList as part of normalization
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in centered_landmarks
                    ])

                    # Flatten [(x1,y1,z1), (x2,y2,z2), ...] into [x1,y1,z1,x2,y2,z2,...]
                    flattened_proto = [value for landmark in hand_landmarks_proto.landmark for value in (landmark.x, landmark.y, landmark.z)]

                    augmented_data.append(flattened_proto + [label]) # append augmented landmarks and the label to augmented list for csv append
    
    return augmented_data

# Set up MediaPipe Task API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

# Set up Mediapipe Hand Landmarker to detect 1 hand from an IMAGE
landmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=landmarker_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(landmarker_options)

# Open/create a CSV file for writing. Walk through all images in the dataset, run MediaPipe hands
# to extract a landmark set, then augment that landmark set into 250 more instances.
# Add these augmented instances, along with their labels, to the CSV file.
with open('mp-landmarks-to-asl.csv', newline='', mode='w+') as data:
    writer = csv.writer(data, quotechar='"', quoting=csv.QUOTE_STRINGS)
    # 21 hand landmark coordinates from MediaPipe, each with x,y,z, means 63 features in total
    writer.writerow([f"{axis}{idx}" for idx in range(1,22) for axis in ["x", "y", "z"]] + ["label"])
    for path, folders, files in os.walk(dataset_directory):
        for file in files:
            print(file) # output file name for transparency

            # Process image with opencv2
            # Convert colour coding to RGB and size to 400x400 for MediaPipe hand detection
            image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
            image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0,0,0))
            image = cv2.resize(image, (400,400))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            results = landmarker.detect(mp_frame) # detect the hand with the landmark detector

            # if no hand was detected from the image, skip image
            if (len(results.hand_landmarks) == 0):
                continue

            # Call the augmentation function and save results
            augmented_landmarks = augment_landmarks(
                results.hand_landmarks[0], 
                path.split('\\' if os.name == 'nt' else '/')[-1].upper()
            )

            # For each augmented landmark, add it to the CSV file
            for augmented_landmark in augmented_landmarks:
                writer.writerow(augmented_landmark)
