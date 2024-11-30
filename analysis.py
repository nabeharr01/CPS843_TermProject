
# This file is responsible for obtaining statistics from the two models
# The statistics include:
# Prediction Time
# Prediction Accuracy
# Confusion Matrix

import os
import cv2
import csv
import math
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow import keras
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from predictor import Predictor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# MediaPipe Hands drawing classes
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to transform MediaPipe Hand landmarker results to model inputs
def process_landmarks(results):
    black_image_new = np.zeros((400,400,3), np.uint8)
    landmarks_flattened = np.zeros((63,), np.float32)

    # For each detected hand (always 1)...
    for hand_landmarks in results.hand_landmarks:
        # Normalized Landmarks for MediaPipe drawing utils to work
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

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

    return black_image_new, landmarks_flattened # Return inputs to be used later

# Directory of the Hand Landmarker model
landmarker_model_path = "hand_landmarker.task"
# Directory of image dataset to process
dataset_directory = "./asl_alphabet_train"

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

# Load models and predictors
# Dense Model
dense_model = keras.models.load_model("MP-to-ASL-dense-model.keras")
dense_predictor = Predictor(dense_model, "label_classes_dense.npy")
# CNN Model
cnn_model = keras.models.load_model("MP-to-ASL-CNN-model.keras")
cnn_predictor = Predictor(cnn_model, "label_classes_cnn.npy")

# Store prediction times for each model
times_cnn = []
times_dense = []

# Store true labels and model outputs for confusion matrices
true_labels = []
output_cnn = []
output_dense = []

# Open/create a CSV file for writing. Walk through all images in the dataset, run MediaPipe hands
# to extract a landmark set, then augment that landmark set into 250 more instances.
# Add these augmented instances, along with their labels, to the CSV file.
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
        
        # Get inputs to models from landmarker results
        black_image, flattened_landmarks = process_landmarks(results)

        # Get true label from file path and append
        true_label = path.split('\\' if os.name == 'nt' else '/')[-1].upper()
        true_labels.append(true_label)

        # Calculate time difference and append CNN model prediction
        time1 = time.time()
        prediction_cnn = cnn_predictor.predict_image_array(black_image)
        time2 = time.time()
        times_cnn.append(time2 - time1)
        output_cnn.append(prediction_cnn)

        # Calculate time difference and append Dense model prediction
        time1 = time.time()
        prediction_dense = dense_predictor.predict_landmarks(flattened_landmarks.reshape(1, -1))
        time2 = time.time()
        times_dense.append(time2 - time1)
        output_dense.append(prediction_dense)

# Calculate and print average prediction times for each model
print(f"Avg. CNN Model Prediction Time: {np.mean(times_cnn)}")
print(f"Avg. Dense Model Prediction Time: {np.mean(times_dense)}")

# ASCII A-Z from integer range
classes = [chr(a) for a in range(65,91)]

cm_cnn = confusion_matrix(true_labels, output_cnn)

correct_predictions = np.trace(cm_cnn) # Sum of diaganol elements
total_predictions = np.sum(cm_cnn) # Sum of all elements
# Calculate and print prediction accuracy for CNN model
print(f"Accuracy of CNN Model: {(correct_predictions / total_predictions):.2f}")

# Plot CNN Model's confusion matrix
fig, ax = plt.subplots(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=classes)
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
plt.title("CNN Confusion Matrix")
plt.show()

cm_dense = confusion_matrix(true_labels, output_dense)

correct_predictions = np.trace(cm_dense) # Sum of diaganol elements
total_predictions = np.sum(cm_dense) # Sum of all elements
# Calculate and print prediction accuracy for Dense model
print(f"Accuracy of Dense Model: {(correct_predictions / total_predictions):.2f}")

# Plot Dense Model's confusion matrix
fig, ax = plt.subplots(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_dense, display_labels=classes)
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
plt.title("Dense Confusion Matrix")
plt.show()
