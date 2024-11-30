# This file defines a Predictor class that takes a Keras model and a label encoder, for taking inputs
# 1. Can take a 400x400 image path or Numpy array and pass it to a model, return decoded ASL alphabet label
# 2. Can take a length 63 array of 21 3-D points and pass it to a model, returning decoded ASL alphabet label
#
# This is used in main.py to predict letters posed by the user

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

class Predictor:
    def __init__(self, model, label_classes_path):
        self.model = model
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_classes_path, allow_pickle=True)

    def __preprocess_image_from_path(self, img_path):
        # Load image from path, in grayscale
        img = image.load_img(img_path, color_mode="grayscale", target_size=(400, 400))
        # Normalize
        img_array = image.img_to_array(img) / 255.0
        # Transform from shape (400,400) to (400,400,1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    
    def __preprocess_image_array(self, img_array):
        # Check image shape
        assert img_array.shape == (400,400,3), "Input must be a (400,400,3) array"

        # Average 3 colour values (RGB) to grayscale
        grayscale_img = np.mean(img_array, axis=-1, keepdims=True)
        # Normalize
        grayscale_img = grayscale_img / 255.0
        # Transform from shape (400,400) to (400,400,1)
        grayscale_img = np.expand_dims(grayscale_img, axis=0)
        return grayscale_img

    def predict_image_from_path(self, img_path):
        processed_array = self.__preprocess_image_from_path(img_path)
        # Predict with model, take argmax of output nodes, get decoded label
        predictions = self.model.predict(processed_array, verbose=0)
        predicted_index = np.argmax(predictions, axis=1)
        predicted_label = self.label_encoder.inverse_transform(predicted_index)

        return predicted_label[0]
    
    def predict_image_array(self, img_array):
        processed_array = self.__preprocess_image_array(img_array)
        # Predict with model, take argmax of output nodes, get decoded label
        predictions = self.model.predict(processed_array, verbose=0)
        predicted_index = np.argmax(predictions, axis=1)
        predicted_label = self.label_encoder.inverse_transform(predicted_index)

        return predicted_label[0]
    
    def predict_landmarks(self, landmarks):
        # Check landmarks shape
        assert landmarks.shape == (1,63), "Input must be have shape (1,63)"

        # Predict with model, take argmax of output nodes, get decoded label
        predictions = self.model.predict(landmarks, verbose=0)
        predicted_index = np.argmax(predictions, axis=1)
        predicted_label = self.label_encoder.inverse_transform(predicted_index)

        return predicted_label[0]