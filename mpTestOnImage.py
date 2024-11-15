import cv2
import tensorflow as tf
import numpy as np
from predictor import Predictor
from tensorflow.keras.preprocessing import image

loaded_model = tf.keras.models.load_model('mp-to-asl-cnn-model.keras')

predictor = Predictor(loaded_model, "label_classes.npy")

img_path = 'test.png'
predicted_label_img = predictor.predict_image_from_path(img_path)
print(f"Predicted ASL letter or digit: {predicted_label_img}")

img_array = image.load_img(img_path)
img_array = image.img_to_array(img_array) 
grayscale_img = np.mean(img_array, axis=-1, keepdims=True)
grayscale_img = grayscale_img / 255.0
predicted_label_array = predictor.predict_image_array(img_array)
print(f"Predicted ASL (array): {predicted_label_array}")