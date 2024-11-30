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
        img = image.load_img(img_path, color_mode="grayscale", target_size=(400, 400))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    
    def __preprocess_image_array(self, img_array):
        assert img_array.shape == (400,400,3), "Input must be a (400,400,3) array"

        grayscale_img = np.mean(img_array, axis=-1, keepdims=True)
        grayscale_img = grayscale_img / 255.0
        grayscale_img = np.expand_dims(grayscale_img, axis=0)
        return grayscale_img

    def predict_image_from_path(self, img_path):
        processed_array = self.__preprocess_image_from_path(img_path)
        predictions = self.model.predict(processed_array, verbose=0)
        predicted_index = np.argmax(predictions, axis=1)
        predicted_label = self.label_encoder.inverse_transform(predicted_index)

        return predicted_label[0]
    
    def predict_image_array(self, img_array):
        processed_array = self.__preprocess_image_array(img_array)
        predictions = self.model.predict(processed_array, verbose=0)
        predicted_index = [max(0, np.argmax(predictions, axis=1)[0] - 10)]
        predicted_label = self.label_encoder.inverse_transform(predicted_index)

        return predicted_label[0]
    
    def predict_landmarks(self, landmarks):
        assert landmarks.shape == (1,63), "Input must be have shape (1,63)"

        predictions = self.model.predict(landmarks, verbose=0)
        predicted_index = np.argmax(predictions, axis=1)
        predicted_label = self.label_encoder.inverse_transform(predicted_index)

        return predicted_label[0]