import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

loaded_model = tf.keras.models.load_model('mp-to-asl-cnn-model.keras')

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

def preprocess_image(img_path):
    img = image.load_img(img_path, color_mode="grayscale", target_size=(400, 400))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_image(img_path, model, label_encoder):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_index)
    return predicted_label[0]

img_path = 'test.png'
predicted_label = predict_image(img_path, loaded_model, label_encoder)
print(f"Predicted ASL letter or digit: {predicted_label}")