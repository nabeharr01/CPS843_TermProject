from dataGen import DataGenerator
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder

batch_size = 128
train_generator = DataGenerator(csv_file='mp-to-asl.csv', batch_size=batch_size)
val_generator = DataGenerator(csv_file='mp-to-asl.csv', batch_size=batch_size, shuffle=False)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(36, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, validation_data=val_generator, epochs=15)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("label_classes.npy", allow_pickle=True)

def preprocess_image(img_path):
    img = keras.preprocessing.image.load_img(img_path, color_mode="grayscale", target_size=(400,400))
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    return img_array

def predict_image(img_path, model, label_encoder):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_index)

    return predicted_label[0]

img_path = 'test.png'
predicted_label = predict_image(img_path, model, label_encoder)
print(f"Predicted ASL letter/digit: {predicted_label}")

