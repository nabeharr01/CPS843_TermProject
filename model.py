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

model.save("mp-to-asl-cnn-model.keras")