import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

batch_size = 256
image_size = (400,400)
num_outputs = 26

data = pd.read_csv("mp-to-asl.csv")
image_paths = data['image_path'].values
labels = data['label'].values

paths_train, paths_val, labels_train, labels_val = train_test_split(
    image_paths, labels, test_size=0.4, random_state=42, stratify=labels
)

train_generator = DataGenerator(paths_train, labels_train, batch_size, image_size, shuffle=False)
val_generator = DataGenerator(paths_val, labels_val, batch_size, image_size, shuffle=False)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5), # Dropout for regularization
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout for regularization
    Dense(num_outputs, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
  factor=0.5,  # Reduce learning rate by a factor of 0.5
  patience=2,  # Wait for 2 epochs with no improvement before reducing
  min_lr=1e-6)  # Minimum learning rate

with tf.device('/gpu:0'):
    model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[reduce_lr])

model.save("mp-to-asl-cnn-model.keras")