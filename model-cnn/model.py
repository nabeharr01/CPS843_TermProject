# This file is responsible for constructing and training
# a Convolutional Neural Network, trained on the processed MP-ASL dataset.
# It uses the CSV file generated from classify.py.
# If you are not constructing + training a CNN, no need to run this file.

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from dataGen import DataGenerator

batch_size = 256 # Batch size for loading images into memory (DataGen.py)
image_size = (400,400) # Standardized image size from dataset
num_outputs = 26 # 26 classes for 26 ASL alphabet letters

# Extract image paths + labels from CSV file
data = pd.read_csv("mp-to-asl.csv")
image_paths = data['image_path'].values
labels = data['label'].values

# Split data into training and validation sets, stratified
paths_train, paths_val, labels_train, labels_val = train_test_split(
    image_paths, labels, test_size=0.4, random_state=42, stratify=labels
)

# Generators for training and validation for batching
train_generator = DataGenerator(paths_train, labels_train, batch_size, image_size, shuffle=False)
val_generator = DataGenerator(paths_val, labels_val, batch_size, image_size, shuffle=False)

# Model structure
# Convolutional layers followed by flattening and fully connected layers (with some dropouts)
# as befits a CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 1)), # input shape is 400x400 grayscale image
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
    Dense(num_outputs, activation='softmax') # Output shape is num_outputs (26 for ASL alphabet)
])

# Compiling with an optimizer and the appropriate loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callback function for reducing the learning rate as training goes on
reduce_lr = ReduceLROnPlateau(
  monitor='val_loss', # Monitor validation loss as the gauge for reduction
  factor=0.5, # Reduce learning rate by a factor of 0.5
  patience=2, # Wait for 2 epochs with no improvement before reducing
  min_lr=1e-6 # Minimum learning rate
)

# Train with GPU! (Done with Google Colab A100)
with tf.device('/gpu:0'):
    # Train with batch generators; for 10 epochs; with learning rate reduction
    model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[reduce_lr])

model.save("mp-to-asl-cnn-model.keras") # Save model for later application