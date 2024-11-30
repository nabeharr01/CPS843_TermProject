# This file is responsible for constructing and training
# a Dense (Fully Connected) Neural Network, trained on an ASL image dataset.
# It uses the CSV file generated from processDataset.py.
# If you are not constructing + training a Dense NN, no need to run this file.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Read CSV
data = pd.read_csv("mp-landmarks-to-asl.csv")

X = data.iloc[:, :-1].values # Extract landmark instances
y = data.iloc[:, -1].values # Extract labels

# Label encoder to transform labels -> encoded NN output
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and validation sets, stratified
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Model structure
# Simple; 5 dense/fully-connected hidden layers
model = Sequential([
    Dense(128, input_shape=(63,), activation='relu'), # input shape is length 63 array (21 points in 3-D space)
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax') # Output shape is 26 for ASL alphabet
])

# Compiling with an optimizer and the appropriate loss function
model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback function
# If validation accuracy does not improve after 10 epochs, restore and stop training
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Train with GPU! (Done with Google Colab A100)
with tf.device('/gpu:0'):
    # Train for 300 epochs with early stopping, with 0.001 learning rate
    history = model.fit(
        X_train, y_train, 
        epochs=300, batch_size=256,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

np.save("label_classes_dense.npy", label_encoder.classes_)
model.save("MP-to-ASL-dense-model.keras") # Save model for later application