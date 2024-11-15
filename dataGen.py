import cv2
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, image_size=(400,400), shuffle=True, label_classes_path='label_classes=.npy'):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_classes_path, allow_pickle=True)
        self.labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        X, y = self.__data_generation(batch_paths, batch_labels)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_paths, batch_labels):
        X = np.empty((self.batch_size, *self.image_size, 1), dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=int)

        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, self.image_size)
            image = np.expand_dims(image, axis=-1)
            X[i,] = image / 255.0
            y[i] = label
        
        return X, y
