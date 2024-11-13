import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_file, batch_size=32, image_size=(400,400), shuffle=True):
        self.data = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

        self.label_encoder = LabelEncoder()
        self.data['label'] = self.label_encoder.fit_transform(self.data['label'])

    def __len__(self):
        return len(self.data) // self.batch_size
    
    def __getitem__(self, index):
        batch_data = self.data.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_data)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
    
    def __data_generation(self, batch_data):
        X = np.empty((self.batch_size, *self.image_size, 1))
        y = np.empty((self.batch_size), dtype=int)

        for i, row in enumerate(batch_data.iterrows()):
            img_path, label = row[1]['image_path'], row[1]['label']
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode="grayscale", target_size=self.image_size)
            X[i,] = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            y[i] = label
        
        return X, y
