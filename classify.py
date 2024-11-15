import os
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

directory = "./processed_combine_asl_dataset"
ext = [".png", ".jpg", ".jpeg"]

with open('mp-to-asl.csv', newline='', mode='w+') as data:
    writer = csv.writer(data, quotechar='"', quoting=csv.QUOTE_STRINGS)
    writer.writerow(["image_path", "label"])
    for path, folders, files in os.walk(directory):
        for file in files:
            writer.writerow([
                os.path.join(path, file), 
                path.split('\\' if os.name == 'nt' else '/')[-1].upper()
            ])
    
data = pd.read_csv("mp-to-asl.csv")
labels = data['label'].values

label_encoder = LabelEncoder()
label_encoder.fit(labels)
np.save("label_classes.npy", label_encoder.classes_)