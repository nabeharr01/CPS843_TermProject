# Execute this file when you have the processed MP-ASL dataset
# but do not have a CSV file mapping image paths to labels.
# Such a CSV file is used in training a CNN for this dataset.
# If you are not training a CNN, no need to run this file.

import os
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Directory of image dataset
directory = "./processed_combine_asl_dataset"

# Write to new CSV file
with open('mp-to-asl.csv', newline='', mode='w+') as data:
    writer = csv.writer(data, quotechar='"', quoting=csv.QUOTE_STRINGS)
    writer.writerow(["image_path", "label"]) # Headers
    # For every image in the dataset...
    for path, folders, files in os.walk(directory):
        for file in files:
            writer.writerow([
                os.path.join(path, file), # Image path relative to folder
                path.split('\\' if os.name == 'nt' else '/')[-1].upper() # Last directory name is label
            ])

# Read CSV file after it has been generated
data = pd.read_csv("mp-to-asl.csv")
labels = data['label'].values # Get labels

# LabelEncoder for mapping Labels -> encoded neural network output
label_encoder = LabelEncoder()
label_encoder.fit(labels)
# Save npy file for later, when using the trained CNN and needing encoded output -> labels
np.save("label_classes.npy", label_encoder.classes_)