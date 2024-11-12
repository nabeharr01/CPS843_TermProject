import os
import csv

directory = "./processed_combine_asl_dataset"
ext = [".png", ".jpg", ".jpeg"]

with open('mp-to-asl.csv', newline='', mode='w+') as data:
    writer = csv.writer(data, quotechar='"', quoting=csv.QUOTE_STRINGS)
    writer.writerow(["image_path", "label"])
    for path, folders, files in os.walk(directory):
        for file in files:
            writer.writerow([
                os.path.join(path, file), 
                path.split('\\')[-1].upper()
            ])
    
