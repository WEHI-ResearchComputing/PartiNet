import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split as tts

# put your paths here
labels_path = "/vast/scratch/users/iskander.j/PartiNet_Training/all_labels"
images_path = "/vast/scratch/users/iskander.j/PartiNet_Training/all_images"
output_dir = "/vast/scratch/users/iskander.j/PartiNet_Training"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, "images"))
    os.mkdir(os.path.join(output_dir, "labels"))
    os.mkdir(os.path.join(output_dir, "images", "train"))
    os.mkdir(os.path.join(output_dir, "images", "val"))
    os.mkdir(os.path.join(output_dir, "labels", "train"))
    os.mkdir(os.path.join(output_dir, "labels", "val"))

files = os.listdir(labels_path)
# splitting data into training and val
train_idx, val_idx = tts(np.arange(0, len(files), 1), shuffle=True)

for idx, file in enumerate(files):
    file_name = file[:-4]
    if idx in train_idx:
        # copying train images
        shutil.copy(os.path.join(images_path, file_name+".jpg"), os.path.join(output_dir, "images", "train", file_name+".jpg"))

        # copying train labels
        shutil.copy(os.path.join(labels_path, file_name+".txt"), os.path.join(output_dir, "labels", "train", file_name+".txt"))

    elif idx in val_idx:
        # copying val images
        shutil.copy(os.path.join(images_path, file_name+".jpg"), os.path.join(output_dir, "images", "val", file_name+".jpg"))

        # copying val labels
        shutil.copy(os.path.join(labels_path, file_name+".txt"), os.path.join(output_dir, "labels", "val", file_name+".txt"))


# creating val.txt file
with open(os.path.join(output_dir, "val.txt"), "w") as f:
    for file in os.listdir(os.path.join(output_dir, "images", "val")):
        f.write(str(os.path.join(output_dir, "images", "val", file))+"\n")

# creating train.txt file
with open(os.path.join(output_dir, "train.txt"), "w") as f:
    for file in os.listdir(os.path.join(output_dir, "images", "train")):
        f.write(str(os.path.join(output_dir, "images", "train", file))+"\n")

# creating cryo_training.yaml file
to_write = f"""train: {os.path.join(output_dir, "train.txt")}
val: {os.path.join(output_dir, "val.txt")}

# number of classes
nc: 1

# class names
names: [ 'particle' ]"""
with open(os.path.join(output_dir, "cryo_training.yaml"), "w") as f:
    f.write(to_write)

