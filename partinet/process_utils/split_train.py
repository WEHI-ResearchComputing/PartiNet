import os
import shutil
import argparse

import numpy as np
from sklearn.model_selection import train_test_split as tts

def main(labels_path: str, images_path: str, output_dir: str):
    """
    Splits a dataset of images and labels into training and validation sets and organizes them into 
    a specified output directory. Generates corresponding .txt files for train/val data and a 
    cryo_training.yaml file for use in model training.

    Args:
        labels_path (str): Path to the directory containing label files.
        images_path (str): Path to the directory containing image files.
        output_dir (str): Path to the output directory where split data will be saved.
    """
    # Create output directories if they do not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "images", "train"))
    os.makedirs(os.path.join(output_dir, "images", "val"))
    os.makedirs(os.path.join(output_dir, "labels", "train"))
    os.makedirs(os.path.join(output_dir, "labels", "val"))

    # List all label files
    files = os.listdir(labels_path)

    # Split data into training and validation indices
    train_idx, val_idx = tts(np.arange(0, len(files), 1), shuffle=True)

    # Iterate through files and copy them into train/val directories
    for idx, file in enumerate(files):
        file_name = file[:-4]  # Remove file extension
        if idx in train_idx:
            # Copy training images and labels
            shutil.copy(
                os.path.join(images_path, file_name + ".jpg"),
                os.path.join(output_dir, "images", "train", file_name + ".jpg")
            )
            shutil.copy(
                os.path.join(labels_path, file_name + ".txt"),
                os.path.join(output_dir, "labels", "train", file_name + ".txt")
            )
        elif idx in val_idx:
            # Copy validation images and labels
            shutil.copy(
                os.path.join(images_path, file_name + ".jpg"),
                os.path.join(output_dir, "images", "val", file_name + ".jpg")
            )
            shutil.copy(
                os.path.join(labels_path, file_name + ".txt"),
                os.path.join(output_dir, "labels", "val", file_name + ".txt")
            )

    # Create val.txt file listing validation image paths
    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        for file in os.listdir(os.path.join(output_dir, "images", "val")):
            f.write(str(os.path.join(output_dir, "images", "val", file)) + "\n")

    # Create train.txt file listing training image paths
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for file in os.listdir(os.path.join(output_dir, "images", "train")):
            f.write(str(os.path.join(output_dir, "images", "train", file)) + "\n")

    # Create cryo_training.yaml file with dataset configuration
    to_write = f"""train: {os.path.join(output_dir, "train.txt")}
val: {os.path.join(output_dir, "val.txt")}

# number of classes
nc: 1

# class names
names: [ 'particle' ]"""

    with open(os.path.join(output_dir, "cryo_training.yaml"), "w") as f:
        f.write(to_write)

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the dataset splitting script.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for labels, images, and output.
    """
    parser = argparse.ArgumentParser(description="Create training data split from images and labels.")
    parser.add_argument("--labels", required=True, help="Path to the labels directory")
    parser.add_argument("--images", required=True, help="Path to the images directory")
    parser.add_argument("--output", required=True, help="Path to the output directory")

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments and execute the main function
    args = parse_args()
    main(args.labels, args.images, args.output)
