import math
import os
import pandas as pd
import csv
import cv2
import argparse
from typing import List, Tuple, Dict

def yolo_to_starfile(yolo_coords: Dict[str, float], image_width: int, image_height: int, diameters: List[int]) -> Tuple[int, int, int]:
    """
    Convert YOLO bounding box coordinates to STAR file format.

    Args:
        yolo_coords (dict): Dictionary with YOLO bounding box attributes.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        diameters (list): List to append the calculated diameters.

    Returns:
        tuple: (x_center, y_center, diameter) in STAR file format.
    """
    x_center = math.ceil(yolo_coords['x_centre'] * image_width)
    y_center = math.ceil(yolo_coords['y_centre'] * image_height)
    width = yolo_coords['width'] * image_width
    height = yolo_coords['height'] * image_height

    # Calculate the diameter
    diameter = math.ceil(max(width, height))
    diameters.append(diameter)

    return x_center, y_center, diameter

def generate_output(labels: pd.DataFrame, filename: str, star_writer: csv.writer, diameters: List[int], img_width: int, img_height: int) -> None:
    """
    Write particle data to the STAR file.

    Args:
        labels (pd.DataFrame): DataFrame containing particle bounding box data.
        filename (str): Name of the micrograph file.
        star_writer (csv.writer): CSV writer object for the STAR file.
        diameters (list): List to track calculated diameters.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
    """
    for _, row in labels.iterrows():
        x_centre, y_centre, diameter = yolo_to_starfile(row, img_width, img_height, diameters)
        star_writer.writerow([filename, x_centre, y_centre, diameter])

def main(labels_path: str, images_path: str, star_out_path: str, conf_thresh: float) -> None:
    """
    Main function to process YOLO prediction labels and generate a STAR file.

    Args:
        labels_path (str): Path to the directory containing YOLO label files.
        images_path (str): Path to the directory containing images.
        star_out_path (str): Path to the output STAR file.
        conf_thresh (float): Minimum confidence threshold for predictions.
    """
    diameters: List[int] = []

    with open(star_out_path, "w") as star_file:
        star_writer = csv.writer(star_file, delimiter=' ')

        # Write STAR file header
        star_writer.writerow([])
        star_writer.writerow(["data_"])
        star_writer.writerow([])
        star_writer.writerow(["loop_"])
        star_writer.writerow(["_rlnMicrographName", "#1"])
        star_writer.writerow(["_rlnCoordinateX", "#2"])
        star_writer.writerow(["_rlnCoordinateY", "#3"])
        star_writer.writerow(["_rlnDiameter", "#4"])

        for i, image_file in enumerate(os.listdir(images_path), start=1):
            print(f"Processing image {i}: {image_file}")

            filename = os.path.splitext(image_file)[0]
            label_file_path = os.path.join(labels_path, f"{filename}.txt")

            if not os.path.exists(label_file_path):
                print(f"Warning: Label file not found for image {image_file}. Skipping.")
                continue

            # Read the image to get dimensions
            image = cv2.imread(os.path.join(images_path, image_file))
            img_width, img_height = image.shape[1], image.shape[0]

            # Read YOLO labels
            custom_headers = ['class', 'x_centre', 'y_centre', 'width', 'height', 'conf']
            labels = pd.read_csv(label_file_path, header=None, names=custom_headers, sep=' ')

            # Filter labels by confidence threshold
            labels = labels[labels['conf'] > float(conf_thresh)]

            # Generate output for the STAR file
            star_filename = f"{filename}.mrc"
            generate_output(labels, star_filename, star_writer, diameters, img_width, img_height)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate STAR file from PartiNet predictions")
    parser.add_argument("--labels", required=True, help="Path to the labels directory")
    parser.add_argument("--images", required=True, help="Path to the images directory")
    parser.add_argument("--output", required=True, help="Path to the output STAR file")
    parser.add_argument("--conf", required=True, type=float, help="Minimum confidence threshold for predictions")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.labels, args.images, args.output, args.conf)
