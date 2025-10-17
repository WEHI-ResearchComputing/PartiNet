import math
import os
import pandas as pd
import csv
import cv2
import argparse
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count

def yolo_to_starfile(yolo_coords: Dict[str, float], image_width: int, image_height: int, diameters: List[int]) -> Tuple[int, int, int]:
    x_center = math.ceil(yolo_coords['x_centre'] * image_width)
    y_center = math.ceil(yolo_coords['y_centre'] * image_height)
    width = yolo_coords['width'] * image_width
    height = yolo_coords['height'] * image_height
    diameter = math.ceil(max(width, height))
    diameters.append(diameter)
    return x_center, y_center, diameter

def generate_output(labels: pd.DataFrame, filename: str, img_width: int, img_height: int) -> List[Tuple[str, int, int, int]]:
    """
    Generate rows for the STAR file for a single image
    """
    diameters: List[int] = []
    output_rows = []
    for _, row in labels.iterrows():
        x_centre, y_centre, diameter = yolo_to_starfile(row, img_width, img_height, diameters)
        star_filename = f"{filename}.mrc"
        output_rows.append((star_filename, x_centre, y_centre, diameter))
    return output_rows

def process_image(args_tuple) -> List[Tuple[str, int, int, int]]:
    """
    Process a single image and return STAR rows
    """
    image_file, labels_path, images_path, conf_thresh = args_tuple
    filename = os.path.splitext(image_file)[0]
    label_file_path = os.path.join(labels_path, f"{filename}.txt")

    if not os.path.exists(label_file_path):
        print(f"Warning: Label file not found for image {image_file}. Skipping.")
        return []

    # Read the image to get dimensions
    image = cv2.imread(os.path.join(images_path, image_file))
    if image is None:
        print(f"Warning: Could not read image {image_file}. Skipping.")
        return []
    img_width, img_height = image.shape[1], image.shape[0]

    # Read YOLO labels
    custom_headers = ['class', 'x_centre', 'y_centre', 'width', 'height', 'conf']
    labels = pd.read_csv(label_file_path, header=None, names=custom_headers, sep=' ')
    labels = labels[labels['conf'] > float(conf_thresh)]

    # Generate STAR rows
    return generate_output(labels, filename, img_width, img_height)

def main(labels_path: str, images_path: str, star_out_path: str, conf_thresh: float) -> None:
    image_files = os.listdir(images_path)
    args_list = [(img_file, labels_path, images_path, conf_thresh) for img_file in image_files]

    # Use all available CPUs
    with Pool(cpu_count()) as pool:
        results = pool.map(process_image, args_list)

    # Flatten the results
    all_rows = [row for result in results for row in result]

    # Write STAR file
    with open(star_out_path, "w") as star_file:
        star_writer = csv.writer(star_file, delimiter=' ')
        star_writer.writerow([])
        star_writer.writerow(["data_"])
        star_writer.writerow([])
        star_writer.writerow(["loop_"])
        star_writer.writerow(["_rlnMicrographName", "#1"])
        star_writer.writerow(["_rlnCoordinateX", "#2"])
        star_writer.writerow(["_rlnCoordinateY", "#3"])
        star_writer.writerow(["_rlnDiameter", "#4"])
        star_writer.writerows(all_rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate STAR file from YOLO predictions")
    parser.add_argument("--labels", required=True, help="Path to the labels directory")
    parser.add_argument("--images", required=True, help="Path to the images directory")
    parser.add_argument("--output", required=True, help="Path to the output STAR file")
    parser.add_argument("--conf", required=True, type=float, help="Minimum confidence threshold for predictions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.labels, args.images, args.output, args.conf)