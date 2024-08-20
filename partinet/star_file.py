import math
import os

import pandas as pd
import csv
import cv2
import argparse

def yolo_to_starfile(yolo_coords, image_width, image_height,diameters):
    x_center = math.ceil(yolo_coords['x_centre'] * image_width)
    y_center = math.ceil(yolo_coords['y_centre'] * image_height)
    width = yolo_coords['width'] * image_width
    height = yolo_coords['height'] * image_height

    # Calculate diameter
    diameter = math.ceil(max(width, height))

    diameters.append(diameter)

    # Return x_center, y_center, diameter
    return x_center, y_center, diameter


# Modify the generate_output function to accept DynamicDet DataFrame
def generate_output(labels, filename, star_writer,diameters,img_width,img_height):
    for index, row in labels.iterrows():
        # x_centre = row['X-Coordinate']
        # y_centre = row['Y-Coordinate']
        # diameter = row['Diameter']

        x_centre, y_centre, diameter = yolo_to_starfile(row, img_width, img_height,diameters)
        # print(x_centre, y_centre, diameter)

        star_writer.writerow([filename, x_centre, y_centre, diameter])

def main(labels_path,images_path,star_out_path,conf_thresh):
    # # Read DynamicDet output from txt file
    # labels_path = "path/to/labels"
    # images_path = "path/to/denoised_images"
    # star_out_path = "output/path/file.star"
    # conf_thresh = 0.5

    diameters = list()

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
        
        i = 1
        for image in os.listdir(images_path):
            filename = image.split("/")[-1][:-4]
            
            print(i)
            
            i+=1
            
            label_file_path = os.path.join(labels_path, str(filename + '.txt'))
            
            if not os.path.exists(label_file_path):
                continue
                
            # label_file_path = os.path.join(labels_path, str(filename + '.csv'))
            # labels = pd.read_csv(label_file_path)

            image = cv2.imread(os.path.join(images_path, image))
            img_width = image.shape[1]
            img_height = image.shape[0]

                
            custom_headers = ['class', 'x_centre', 'y_centre', 'width', 'height', 'conf']

            # Read the CSV file with custom headers
            labels = pd.read_csv(label_file_path, header=None, names=custom_headers, sep=' ')
        
            labels = labels[labels['conf'] > conf_thresh]
            
            # print(labels.head())

            # label_file_path = os.path.join(labels_path, str(filename + '.csv'))
            # labels = pd.read_csv(label_file_path)

            star_filename = str(filename + '.mrc')

            generate_output(labels, star_filename, star_writer,diameters,img_width,img_height)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate STAR file from PartiNet predictions")
    parser.add_argument("--labels", required=True, help="Path to the labels directory")
    parser.add_argument("--images", required=True, help="Path to the images directory")
    parser.add_argument("--output", required=True, help="Path to the output STAR file")
    parser.add_argument("--conf", required=True, help="Minimum confidence threshold for predictions")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.labels, args.images, args.output, args.conf)