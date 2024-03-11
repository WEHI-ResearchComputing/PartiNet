import os
import cv2
import pandas as pd

datasets_dir = os.listdir('/vast/projects/miti2324/data/cryoppp_data/cryoppp_lite')

for dataset in datasets_dir:
    print("Getting bounding boxes for", dataset)
    # micrograph jpegs
    images_dir = f"/vast/projects/miti2324/data/cryoppp_data/cryoppp_lite/{dataset}/micrographs/"

    # ground truth coordinates in csv
    coords_dir = f"/vast/projects/miti2324/data/cryoppp_data/cryoppp_lite/{dataset}/ground_truth/particle_coordinates"

    # output directory
    annot_dir = f"/vast/projects/miti2324/data/cryoppp_data/cryoppp_lite/{dataset}/annotations"

    if os.path.exists(annot_dir):
        if len(os.listdir(annot_dir)) == len(os.listdir(images_dir)):
            print(dataset, "already has all annotations!")
            continue

    if not os.path.exists(annot_dir):
        os.makedirs(annot_dir)

    for image in os.listdir(images_dir):
        image_name = image.split(".")[0]
        coords_file = str(image_name + ".csv")

        img = cv2.imread(os.path.join(images_dir, image), cv2.IMREAD_ANYCOLOR)
        pts = pd.read_csv(os.path.join(coords_dir, coords_file))

        with open(os.path.join(annot_dir, f"{image_name}.txt"), "w") as output_file:
            for i in range(len(pts)):
                x = int(pts["X-Coordinate"][i])
                y = int(pts["Y-Coordinate"][i])
                radius = int(pts["Diameter"][i] / 2)

                # Calculate bounding box coordinates
                x_min = x - radius
                y_min = y - radius
                x_max = x + radius
                y_max = y + radius

                # Calculate YOLO coordinates
                x_center = ((x_max + x_min) / 2) / img.shape[1]
                y_center = ((y_min + y_max) / 2) / img.shape[0]
                width = (x_max - x_min) / img.shape[1]
                height = (y_max - y_min) / img.shape[0]

                output_line = f"0 {x_center} {y_center} {width} {height}\n"
                output_file.write(output_line)