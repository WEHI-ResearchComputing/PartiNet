import os
import shutil
import cv2
import pandas as pd

open("annot_images.txt", "w").close()
open("noannot_images.txt", "w").close()
allimages_path="/vast/scratch/users/iskander.j/PartiNet_data/all_images"
if not os.path.exists(allimages_path):
    os.makedirs(allimages_path)

alllabels_path="/vast/scratch/users/iskander.j/PartiNet_data/all_labels"
if not os.path.exists(alllabels_path):
    os.makedirs(alllabels_path)

datasets_path="/vast/scratch/users/iskander.j/PartiNet_data/"
datasets_dir =[name for name in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, name)) ]
for dataset in datasets_dir:
    num_annotated_imgs=0
    print("Getting bounding boxes for", dataset)
    # micrograph jpegs
    images_dir = os.path.join(datasets_path,dataset,"denoised_micrographs","jpg")

    # ground truth coordinates in csv
    coords_dir = os.path.join(datasets_path,dataset,"ground_truth/particle_coordinates")

    # output directory
    annot_dir = os.path.join(datasets_path,dataset,"annotations")

    if os.path.exists(annot_dir):
        if len(os.listdir(annot_dir)) == len(os.listdir(images_dir)):
            print(f"{len(os.listdir(annot_dir))} annotated images already exist.")
            f = open("annot_images.txt", "a")
            f.write(f"{dataset},{len(os.listdir(annot_dir))}\n")
            f.close()
            continue

    if not os.path.exists(annot_dir):
        os.makedirs(annot_dir)

    for image in os.listdir(images_dir):
        if image != "all_images":
            image_name = image.split(".")[0]
            coords_file = str(image_name + ".csv")
            img = cv2.imread(os.path.join(images_dir, image), cv2.IMREAD_ANYCOLOR)
            if os.path.exists(os.path.join(coords_dir, coords_file)):
                num_annotated_imgs=num_annotated_imgs+1
                pts = pd.read_csv(os.path.join(coords_dir, coords_file))
                with open(os.path.join(annot_dir, f"{image_name}.txt"), "w") as output_file:
                    for i in range(len(pts)):
                        x = int(pts["X-Coordinate"][i]) # type: ignore
                        y = int(pts["Y-Coordinate"][i]) # type: ignore
                        radius = int(pts["Diameter"][i] / 2) # type: ignore

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
                shutil.copy2(os.path.join(images_dir, image), allimages_path)
                shutil.copy2(os.path.join(annot_dir, f"{image_name}.txt"), alllabels_path)
            else:
                print(f"{coords_file} do not exist.")
                f = open("noannot_images.txt", "a")
                f.write(f"{dataset},{image_name}\n")
                f.close()

    print(f"{num_annotated_imgs} annotated images exist.")
    f = open("annot_images.txt", "a")
    f.write(f"{dataset},{num_annotated_imgs}\n")
    f.close()