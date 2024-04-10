import os
from distutils.dir_util import copy_tree
import cv2
import pandas as pd

allimages_path="/vast/scratch/users/iskander.j/PartiNet_Training/all_images"
if not os.path.exists(allimages_path):
    os.makedirs(allimages_path)

alllabels_path="/vast/scratch/users/iskander.j/PartiNet_Training/all_labels"
if not os.path.exists(alllabels_path):
    os.makedirs(alllabels_path)

datasets_path="/vast/scratch/users/iskander.j/PartiNet_data/"
datasets_dir =[name for name in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, name)) ]
for dataset in datasets_dir:
    print(f"Copying dataset {dataset}")
    # micrograph jpegs
    images_dir = os.path.join(datasets_path,dataset,"denoised_micrographs","jpg")
    # annotation directory
    annot_dir = os.path.join(datasets_path,dataset,"annotations")

    if not os.path.exists(annot_dir):
        print("Annotation directory not found!")
        break
    
    if not os.path.exists(images_dir):
        print("Image directory not found!")
        break
    if os.path.exists(annot_dir):
        if len(os.listdir(annot_dir)) == len(os.listdir(images_dir)):
            print(dataset, "already has all annotations!")
            copy_tree(images_dir, allimages_path)
            copy_tree(annot_dir, alllabels_path)

