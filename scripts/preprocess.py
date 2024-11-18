#!/stornext/System/data/apps/rc-tools/rc-tools-1.0/bin/tools/envs/py3_11/bin/python3
import json
import shutil
from absl import app
from absl import flags
from absl import logging
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split as tts

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS

flags.DEFINE_string('datasets_path',f'/vast/scratch/users/{os.getenv("USER")}/PartiNet_data/', 'Path to raw datasets')
flags.DEFINE_string('tag','', 'suffix to add to metadata files')
flags.DEFINE_string('dataset',None, 'Dataset name, should correspond to a directory name inside datasets_path, with recommended structure')
flags.DEFINE_boolean('bounding_box',False,'whether to calculate bounding box')

def validate_datasetname(path:str):
    if not os.path.exists(os.path.join(FLAGS.datasets_path,"raw",path)):
        return False
    elif not os.path.exists(os.path.join(FLAGS.datasets_path,"raw",path,"denoised_micrographs")):
        return False
    elif not os.path.exists(os.path.join(FLAGS.datasets_path,"raw",path,"ground_truth/particle_coordinates")):
        return False
    else:
        return True
    
def calculate_bounding_box(images_dir:str, coords_dir:str, annot_dir:str):
    num_annotated_imgs=0
    print("Getting bounding boxes for", FLAGS.dataset)
    # output directory
    
    if os.path.exists(annot_dir):
        if len(os.listdir(annot_dir)) == len(os.listdir(images_dir)):
            print(f"{len(os.listdir(annot_dir))} annotated images already exist.")
            return len(os.listdir(annot_dir))
    else:
        os.makedirs(annot_dir)
    ##Annotating, Bounding Box calculation
    for image in os.listdir(images_dir):
        image_name = (".").join(os.path.basename(image).split('.')[0:-1])
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
        else:
            print(f"{coords_file} do not exist.")
            with open("meta/noannot_images.txt", "a") as f:
                f.write(f"{FLAGS.dataset},{image_name}\n")

    print(f"{num_annotated_imgs} annotated images.")
    return num_annotated_imgs

def write_file_paths_to_file(filename:str, dirname:str, mode:str):
    with open(filename, mode) as f:
        for root, dirs, files in os.walk(dirname):
            for file in files:
                f.write(os.path.join(root, file) + '\n')

def prep_datasplit(images_dir:str, annot_dir:str, set:int):
    datasplit_path=os.path.join(FLAGS.datasets_path,"data_split")
    coords = os.listdir(annot_dir)
    train=0
    val=0
    test=0

    if set==2: # test set
        o_image_path=os.path.join(datasplit_path,"images","test")
        o_label_path=os.path.join(datasplit_path,"labels","test")            
        for idx, file in enumerate(coords):
            file_name = file[:-4]
            # copying test images/labels
            shutil.copy(os.path.join(images_dir, file_name+".jpg"), os.path.join(o_image_path, file_name+".jpg"))
            shutil.copy(os.path.join(annot_dir, file_name+".txt"), os.path.join(o_label_path, file_name+".txt"))
            
        # creating test.txt file 
        write_file_paths_to_file(os.path.join(datasplit_path, "test.txt"),o_image_path,"w")
        test=len(coords)

    elif set == 1: #development set
        # splitting data into training and val
        train_idx, val_idx = tts(np.arange(0, len(coords), 1), shuffle=True)
        train=len(train_idx)
        val=len(val_idx)

        
        o_image_path=os.path.join(datasplit_path,"images")
        o_label_path=os.path.join(datasplit_path,"labels")
                
        for idx, file in enumerate(coords):
            file_name = file[:-4]
            if idx in train_idx:
                # copying train images
                shutil.copy(os.path.join(images_dir, file_name+".jpg"), os.path.join(o_image_path, f"train", file_name+".jpg"))
                # copying train labels
                shutil.copy(os.path.join(annot_dir, file_name+".txt"), os.path.join(o_label_path, f"train", file_name+".txt"))

            elif idx in val_idx:
                # copying val images
                shutil.copy(os.path.join(images_dir, file_name+".jpg"), os.path.join(o_image_path, f"val", file_name+".jpg"))
                # copying val labels
                shutil.copy(os.path.join(annot_dir, file_name+".txt"), os.path.join(o_label_path, f"val", file_name+".txt"))


        # creating val.txt and train.txt file
        write_file_paths_to_file(os.path.join(datasplit_path, "val.txt"),os.path.join(o_image_path, "val"),"w")
        write_file_paths_to_file(os.path.join(datasplit_path, "train.txt"),os.path.join(o_image_path, "train"),"w")
 
        # creating cryo_training.yaml file
        training={
            "train": f"{os.path.join(datasplit_path, 'train.txt')}",
            "val": f"{os.path.join(datasplit_path, 'val.txt')}",
            "nc": 1,
            "names": [ 'particle' ]
        }
        logging.debug(training)
        with open(os.path.join(datasplit_path, "cryo_training.yaml"), "w") as f:
            json.dump(training,f)

    return train,val,test 

def main(argv):
    datasplit_path=os.path.join(FLAGS.datasets_path,"data_split")
    Path(os.path.join(datasplit_path,"images","train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"images","val")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"images","test")).mkdir(parents=True, exist_ok=True)

    Path(os.path.join(datasplit_path,"labels","train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"labels","val")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"labels","test")).mkdir(parents=True, exist_ok=True)

    # micrograph jpegs
    images_dir = os.path.join(FLAGS.datasets_path,"raw",FLAGS.dataset,"denoised_micrographs","jpg")
    # ground truth coordinates in csv
    coords_dir = os.path.join(FLAGS.datasets_path,"raw",FLAGS.dataset,"ground_truth/particle_coordinates")
    
    logging.info(f"Importing meta files, meta/development_set{FLAGS.tag}.txt and meta/test_set{FLAGS.tag}.txt ")
    dvset=pd.read_csv(f"meta/development_set{FLAGS.tag}.txt",names=['name','isnotused'])
    tstset=pd.read_csv(f"meta/test_set{FLAGS.tag}.txt",names=['name','isnotused'])
    set=0
    
    if dvset["name"].astype(str).isin([FLAGS.dataset]).any():
        set=1 #development set
    elif tstset["name"].astype(str).isin([FLAGS.dataset]).any():
        set=2 #test set
    else:
        logging.info(f"Dataset name {FLAGS.dataset} not found in lists. Please check your development and test lists, and try again.")
        return
    
    annot_dir = os.path.join(FLAGS.datasets_path,"raw",FLAGS.dataset,"annotations")
    if FLAGS.bounding_box:
        
        annotated=calculate_bounding_box(images_dir, coords_dir, annot_dir)
        logging.debug(f"set {set}, annotated:{annotated}")
    else:
        if os.path.exists(annot_dir) and (len(os.listdir(annot_dir))>0):
            logging.info("annotation file found")
        else:
            logging.error("Annotation not found, rerun with bounding_box flag to annotate images.")
            return

    
    train,val,test=prep_datasplit(images_dir, annot_dir, set)
    with open("meta/development_set_split.txt", "a") as f:
        f.write(f"{FLAGS.dataset},{train},{val},{test},\n")

if __name__ == '__main__':
    
    flags.mark_flag_as_required('dataset')
    flags.register_validator('dataset',
                         validate_datasetname,
                         message='--dataset must correspond to a directory name inside datasets_path, with  denoised_micrographs and ground_truth/particle_coordinates directories.')
    app.run(main)