#!/stornext/System/data/apps/rc-tools/rc-tools-1.0/bin/tools/envs/py3_11/bin/python3
import shutil
from absl import app
from absl import flags
from absl import logging
import os
import cv2
import pandas as pd
from pathlib import Path

logging.set_verbosity(logging.ERROR)
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path',f'/vast/scratch/users/{os.getenv("USER")}/PartiNet_data/', 'Path to raw datasets')
flags.DEFINE_string('dataset',None, 'Dataset name, should correspond to a directory name inside datasets_path, with recommended structure')


def validate_datasetname(path:str):
    if not os.path.exists(os.path.join(FLAGS.datasets_path,path)):
        return False
    elif not os.path.exists(os.path.join(FLAGS.datasets_path,path,"denoised_micrographs")):
        return False
    elif not os.path.exists(os.path.join(FLAGS.datasets_path,path,"ground_truth/particle_coordinates")):
        return False
    else:
        return True
    
def calculate_bounding_box(images_dir, coords_dir):
    num_annotated_imgs=0
    print("Getting bounding boxes for", FLAGS.dataset)
    # output directory
    annot_dir = os.path.join(FLAGS.datasets_path,FLAGS.dataset,"annotations")

    if os.path.exists(annot_dir):
        if len(os.listdir(annot_dir)) == len(os.listdir(images_dir)):
            print(f"{len(os.listdir(annot_dir))} annotated images already exist.")
            f = open("annot_images.txt", "a")
            f.write(f"{FLAGS.dataset},{len(os.listdir(annot_dir))}\n")
            f.close()
            return
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
    with open("meta/annot_images.txt", "a") as f:
        f.write(f"{FLAGS.dataset},{num_annotated_imgs}\n")

def prep_datasplit(images_dir:str, coords_dir:str, set:int,notused:bool):
    datasplit_path=os.path.join(FLAGS.dataset_path,"data_split")

    if set==2: # test set
        if notused:
            image_path=os.path.join(datasplit_path,"images","test_all")
            label_path=os.path.join(datasplit_path,"labels","test_all")
    
    files = os.listdir(coords_dir)
    shutil.copy2(os.path.join(images_dir, image), image_path)
    shutil.copy2(os.path.join(coords_dir, f"{image_name}.txt"), FLAGS.label_path)

def main(argv):
    datasplit_path=os.path.join(FLAGS.dataset_path,"data_split")
    Path(os.path.join(datasplit_path,"images","train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"images","val")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"images","test")).mkdir(parents=True, exist_ok=True)

    Path(os.path.join(datasplit_path,"labels","train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"labels","val")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"labels","test")).mkdir(parents=True, exist_ok=True)

    Path(os.path.join(datasplit_path,"images","train_all")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"images","val_all")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"images","test_all")).mkdir(parents=True, exist_ok=True)

    Path(os.path.join(datasplit_path,"labels","train_all")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"labels","val_all")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(datasplit_path,"labels","test_all")).mkdir(parents=True, exist_ok=True)
    # micrograph jpegs
    images_dir = os.path.join(FLAGS.dataset_path,FLAGS.dataset,"denoised_micrographs","jpg")
    # ground truth coordinates in csv
    coords_dir = os.path.join(FLAGS.dataset_path,FLAGS.dataset,"ground_truth/particle_coordinates")
    
    dvset=pd.read_csv("meta/development_set.txt",names=['name','isnotused'])
    tstset=pd.read_csv("meta/test_set.txt",names=['name','isnotused'])
    set=0
    notused=False
    if dvset["name"].isin([FLAGS.dataset]).any():
        notused=(dvset[dvset["name"]==FLAGS.dataset]["isnotused"]==0)
        set=1 #development set
    elif tstset["name"].isin([FLAGS.dataset]).any():
        notused=(tstset[tstset["name"]==FLAGS.dataset]["isnotused"]==0)
        set=2 #test set
    else:
        logging.fatal("Dataset name not found in lists. Please check your development and test lists, and try again.")
    calculate_bounding_box(images_dir, coords_dir)
    prep_datasplit(images_dir, coords_dir,set,notused)

if __name__ == '__main__':
    
    flags.mark_flag_as_required('dataset')
    flags.register_validator('dataset',
                         validate_datasetname,
                         message='--dataset must correspond to a directory name inside datasets_path, with  denoised_micrographs and ground_truth/particle_coordinates directories.')
    app.run(main)