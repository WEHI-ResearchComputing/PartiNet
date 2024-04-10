import shutil
from absl import app
from absl import flags
import os
import cv2
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string('images_path', f'/vast/scratch/users/{os.getenv("USER")}/PartiNet_data/all_images', 'Path to all datasets images.')
flags.DEFINE_string('labels_path', f'/vast/scratch/users/{os.getenv("USER")}/PartiNet_data/alllabels', 'Path to all datasets labels.')
flags.DEFINE_string('output_path',f'/vast/scratch/users/{os.getenv("USER")}/PartiNet_data/dataset_split', 'Path to dataset')

def main(argv):
    if FLAGS.debug:
        print('non-flag arguments:', argv)
    if not os.path.exists(FLAGS.output_path):
        os.mkdir(FLAGS.output_path)
        os.mkdir(os.path.join(FLAGS.output_path, "images"))
        os.mkdir(os.path.join(FLAGS.output_path, "labels"))
    
if __name__ == '__main__':
    app.run(main)

def create_test_set():
    os.mkdir(os.path.join(FLAGS.output_path, "images", "test"))
    os.mkdir(os.path.join(FLAGS.output_path, "labels", "test"))

    for idx, file in enumerate(files):
        file_name = (".").join(os.path.basename(file).split('.')[0:-1])
            # copying test images
            shutil.copy(os.path.join(FLAGS.images_path, file_name+".jpg"), os.path.join(FLAGS.output_path, "images", "test", file_name+".jpg"))
            # copying test labels
            shutil.copy(os.path.join(FLAGS.labels_path, file_name+".txt"), os.path.join(FLAGS.output_path, "labels", "test", file_name+".txt"))

    # creating test.txt file
    with open(os.path.join(FLAGS.output_path, "test.txt"), "w") as f:
        for file in os.listdir(os.path.join(FLAGS.output_path, "images", "test")):
            f.write(str(os.path.join(FLAGS.output_path, "images", "test", file))+"\n")