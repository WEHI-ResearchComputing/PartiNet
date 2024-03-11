import os
import shutil

# put your paths here
labels_path = "/path/to/yolo_format/labels"
images_path = "/path/to/denoised/images"
output_dir = "/training/directory"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, "images"))
    os.mkdir(os.path.join(output_dir, "labels"))
    os.mkdir(os.path.join(output_dir, "images", "test"))
    os.mkdir(os.path.join(output_dir, "images", "test"))

for idx, file in enumerate(files):
    file_name = file[:-4]
        # copying test images
        shutil.copy(os.path.join(images_path, file_name+".jpg"), os.path.join(output_dir, "images", "test", file_name+".jpg"))

        # copying test labels
        shutil.copy(os.path.join(labels_path, file_name+".txt"), os.path.join(output_dir, "labels", "test", file_name+".txt"))

# creating test.txt file
with open(os.path.join(output_dir, "test.txt"), "w") as f:
    for file in os.listdir(os.path.join(output_dir, "images", "test")):
        f.write(str(os.path.join(output_dir, "images", "test", file))+"\n")