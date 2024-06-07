# PartiNet
A particle picking tool that uses DynamicDet and trained on CryoPPP


## File structure
* DynamicDet  --> submodule forked from [DynamicDet repo](https://github.com/VDIGPKU/DynamicDet)
* scripts
    * meta --> have text files that include:
        * `datasets.txt` : preprocess script to choose datasets to denoise and calculate bounding box ()
        * `development_set.txt` & `test_set.txt` : preprocess script to split into development and test sets.
        * `fix_names_datasets.txt`: contain dataset names that reqiored manual intervention after download.
    *`download.sh` : Slurm job to get dataset name then download and untar.
    * `visualise_denoise_reaults.ipynb` : for checking results visually.
    *  `preprocess.sh`: Slurm job that runs 
        * topaz denoising 
        * `preprocess.py` which calculates bounding box then split images of a dataset into train and val sets if in the development_set, or move to test if in test_set.
    * `generate-star-file` --> ??
    * `generate-star-file` --> ??
    * `detect` --> ??
    * `train_step1` and `train_step2`  --> ??

## Usage
### Preprocessing
This step runs on one dataset to creating bounding boxes and split images into development and test sets.

The Python script uses the conda env installed `/stornext/System/data/apps/rc-tools/rc-tools-1.0/bin/tools/envs/py3_11/bin/python3`

Example run found in `preprocess.py`

```

./preprocess.py --dataset <dataset_name> --datasets_path /vast/scratch/users/iskander.j/PartiNet_data/testing/ --tag _test --bounding_box

```

#### Arguments:

* dataset: dataset name, e.g. 10005, and it must be a directory found in datasets_path
* datasets_path: path to all datasets
* tag: to add a suffix to text files used by script.
* bounding_box: whether to run the calculate bounding box annotation step, default is false and will error if annotation directory is empty.

#### How it works:
First step is go to dataset path and create directory called `annotations` where bounding box data will be saved as `*.txt` files.

Then, using the two text files `development_set<tag>.txt` & `test_set<tag>.txt`, the dataset images and label (annotations) will be split into test and development (train and val) sets and saved to `data_split` directory.

The script writes 
* noannot_images.txt: for each dataset, list of micrographs without annotation.
* development_set_split.txt: csv that saves dataset, number of annotated micrographs, number of images in training set, number of images in validation set,number of images in test set.


**Note: There are a few manual steps required before running any preprocessing (https://github.com/WEHI-ResearchComputing/PartiNet/blob/main/scripts/meta/fix_names_datasets.txt)**

