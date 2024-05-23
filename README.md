# PartiNet
A particle picking tool that uses DynamicDet and trained on CryoPPP


## File structure
* DynamicDet  --> submodule forked from [DynamicDet repo](https://github.com/VDIGPKU/DynamicDet)
* scripts
    * meta --> have text files that are include:
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




