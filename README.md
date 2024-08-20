# PartiNet
A particle picking tool that uses DynamicDet and trained on CryoPPP

## File structure

* DynamicDet  --> submodule forked from [DynamicDet repo](https://github.com/VDIGPKU/DynamicDet)
* scripts
    * meta --> have text files that include:
        * `datasets.txt` : preprocess script to choose datasets to denoise and calculate bounding box ()
        * `development_set.txt` & `test_set.txt` : preprocess script to split into development and test sets.
        * `fix_names_datasets.txt`: contain dataset names that reqiored manual intervention after download.
    * `download.sh` : Slurm job to get dataset name then download and untar. 
    You can also retrieve from RCP a tarred file`raw.tar.gz` with all datasets and untar then using 
            ```
            for f in /vast/projects/RCP/PartiNet_data/tarred/*.tar.gz; do tar xvf "$f"; done
            ```

    * `visualise_denoise_reaults.ipynb` : for checking results visually.
    *  `preprocess.sh`: Slurm job that runs 
        * topaz denoising 
        * `preprocess.py` which calculates bounding box then split images of a dataset into train and val sets if in the development_set, or move to test if in test_set.
    * `generate-star-file` --> ??
    * `generate-star-file` --> ??
    * `detect` --> ??
    * `train_step1` and `train_step2`  --> ??

## Install

```bash
git clone --recursive git@github.com:WEHI-ResearchComputing/PartiNet.git
cd PartiNet
pip install .
```

## Usage

```bash
partinet --help
```
```
Usage: partinet [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  detect
  preprocess
  train
```

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

### Training

Training the DynamicDet network is seperated into two steps and therefore two subcommands:

```bash
partinet train --help
```
```
Usage: partinet train [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  step1
  step2
```

#### Training Step 1

relevant training args are passed to the `train` subcommand with step1 specific args passed to the
`step1` subsubcommand.

```bash
partinet train step1 --help
```
```
Usage: partinet train step1 [OPTIONS]

Options:
  --backbone-detector [yolov7|yolov7-w6|yolov7x]
                                  The choice of backbone to be used.
                                  [default: yolov7]
  --weight TEXT                   initial weights path  [required]
  --data TEXT                     data.yaml path  [default: data/coco.yaml]
  --hyp [scratch.p5|scratch.p6|finetune.dynamic.adam]
                                  hyperparameters path  [default: scratch.p5]
  --epochs INTEGER                [default: 300]
  --batch-size INTEGER            total batch size for all GPUs  [default: 16]
  --img-size INTEGER...           [train, test] image sizes  [default: 640,
                                  640]
  --rect                          rectangular training
  --resume                        resume most recent training
  --resume-ckpt TEXT              checkpoint to resume from
  --nosave                        only save final checkpoint
  --notest                        only test final epoch
  --noautoanchor                  disable autoanchor check
  --bucket TEXT                   gsutil bucket
  --cache-images                  cache images for faster training
  --image-weights                 use weighted image selection for training
  --device TEXT                   cuda device, i.e. 0 or 0,1,2,3 or cpu
  --multi-scale                   vary img-size +/- 50%%
  --single-cls                    train multi-class data as single-class
  --adam                          use torch.optim.Adam() optimizer
  --sync-bn                       use SyncBatchNorm, only available in DDP
                                  mode
  --local_rank INTEGER            DDP parameter, do not modify  [default: -1]
  --workers INTEGER               maximum number of dataloader workers
                                  [default: 8]
  --project TEXT                  save to project/name  [default: runs/train]
  --entity TEXT                   W&B entity
  --name TEXT                     save to project/name  [default: exp]
  --exist-ok                      existing project/name ok, do not increment
  --quad                          quad dataloader
  --label-smoothing FLOAT         Label smoothing epsilon  [default: 0.0]
  --upload_dataset                Upload dataset as W&B artifact table
  --bbox_interval INTEGER         Set bounding-box image logging interval for
                                  W&B  [default: -1]
  --save_period INTEGER           Log model after every "save_period" epoch
                                  [default: -1]
  --artifact_alias TEXT           version of dataset artifact to be used
                                  [default: latest]
  --freeze INTEGER                Freeze layers: backbone of yolov7=50,
                                  first3=0 1 2  [default: 0]
  --v5-metric                     assume maximum recall as 1.0 in AP
                                  calculation
  --single-backbone               train single backbone model
  --linear-lr                     linear LR
  --help                          Show this message and exit.
```

TODO: more details...
Example
```
partinet train step1 --cfg partinet/DynamicDet/cfg/dy-yolov7-step1.yaml --weight '' --data path/to/cryo_training.yaml --hyp partinet/DynamicDet/hyp/hyp.scratch.p5.yaml --name train_step1 --save_period 10 --epochs 20 --batch-size 16 --img-size 640 640 --workers 16 --device 0,1,2,3 --sync-bn

```
#### Training Step 2

Like step1, training args are passed to `train`, but no special arguments are passed to the `step2`
subsubcommand.

```output
Usage: partinet train step2 [OPTIONS]

Options:
  --backbone-detector [yolov7|yolov7-w6|yolov7x]
                                  The choice of backbone to be used.
                                  [default: yolov7]
  --weight TEXT                   initial weights path  [required]
  --data TEXT                     data.yaml path  [default: data/coco.yaml]
  --hyp [scratch.p5|scratch.p6|finetune.dynamic.adam]
                                  hyperparameters path  [default: scratch.p5]
  --epochs INTEGER                [default: 300]
  --batch-size INTEGER            total batch size for all GPUs  [default: 16]
  --img-size INTEGER...           [train, test] image sizes  [default: 640,
                                  640]
  --rect                          rectangular training
  --resume                        resume most recent training
  --resume-ckpt TEXT              checkpoint to resume from
  --nosave                        only save final checkpoint
  --notest                        only test final epoch
  --noautoanchor                  disable autoanchor check
  --bucket TEXT                   gsutil bucket
  --cache-images                  cache images for faster training
  --image-weights                 use weighted image selection for training
  --device TEXT                   cuda device, i.e. 0 or 0,1,2,3 or cpu
  --multi-scale                   vary img-size +/- 50%%
  --single-cls                    train multi-class data as single-class
  --adam                          use torch.optim.Adam() optimizer
  --sync-bn                       use SyncBatchNorm, only available in DDP
                                  mode
  --local_rank INTEGER            DDP parameter, do not modify  [default: -1]
  --workers INTEGER               maximum number of dataloader workers
                                  [default: 8]
  --project TEXT                  save to project/name  [default: runs/train]
  --entity TEXT                   W&B entity
  --name TEXT                     save to project/name  [default: exp]
  --exist-ok                      existing project/name ok, do not increment
  --quad                          quad dataloader
  --label-smoothing FLOAT         Label smoothing epsilon  [default: 0.0]
  --upload_dataset                Upload dataset as W&B artifact table
  --bbox_interval INTEGER         Set bounding-box image logging interval for
                                  W&B  [default: -1]
  --save_period INTEGER           Log model after every "save_period" epoch
                                  [default: -1]
  --artifact_alias TEXT           version of dataset artifact to be used
                                  [default: latest]
  --freeze INTEGER                Freeze layers: backbone of yolov7=50,
                                  first3=0 1 2  [default: 0]
  --v5-metric                     assume maximum recall as 1.0 in AP
                                  calculation
  --help                          Show this message and exit.
```

TODO: more details...
Example
```bash
partinet train step2 --backbone-detector yolov7 --weight /path/to/runs/train/train-step1-300epochs/weights/last.pt --workers 4 --device 0 --batch-size 1 --epochs 10 --img-size 640 640  --adam --data /path/to/cryo_training_all.yaml --hyp finetune.dynamic.adam --name train_step2
```
## Detection

```bash
partinet detect --help
```
```
Options:
  --backbone-detector [yolov7|yolov7-w6|yolov7x]
                                  The choice of backbone to be used.
                                  [default: yolov7]
  --weight TEXT                   model.pt path(s)  [required]
  --source TEXT                   source  [default: inference/images]
  --num-classes INTEGER           number of classes  [default: 80]
  --img-size INTEGER              inference size (pixels)  [default: 640]
  --conf-thres FLOAT              object confidence threshold  [default: 0.25]
  --iou-thres FLOAT               IOU threshold for NMS  [default: 0.45]
  --device TEXT                   cuda device, i.e. 0 or 0,1,2,3 or cpu
  --view-img                      display results
  --save-txt                      save results to *.txt
  --save-conf                     save confidences in --save-txt labels
  --nosave                        do not save images/videos
  --classes INTEGER               filter by class: --classes 0, or --classes 0
                                  --classes 2 --classes 3
  --agnostic-nms                  class-agnostic NMS
  --augment                       augmented inference
  --project TEXT                  save results to project/name  [default:
                                  runs/detect]
  --name TEXT                     save results to project/name  [default: exp]
  --exist-ok                      existing project/name ok, do not increment
  --dy-thres FLOAT                dynamic thres  [default: 0.5]
  --help                          Show this message and exit.
```

TODO: more details...
