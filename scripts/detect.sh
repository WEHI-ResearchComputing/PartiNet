#!/bin/bash

#SBATCH --cpus-per-task 16 --mem=499G --gres=gpu:A100:1 -p gpuq
#SBATCH --output=detect.out --job-name=detect
#SBATCH --mail-type=ALL

python DynamicDet/detect.py \
        --img-size 640 \
        --cfg DynamicDet/cfg/dy-yolov7-step2.yaml \
        --weight path/to/model/weights.pt \
        --source /path/to/denoised/output \
        --dy-thres variable-threshold \
        --num-classes 1 \
        --device 0 \
        --save-txt --save-conf --name detect-output




