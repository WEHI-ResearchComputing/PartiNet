#!/bin/bash

#SBATCH --cpus-per-task 16 --mem=499G --gres=gpu:A100:1 -p gpuq
#SBATCH --output=detect.out --job-name=detect
#SBATCH --mail-type=ALL

module load miniconda3/latest
conda activate /vast/projects/miti2324/envs/dynamicdet_mamba37

python /vast/projects/miti2324/cryo_em_dynamic_det/DynamicDet/detect.py \
        --img-size 640 \
        --cfg /vast/projects/miti2324/cryo_em_dynamic_det/DynamicDet/cfg/dy-yolov7-step2.yaml \
        --weight path/to/model/weights.pt \
        --source /path/to/denoised/output \
        --dy-thres variable-threshold \
        --num-classes 1 --device 0 \
        --save-txt --save-conf --name output_folder_name




