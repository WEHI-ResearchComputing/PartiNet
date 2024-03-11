#!/bin/bash

#SBATCH --cpus-per-task 16 --mem=499G --gres=gpu:A100:2 -p gpuq --qos=bonus
#SBATCH --output=test.out --job-name=test

module load miniconda3/latest
conda activate /vast/projects/miti2324/envs/dynamicdet_mamba37

python /vast/projects/miti2324/cryo_em_dynamic_det/DynamicDet/test.py \
        --img-size 640 --batch-size 1 --device 0 \
        --cfg /vast/projects/miti2324/cryo_em_dynamic_det/DynamicDet/cfg/dy-yolov7-step2.yaml \
        --weight /path/to/train_step2/last.pt \
        --data /path/to/test.yaml \
        --dy-thres 0 --task test --save-txt --save-conf \
        --name output_folder_name
        
        
        