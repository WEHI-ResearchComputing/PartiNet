#!/bin/bash

#SBATCH --cpus-per-task 16 --mem=499G --gres=gpu:A100:1 -p gpuq
#SBATCH --output=train_step2.out --job-name=train_step2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aggarwal.m@wehi.edu.au


module load miniconda3/latest
conda activate /vast/projects/miti2324/envs/dynamicdet_mamba37
python /vast/projects/miti2324/cryo_em_dynamic_det/DynamicDet/train_step2.py \
    --workers 4 --device 0 --batch-size 1 --epochs 10 --img 640 --adam \
    --cfg DynamicDet/cfg/dy-yolov7-step2.yaml \
    --weight ./runs/train/train_step1/weights/epoch_049.pt \
    --data /vast/scratch/users/jain.o/data/cryo_training_22/cryo_training.yaml \
    --hyp DynamicDet/hyp/hyp.finetune.dynamic.adam.yaml \
    --name train_step2_epoch_049
