#!/bin/bash

#SBATCH --cpus-per-task 2 
#SBATCH --mem=5G 
#SBATCH -p regular
#SBATCH --output=/vast/scratch/users/iskander.j/logs/preprocess_%A_%a.out 
#SBATCH --job-name=preprocess
#SBATCH --array=1-35

module load topaz

dataset_name=$(sed -n "$SLURM_ARRAY_TASK_ID"p meta/datasets.txt)
dataset_path=/vast/scratch/users/iskander.j/PartiNet_data_Aug2024
#echo "Generating bounding box then splitting datasets into train, val, test...."
#./preprocess.py --dataset ${dataset_name} \
#               --datasets_path ${dataset_path} \
#                --bounding_box

echo "Splitting datasets into train, val, test...."
./preprocess.py --dataset ${dataset_name} \
               --datasets_path ${dataset_path} \

echo "Done."



