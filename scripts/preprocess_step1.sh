#!/bin/bash

#SBATCH --cpus-per-task 6 
#SBATCH --mem=5G 
#SBATCH --gres=gpu:A30:1 
## SBATCH --qos=bonus  #uncomment to use bonus qos
#SBATCH -p gpuq
#SBATCH --output=/vast/scratch/users/iskander.j/logs/denoise_%A_%a.out 
#SBATCH --job-name=denoise
#SBATCH --array=1-35

module load topaz

dataset_name=$(sed -n "$SLURM_ARRAY_TASK_ID"p meta/datasets.txt)
dataset_path=/vast/scratch/users/iskander.j/PartiNet_data
output_dir=${dataset_path}/raw/${dataset_name}/denoised_micrographs/jpg
mkdir -p $output_dir

echo "Denoising...."
topaz denoise ${dataset_path}/raw/${dataset_name}/micrographs/*.mrc -o $output_dir --format "jpg"
echo "Denoising finished."
echo "Done."