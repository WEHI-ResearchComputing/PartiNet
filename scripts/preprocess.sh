#!/bin/bash

#SBATCH --cpus-per-task 6 
#SBATCH --mem=5G 
#SBATCH --gres=gpu:A30:1 
#SBATCH -p gpuq
#SBATCH --output=/vast/scratch/users/iskander.j/logs/preprocess_%A_%a.out 
#SBATCH --job-name=denoise
#SBATCH --array=1-3

#module load miniconda3/latest
#conda activate /vast/projects/miti2324/envs/topaz_env/
module load topaz

dataset_name=$(sed -n "$SLURM_ARRAY_TASK_ID"p meta/datasets_test.txt)
dataset_path=/vast/scratch/users/iskander.j/PartiNet_data/testing/${dataset_name}
output_dir=${dataset_path}/denoised_micrographs/jpg
mkdir -p $output_dir

echo "Denoising...."
topaz denoise ${dataset_path}/micrographs/*.mrc -o $output_dir --format "jpg"
echo "Denoising finished."
echo "Generating bounding box...."
./preprocess.py --dataset ${dataset_name} --datasets_path /vast/scratch/users/iskander.j/PartiNet_data/testing/ --tag _test
echo "Done."



