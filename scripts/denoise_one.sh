#!/bin/bash

#SBATCH --cpus-per-task 6 
#SBATCH --mem=5G 
#SBATCH --gres=gpu:A30:1 
#SBATCH -p gpuq
#SBATCH --output=/vast/scratch/users/iskander.j/denoise_%j.out 
#SBATCH --job-name=denoise

#module load miniconda3/latest
#conda activate /vast/projects/miti2324/envs/topaz_env/
module load topaz

dataset_name=$1
dataset_path=/vast/scratch/users/iskander.j/PartiNet_data/${dataset_name}
output_dir=${dataset_path}/denoised_micrographs/jpg
mkdir -p $output_dir

topaz denoise ${dataset_path}/micrographs/*.mrc -o $output_dir --format "jpg"





