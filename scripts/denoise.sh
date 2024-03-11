#!/bin/bash

#SBATCH --cpus-per-task 16 --mem=499G --gres=gpu:A30:4 -p gpuq
#SBATCH --output=denoise.out --job-name=denoise

module load miniconda3/latest
conda activate /vast/projects/miti2324/envs/topaz_env/

mkdir -p /output/path

topaz denoise /path/to/your/micrographs/*.mrc -o /output/path --format "jpg"





