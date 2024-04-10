#!/bin/bash

#SBATCH --cpus-per-task 6 
#SBATCH --mem=5G 
#SBATCH -p regular
#SBATCH --output=/vast/scratch/users/iskander.j/download_%a.out 
#SBATCH --job-name=download
#SBATCH --array=1-34%10

module load topaz

dataset_name=$(sed -n "$SLURM_ARRAY_TASK_ID"p datasets.txt)
download_path=/vast/scratch/users/iskander.j/PartiNet_data/raw
##/vast/projects/miti2324/PartiNet_data

mkdir -p $download_path

cd $download_path

wget https://calla.rnet.missouri.edu/cryoppp/${dataset_name}.tar.gz
tar -zxvf ${dataset_name}.tar.gz