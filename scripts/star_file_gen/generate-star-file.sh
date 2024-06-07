#!/bin/bash

#SBATCH --cpus-per-task 16 --mem=499G --gres=gpu:A30:4 -p gpuq
#SBATCH --output=generate-star-file.out --job-name=generate-star-file.py

outputLabels = 'path/to/labels'
denoisedMRC = 'path/to/denoised/mrc'
starFile = 'path/to/star/file'
conf_thresh = 0.5 #default value


module load miniconda3/latest
conda activate /vast/projects/miti2324/envs/dynamicdet_mamba37/

python generate-star-file.py --outputLabels=$outputLabels --denoisedMRC=$denoisedMRC --starFile=$starFile --conf_thresh=$conf_thresh






