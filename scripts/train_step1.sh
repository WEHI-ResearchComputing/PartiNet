#!/bin/bash

#SBATCH --job-name=train_step1
#SBATCH --time 100:00:00
#SBATCH --cpus-per-task=20
#SBATCH  --mem 400G
#SBATCH --nodes=2
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:4 
#SBATCH --output=train_step1.out
#SBATCH --qos=bonus

export MASTER_ADDR=$(hostname -I | grep -o '10.11.\w*.\w*') MASTER_PORT=29500
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=bond1.1330 NCCL_IB_DISABLE=1

module load miniconda3/latest
conda activate /vast/projects/miti2324/envs/dynamicdet_mamba37

srun torchrun --rdzv_id $RANDOM --rdzv_backend c10d \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc_per_node 4 --nnodes 2 \
    DynamicDet/train_step1.py \
    --workers 16 --device 0,1,2,3 --sync-bn --batch-size 16 \
    --epochs 100 --img 640 \
    --cfg /vast/projects/miti2324/cryo_em_dynamic_det/DynamicDet/cfg/dy-yolov7-step1.yaml \
    --weight '' \
    --data /path/to/cryo_training.yaml \
    --hyp /vast/projects/miti2324/cryo_em_dynamic_det/DynamicDet/hyp/hyp.scratch.p5.yaml \
    --name train_step1 --save_period 10
