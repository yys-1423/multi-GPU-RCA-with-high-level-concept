#!/bin/bash
#SBATCH --job-name=llama-33b-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ============================================================
# Multi-Node LLaMA 33B Training Script for SLURM
# ============================================================

# Create logs directory
mkdir -p logs

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Distributed training setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))  # nodes * gpus_per_node

echo "=========================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "=========================================="

# Activate conda environment (adjust path as needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_training

# Run training with torchrun (PyTorch distributed launcher)
srun --kill-on-bad-exit=1 \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_llama.py

echo "Training completed!"
