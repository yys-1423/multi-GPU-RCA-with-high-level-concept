#!/bin/bash
# ============================================================
# Manual Multi-Node Launch Script
# Run this on each node with appropriate RANK
# ============================================================

# Configuration - MODIFY THESE
MASTER_ADDR="192.168.1.100"  # IP of node 0
MASTER_PORT=29500
NUM_NODES=2
GPUS_PER_NODE=8
NODE_RANK=${1:-0}  # Pass node rank as argument: ./run_node.sh 0, ./run_node.sh 1

# Calculate world size
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "=========================================="
echo "Starting training on Node $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "GPUs per node: $GPUS_PER_NODE"
echo "=========================================="

# Environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Adjust to your network interface
export NCCL_IB_DISABLE=0        # Enable InfiniBand if available

# Option 1: Using torchrun (recommended)
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_llama.py

# Option 2: Using accelerate (alternative)
# accelerate launch \
#     --num_processes=$WORLD_SIZE \
#     --num_machines=$NUM_NODES \
#     --machine_rank=$NODE_RANK \
#     --main_process_ip=$MASTER_ADDR \
#     --main_process_port=$MASTER_PORT \
#     --config_file=accelerate_config.yaml \
#     train_llama.py
