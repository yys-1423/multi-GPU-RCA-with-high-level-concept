#!/bin/bash
# ============================================================
# MASTER NODE - goguma3에서 실행
# ============================================================

# 네트워크 설정
export MASTER_ADDR=143.248.53.131  # goguma3 IP
export MASTER_PORT=29500

# 클러스터 설정
NUM_NODES=2
NODE_RANK=0  # 마스터는 0
GPUS_THIS_NODE=4  # goguma3: 3060 Ti 4개

# NCCL 설정
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eno1  # 안되면 ifconfig로 확인 후 수정
export NCCL_IB_DISABLE=1  # InfiniBand 없으면 1
export NCCL_DEBUG_SUBSYS=COLL,NET,INIT
export NCCL_DEBUG_FILE=/home/sklee/cs540/logs/nccl_%h_%p.txt
export TORCH_NCCL_ENABLE_TIMING=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export GLOO_SOCKET_IFNAME=eno1

# CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "============================================"
echo "MASTER NODE: goguma3"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NUM_NODES: $NUM_NODES"
echo "GPUS_THIS_NODE: $GPUS_THIS_NODE"
echo "============================================"

# conda 환경 활성화 (경로 확인 필요)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_training

# torchrun 실행
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$GPUS_THIS_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_llama.py
