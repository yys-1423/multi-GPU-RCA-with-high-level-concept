#!/bin/bash
# ============================================================
# WORKER NODE - goguma4에서 실행
# ============================================================

# 네트워크 설정 (마스터 주소)
export MASTER_ADDR=143.248.53.131  # goguma3 IP
export MASTER_PORT=29500

# 클러스터 설정
NUM_NODES=2
NODE_RANK=1  # 워커는 1
GPUS_THIS_NODE=2  # goguma4: 2080 Ti 2개만 사용

# NCCL 설정
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 안되면 ifconfig로 확인 후 수정
export NCCL_IB_DISABLE=1  # InfiniBand 없으면 1

# CUDA - 2080 Ti만 사용 (4090 제외)
export CUDA_VISIBLE_DEVICES=0,1  # GPU 0,1만 (4090은 GPU 2)

echo "============================================"
echo "WORKER NODE: goguma4"
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
