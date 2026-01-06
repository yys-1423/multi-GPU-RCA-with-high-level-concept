#!/bin/bash
TYPE=${1:-"cpu"}
DURATION=${2:-60}
GPU_ID=${3:-0}
LOG_FILE="./fault_injection_log.csv"

# 헤더 없으면 추가
if [ ! -f "$LOG_FILE" ]; then
    echo "fault_type,start_timestamp,start_datetime,end_timestamp,end_datetime,duration,node,extra" > $LOG_FILE
fi

HOST=$(hostname)
START_TS=$(date +%s.%N)
START_DT=$(date '+%Y-%m-%d %H:%M:%S')
echo "=== Starting $TYPE interference for ${DURATION}s on $HOST ==="
echo "Start: $START_DT"

case $TYPE in
cpu)
    stress-ng --cpu 0 --timeout ${DURATION}s
    EXTRA=""
    ;;
gpu)
    echo "Targeting GPU $GPU_ID with maximum compute stress"
    python3 -c "
import torch
import time

gpu_id = $GPU_ID
duration = $DURATION
device = f'cuda:{gpu_id}'

print(f'GPU {gpu_id}: Maximum compute stress mode')

# 다양한 크기의 행렬 (캐시 thrashing 유도)
sizes = [512, 1024, 2048]
num_streams = 16

tensors = []
for s in sizes:
    for _ in range(num_streams // len(sizes)):
        tensors.append((torch.randn(s, s, device=device),
                        torch.randn(s, s, device=device)))

streams = [torch.cuda.Stream(device=device) for _ in range(len(tensors))]

print(f'Using {len(streams)} streams with sizes {sizes}')

end = time.time() + duration
count = 0
while time.time() < end:
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            a, b = tensors[i]
            c = torch.matmul(a, b)
            for _ in range(10):
                c = torch.matmul(c, a)
                c = torch.matmul(c, b)
                c = torch.add(c, a)
                c = torch.mul(c, b)
    count += 1

torch.cuda.synchronize()
print(f'Completed {count} iterations')
"
    EXTRA="gpu_$GPU_ID"
    ;;
gpu_throttle)
    echo "Throttling GPU $GPU_ID (power limit)"
    DEFAULT_PL=$(nvidia-smi -i $GPU_ID --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
    THROTTLE_PL=$(echo "$DEFAULT_PL * 0.5" | bc | cut -d'.' -f1)
    echo "Reducing power from ${DEFAULT_PL}W to ${THROTTLE_PL}W"
    sudo nvidia-smi -i $GPU_ID -pl $THROTTLE_PL
    sleep $DURATION
    sudo nvidia-smi -i $GPU_ID -pl $DEFAULT_PL
    echo "GPU $GPU_ID power restored to ${DEFAULT_PL}W"
    EXTRA="gpu_${GPU_ID}_throttle"
    ;;
network)
    iperf3 -c 143.248.53.206 -t $DURATION -p 5201 -P 4
    EXTRA=""
    ;;
disk)
    timeout $DURATION dd if=/dev/zero of=/tmp/stress_test bs=1M count=10000 conv=fdatasync
    rm -f /tmp/stress_test
    EXTRA=""
    ;;
memory)
    stress-ng --vm 2 --vm-bytes 8G --timeout ${DURATION}s
    EXTRA=""
    ;;
*)
    echo "Unknown type: $TYPE"
    echo "Available: cpu, gpu, gpu_throttle, network, disk, memory"
    exit 1
    ;;
esac

END_TS=$(date +%s.%N)
END_DT=$(date '+%Y-%m-%d %H:%M:%S')
echo "End: $END_DT"
echo "$TYPE,$START_TS,$START_DT,$END_TS,$END_DT,$DURATION,$HOST,$EXTRA" >> $LOG_FILE
echo "=== Logged to $LOG_FILE ==="