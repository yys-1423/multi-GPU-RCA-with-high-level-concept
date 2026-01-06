#!/bin/bash
# run_all_fault_scenarios.sh

INTERVAL=180  # 3분 = 180초
FAULT_DURATION=120  # 각 장애 지속 시간
LOG_FILE="./fault_scenario_master_log.csv"
HOST=$(hostname)

echo "fault_type,start_timestamp,start_datetime,end_timestamp,end_datetime,node" > $LOG_FILE

log_event() {
    local type=$1
    local start_ts=$2
    local start_dt=$3
    local end_ts=$4
    local end_dt=$5
    echo "$type,$start_ts,$start_dt,$end_ts,$end_dt,$HOST" >> $LOG_FILE
}

run_interference() {
    local type=$1
    local extra=$2
    
    START_TS=$(date +%s.%N)
    START_DT=$(date '+%Y-%m-%d %H:%M:%S')
    echo "=== [$START_DT] Starting: $type $extra ==="
    
    if [ -z "$extra" ]; then
        ./interference_workloads.sh $type $FAULT_DURATION
    else
        ./interference_workloads.sh $type $FAULT_DURATION $extra
    fi
    
    END_TS=$(date +%s.%N)
    END_DT=$(date '+%Y-%m-%d %H:%M:%S')
    echo "=== [$END_DT] Finished: $type $extra ==="
    log_event "${type}_${extra}" $START_TS "$START_DT" $END_TS "$END_DT"
}

run_network_fault() {
    local type=$1
    
    START_TS=$(date +%s.%N)
    START_DT=$(date '+%Y-%m-%d %H:%M:%S')
    echo "=== [$START_DT] Starting: network_$type ==="
    
    sudo ./network_fault_injection.sh $type $FAULT_DURATION
    
    END_TS=$(date +%s.%N)
    END_DT=$(date '+%Y-%m-%d %H:%M:%S')
    echo "=== [$END_DT] Finished: network_$type ==="
    log_event "network_$type" $START_TS "$START_DT" $END_TS "$END_DT"
}

wait_interval() {
    echo "=== Waiting $INTERVAL seconds (20 min) until next scenario ==="
    sleep $INTERVAL
}

echo "========================================"
echo "Starting all fault scenarios on $HOST"
echo "Interval: ${INTERVAL}s (20 min), Duration: ${FAULT_DURATION}s"
echo "Log: $LOG_FILE"
echo "========================================"

# 1. CPU 간섭
run_interference cpu
wait_interval

# 2. Memory 간섭
run_interference memory
wait_interval

# 3. Disk 간섭
run_interference disk
wait_interval

# 4. GPU 간섭 (노드에 따라 다름)
if [ "$HOST" == "goguma3" ]; then
    for gpu_id in 0 1 2 3; do
        run_interference gpu $gpu_id
        wait_interval
    done
else
    for gpu_id in 0 1 2; do
        run_interference gpu $gpu_id
        wait_interval
    done
fi

# 5. GPU throttle (노드에 따라 다름)
if [ "$HOST" == "goguma3" ]; then
    for gpu_id in 0 1 2 3; do
        run_interference gpu_throttle $gpu_id
        wait_interval
    done
else
    for gpu_id in 0 1 2; do
        run_interference gpu_throttle $gpu_id
        wait_interval
    done
fi

# 6. Network 간섭 (iperf)
run_interference network
wait_interval

# 7. Network latency
run_network_fault latency
wait_interval

# 8. Network packet loss
run_network_fault loss
wait_interval

# 9. Network bandwidth limit
run_network_fault bandwidth
wait_interval

# 10. Network latency + loss
run_network_fault latency_loss
wait_interval

# 11. Network corruption (bit errors)
run_network_fault corruption
wait_interval

# 12. Network link down (5초만)
run_network_fault link_down
wait_interval

# 13. Network flapping (intermittent)
run_network_fault flapping
wait_interval

# 14. Network MTU small (fragmentation)
run_network_fault mtu_small
wait_interval

# 15. Network TCP buffer reduced
run_network_fault tcp_buffer

echo "========================================"
echo "All scenarios completed!"
echo "Log saved to: $LOG_FILE"
echo "========================================"
```

업데이트된 시나리오 목록:
```
goguma3 기준 (4 GPU):
1. cpu                → 20분 대기
2. memory             → 20분 대기
3. disk               → 20분 대기
4. gpu_0              → 20분 대기
5. gpu_1              → 20분 대기
6. gpu_2              → 20분 대기
7. gpu_3              → 20분 대기
8. gpu_throttle_0     → 20분 대기
9. gpu_throttle_1     → 20분 대기
10. gpu_throttle_2    → 20분 대기
11. gpu_throttle_3    → 20분 대기
12. network (iperf)   → 20분 대기
13. latency           → 20분 대기
14. loss              → 20분 대기
15. bandwidth         → 20분 대기
16. latency_loss      → 20분 대기
17. corruption        → 20분 대기
18. link_down (5초)   → 20분 대기
19. flapping          → 20분 대기
20. mtu_small         → 20분 대기
21. tcp_buffer        → 끝

총 시나리오: 21개
총 소요 시간: 약 7시간 20분