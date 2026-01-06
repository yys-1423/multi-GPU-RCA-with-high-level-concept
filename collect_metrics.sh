#!/bin/bash
LOG_DIR="./system_metrics_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR
HOST=$(hostname)

echo "=== Starting metric collection on $HOST ==="
echo "Log dir: $LOG_DIR"

# 1. GPU metrics (1초마다)
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1 > $LOG_DIR/gpu_${HOST}.csv &
PID_GPU=$!

# 2. Network throughput (eno1)
sar -n DEV 1 > $LOG_DIR/network_${HOST}.txt &
PID_NET=$!

# 3. TCP retransmissions (패킷 손실 지표)
(while true; do
    ts=$(date +%s.%N)
    ts_readable=$(date '+%Y-%m-%d %H:%M:%S')
    retrans=$(cat /proc/net/snmp | grep "^Tcp:" | tail -1 | awk '{print $13}')
    echo "$ts,$ts_readable,$retrans"
    sleep 1
done) > $LOG_DIR/tcp_retrans_${HOST}.csv &
PID_TCP=$!

# 4. CPU + Memory
(while true; do
    ts=$(date +%s.%N)
    ts_readable=$(date '+%Y-%m-%d %H:%M:%S')
    cpu=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
    mem_used=$(free -m | grep Mem | awk '{print $3}')
    mem_total=$(free -m | grep Mem | awk '{print $2}')
    swap=$(free -m | grep Swap | awk '{print $3}')
    echo "$ts,$ts_readable,$cpu,$mem_used,$mem_total,$swap"
    sleep 1
done) > $LOG_DIR/cpu_mem_${HOST}.csv &
PID_CPU=$!

# 5. 모든 프로세스 리소스 사용량
(while true; do
    ts=$(date +%s.%N)
    ts_readable=$(date '+%Y-%m-%d %H:%M:%S')
    ps aux --sort=-%cpu | head -10 | tail -9 | while read line; do
        user=$(echo $line | awk '{print $1}')
        pid=$(echo $line | awk '{print $2}')
        cpu=$(echo $line | awk '{print $3}')
        mem=$(echo $line | awk '{print $4}')
        cmd=$(echo $line | awk '{print $11}')
        echo "$ts,$ts_readable,$pid,$user,$cpu,$mem,$cmd"
    done
    sleep 2
done) > $LOG_DIR/top_processes_${HOST}.csv &
PID_PS=$!

# 6. 노드 간 ping latency (goguma3에서만)
if [ "$HOST" == "goguma3" ]; then
    ping -i 0.5 143.248.53.206 | while read line; do
        ts=$(date +%s.%N)
        ts_readable=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$ts,$ts_readable,$line"
    done > $LOG_DIR/ping_latency.csv &
    PID_PING=$!
fi

# 7. 네트워크 에러/드롭 패킷
(while true; do
    ts=$(date +%s.%N)
    ts_readable=$(date '+%Y-%m-%d %H:%M:%S')
    rx_dropped=$(cat /sys/class/net/eno1/statistics/rx_dropped)
    tx_dropped=$(cat /sys/class/net/eno1/statistics/tx_dropped)
    rx_errors=$(cat /sys/class/net/eno1/statistics/rx_errors)
    tx_errors=$(cat /sys/class/net/eno1/statistics/tx_errors)
    echo "$ts,$ts_readable,$rx_dropped,$tx_dropped,$rx_errors,$tx_errors"
    sleep 1
done) > $LOG_DIR/net_errors_${HOST}.csv &
PID_NETERR=$!

# 8. 커널 메시지
dmesg -w --time-format iso > $LOG_DIR/dmesg_${HOST}.log &
PID_DMESG=$!

# 9. 네트워크 설정 (MTU, TCP buffer, 링크 상태)
(while true; do
    ts=$(date +%s.%N)
    ts_readable=$(date '+%Y-%m-%d %H:%M:%S')
    mtu=$(cat /sys/class/net/eno1/mtu)
    link_state=$(cat /sys/class/net/eno1/operstate)
    rmem_max=$(sysctl -n net.core.rmem_max)
    wmem_max=$(sysctl -n net.core.wmem_max)
    echo "$ts,$ts_readable,$mtu,$link_state,$rmem_max,$wmem_max"
    sleep 1
done) > $LOG_DIR/net_config_${HOST}.csv &
PID_NETCFG=$!

# 10. GPU 파워/온도 상세 (throttle 감지용)
(while true; do
    ts=$(date +%s.%N)
    ts_readable=$(date '+%Y-%m-%d %H:%M:%S')
    nvidia-smi --query-gpu=index,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader,nounits | while read line; do
        echo "$ts,$ts_readable,$line"
    done
    sleep 1
done) > $LOG_DIR/gpu_power_${HOST}.csv &
PID_GPUPWR=$!

# 11. Disk I/O
(while true; do
    ts=$(date +%s.%N)
    ts_readable=$(date '+%Y-%m-%d %H:%M:%S')
    iostat -x 1 1 | grep -E "^sd|^nvme" | while read line; do
        echo "$ts,$ts_readable,$line"
    done
    sleep 1
done) > $LOG_DIR/disk_io_${HOST}.csv &
PID_DISK=$!

# 12. GPU 프로세스별 사용량 (pmon: gpu, pid, type, sm%, mem%, enc%, dec%, command)
nvidia-smi pmon -s um -d 1 -o T > $LOG_DIR/gpu_pmon_${HOST}.csv &
PID_GPUPROC=$!

# 13. GPU 프로세스 메모리 상세
(while true; do
    ts=$(date +%s.%N)
    ts_readable=$(date '+%Y-%m-%d %H:%M:%S')
    nvidia-smi --query-compute-apps=pid,gpu_name,used_memory --format=csv,noheader,nounits 2>/dev/null | while read line; do
        echo "$ts,$ts_readable,$line"
    done
    sleep 1
done) > $LOG_DIR/gpu_proc_mem_${HOST}.csv &
PID_GPUMEM=$!

# PID 저장
if [ "$HOST" == "goguma3" ]; then
    echo "$PID_GPU $PID_NET $PID_TCP $PID_CPU $PID_PS $PID_PING $PID_NETERR $PID_DMESG $PID_NETCFG $PID_GPUPWR $PID_DISK $PID_GPUPROC $PID_GPUMEM" > $LOG_DIR/pids.txt
else
    echo "$PID_GPU $PID_NET $PID_TCP $PID_CPU $PID_PS $PID_NETERR $PID_DMESG $PID_NETCFG $PID_GPUPWR $PID_DISK $PID_GPUPROC $PID_GPUMEM" > $LOG_DIR/pids.txt
fi

echo "=== Collection started ==="
echo "To stop: kill \$(cat $LOG_DIR/pids.txt)"
echo "PIDs:"
echo "  GPU=$PID_GPU NET=$PID_NET TCP=$PID_TCP CPU=$PID_CPU"
echo "  PS=$PID_PS NETERR=$PID_NETERR DMESG=$PID_DMESG"
echo "  NETCFG=$PID_NETCFG GPUPWR=$PID_GPUPWR DISK=$PID_DISK"
echo "  GPUPROC=$PID_GPUPROC GPUMEM=$PID_GPUMEM"
if [ "$HOST" == "goguma3" ]; then
    echo "  PING=$PID_PING"
fi
