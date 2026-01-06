#!/bin/bash
TYPE=${1:-"latency"}
DURATION=${2:-60}
INTERFACE="eno1"
LOG_FILE="./fault_injection_log.csv"

# 헤더 없으면 추가
if [ ! -f "$LOG_FILE" ]; then
    echo "fault_type,start_timestamp,start_datetime,end_timestamp,end_datetime,duration,node" > $LOG_FILE
fi

HOST=$(hostname)
START_TS=$(date +%s.%N)
START_DT=$(date '+%Y-%m-%d %H:%M:%S')

# 기존 규칙 초기화
sudo tc qdisc del dev $INTERFACE root 2>/dev/null

echo "=== Starting network $TYPE fault for ${DURATION}s on $HOST ==="
echo "Start: $START_DT"

case $TYPE in
latency)
    sudo tc qdisc add dev $INTERFACE root netem delay 200ms 50ms
    echo "Added 200ms latency (±50ms jitter)"
    sleep $DURATION
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null
    ;;
loss)
    sudo tc qdisc add dev $INTERFACE root netem loss 10%
    echo "Added 10% packet loss"
    sleep $DURATION
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null
    ;;
bandwidth)
    sudo tc qdisc add dev $INTERFACE root tbf rate 50mbit burst 32kbit latency 400ms
    echo "Limited bandwidth to 50Mbps"
    sleep $DURATION
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null
    ;;
latency_loss)
    sudo tc qdisc add dev $INTERFACE root netem delay 100ms 30ms loss 5%
    echo "Added 100ms latency + 5% loss"
    sleep $DURATION
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null
    ;;
corruption)
    sudo tc qdisc add dev $INTERFACE root netem corrupt 5%
    echo "Added 5% packet corruption (bit errors)"
    sleep $DURATION
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null
    ;;
link_down)
    # 5초만 다운 (NCCL timeout 전에 복구)
    sudo ip link set $INTERFACE down
    echo "Link down for 5 seconds"
    sleep 5
    sudo ip link set $INTERFACE up
    echo "Link restored"
    ;;
flapping)
    echo "Link flapping - 5s down / 10s up cycle"
    END_TIME=$(($(date +%s) + $DURATION))
    while [ $(date +%s) -lt $END_TIME ]; do
        sudo ip link set $INTERFACE down
        echo "[$(date '+%H:%M:%S')] Link DOWN"
        sleep 5
        sudo ip link set $INTERFACE up
        echo "[$(date '+%H:%M:%S')] Link UP"
        sleep 10
        if [ $(date +%s) -ge $END_TIME ]; then break; fi
    done
    sudo ip link set $INTERFACE up
    echo "Flapping ended - link restored"
    ;;
mtu_small)
    ORIG_MTU=$(cat /sys/class/net/$INTERFACE/mtu)
    sudo ip link set $INTERFACE mtu 576
    echo "MTU reduced to 576 (was $ORIG_MTU)"
    sleep $DURATION
    sudo ip link set $INTERFACE mtu $ORIG_MTU
    echo "MTU restored to $ORIG_MTU"
    ;;
tcp_buffer)
    # 원본 저장
    ORIG_RMEM_MAX=$(sysctl -n net.core.rmem_max)
    ORIG_WMEM_MAX=$(sysctl -n net.core.wmem_max)
    ORIG_TCP_RMEM=$(sysctl -n net.ipv4.tcp_rmem)
    ORIG_TCP_WMEM=$(sysctl -n net.ipv4.tcp_wmem)
    echo "Original: rmem_max=$ORIG_RMEM_MAX, wmem_max=$ORIG_WMEM_MAX"
    
    # 64KB로 축소
    sudo sysctl -w net.core.rmem_max=65536
    sudo sysctl -w net.core.wmem_max=65536
    sudo sysctl -w net.ipv4.tcp_rmem="4096 16384 65536"
    sudo sysctl -w net.ipv4.tcp_wmem="4096 16384 65536"
    echo "TCP buffers reduced to 64KB"
    
    sleep $DURATION
    
    # 복구
    sudo sysctl -w net.core.rmem_max=$ORIG_RMEM_MAX
    sudo sysctl -w net.core.wmem_max=$ORIG_WMEM_MAX
    sudo sysctl -w net.ipv4.tcp_rmem="$ORIG_TCP_RMEM"
    sudo sysctl -w net.ipv4.tcp_wmem="$ORIG_TCP_WMEM"
    echo "TCP buffers restored"
    ;;
reset)
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null
    sudo ip link set $INTERFACE up 2>/dev/null
    sudo iptables -F 2>/dev/null
    echo "[$START_DT] Reset network to normal"
    echo "network_reset,$START_TS,$START_DT,,,$HOST" >> $LOG_FILE
    exit 0
    ;;
show)
    echo "=== TC rules ==="
    tc qdisc show dev $INTERFACE
    echo "=== Interface status ==="
    ip link show $INTERFACE
    echo "=== MTU ==="
    cat /sys/class/net/$INTERFACE/mtu
    echo "=== TCP buffers ==="
    sysctl net.core.rmem_max net.core.wmem_max net.ipv4.tcp_rmem net.ipv4.tcp_wmem
    echo "=== IPTables ==="
    sudo iptables -L -n
    exit 0
    ;;
*)
    echo "Unknown type: $TYPE"
    echo "Available: latency, loss, bandwidth, latency_loss, corruption, link_down, flapping, mtu_small, tcp_buffer, reset, show"
    exit 1
    ;;
esac

END_TS=$(date +%s.%N)
END_DT=$(date '+%Y-%m-%d %H:%M:%S')
echo "End: $END_DT"
echo "network_$TYPE,$START_TS,$START_DT,$END_TS,$END_DT,$DURATION,$HOST" >> $LOG_FILE
echo "=== Network reset to normal ==="