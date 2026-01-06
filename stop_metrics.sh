#!/bin/bash
LOG_DIR=$(ls -td ./system_metrics_* | head -1)
if [ -f "$LOG_DIR/pids.txt" ]; then
    kill $(cat $LOG_DIR/pids.txt) 2>/dev/null
    echo "Stopped metric collection"
else
    echo "No pids.txt found"
fi

pkill -f "collect_metrics.sh"