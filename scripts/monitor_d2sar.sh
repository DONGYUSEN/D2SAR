#!/bin/bash
set -e

CONTAINER_NAME="d2sar_monitor"
CUDA_IMAGE="d2sar:cuda"
MANIFEST1="/results/20231110/manifest.json"
MANIFEST2="/results/20231121/manifest.json"
OUTPUT_ROOT="/results2"
SCRIPT_PATH="/work/scripts/strip_insar2.py"
LOG_FILE="/tmp/d2sar_monitor.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_docker_memory() {
    docker stats --no-stream --format "{{.MemUsage}}" "$1" 2>/dev/null | awk '{print $1}' || echo "N/A"
}

log "开始监控 d2sar: cuda 容器..."

if ! docker ps -q --filter "ancestor=$CUDA_IMAGE" | grep -q .; then
    log "错误: 未找到运行中的 d2sar: cuda 容器"
    exit 1
fi

while true; do
    container_id=$(docker ps -q --filter "ancestor=$CUDA_IMAGE" 2>/dev/null)

    if [ -z "$container_id" ]; then
        log "检测到 d2sar: cuda 容器已停止!"
        sleep 5

        if docker ps -q --filter "ancestor=$CUDA_IMAGE" | grep -q .; then
            log "容器仍在运行，继续监控..."
            sleep 60
            continue
        fi

        log "启动 CPU 模式..."

        docker run --rm \
            -v /home/ysdong/Software/D2SAR:/work \
            -v /home/ysdong/Software/D2SAR/results:/results \
            -v /home/ysdong/Temp/:/temp \
            d2sar:cuda \
            python3 $SCRIPT_PATH \
            $MANIFEST1 $MANIFEST2 \
            --gpu-model cpu \
            --output-root $OUTPUT_ROOT 2>&1 | tee -a "$LOG_FILE"

        EXIT_CODE=${PIPESTATUS[0]}
        log "CPU 模式结束，退出码: $EXIT_CODE"
        exit $EXIT_CODE
    fi

    mem_usage=$(check_docker_memory "$container_id")
    log "容器运行中 | 内存: $mem_usage"
    sleep 60
done