#!/bin/bash

# GPU监控脚本：每2秒记录一次GPU使用情况到gpu_usage0.txt

OUTPUT_FILE="/home/ysdong/Software/D2SAR/gpu_usage0.txt"

# 创建输出文件并添加时间戳
> "$OUTPUT_FILE"
echo "GPU使用监控记录" >> "$OUTPUT_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "监控间隔: 2秒" >> "$OUTPUT_FILE"
echo "==============================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 循环记录GPU使用情况
echo "开始监控GPU使用情况（按 Ctrl+C 停止）..."
while true; do
    echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$OUTPUT_FILE"
    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    sleep 2
done
