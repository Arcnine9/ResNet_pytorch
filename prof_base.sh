#!/bin/bash

# ============================================
# 用法: ./prof_base.sh [训练命令]
# 功能: 监控系统资源，支持手动停止训练进程
# ============================================

# 配置
SAMPLE_INTERVAL=1
OUTPUT_DIR="./monitor_logs"
PREFIX=$(date +%Y%m%d_%H%M%S)
TIMEOUT=0  # 超时时间（秒），0表示不超时
TARGET_NPU=7  # 要监控的NPU编号，默认为0，可以根据需要修改

# 全局变量用于信号处理
TRAIN_PID=""
MONITOR_PID=$$
STOP_REQUESTED=0

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 日志文件路径
CPU_PCT="${OUTPUT_DIR}/${PREFIX}_cpu_mem_pct.log"
CPU_ABS="${OUTPUT_DIR}/${PREFIX}_cpu_mem_abs.log"
GPU_PCT="${OUTPUT_DIR}/${PREFIX}_gpu_mem_pct.log"
GPU_ABS="${OUTPUT_DIR}/${PREFIX}_gpu_mem_abs.log"
SUMMARY="${OUTPUT_DIR}/${PREFIX}_summary.log"

# 清空文件
: > "$CPU_PCT"
: > "$CPU_ABS"
: > "$GPU_PCT"
: > "$GPU_ABS"
: > "$SUMMARY"

# ============================================
# 彻底清理NPU相关进程
# ============================================
cleanup_npu_processes() {
    echo "正在清理NPU相关进程..."
    
    # 获取当前用户的UID
    local USER_ID=$(id -u)
    
    # 1. 首先尝试优雅终止训练进程
    if [ -n "$TRAIN_PID" ] && kill -0 $TRAIN_PID 2>/dev/null; then
        echo "发送SIGTERM信号给训练进程 (PID: $TRAIN_PID)..."
        kill -TERM $TRAIN_PID 2>/dev/null
        sleep 2
    fi
    
    # 2. 查找并终止所有python训练进程（当前用户）
    echo "查找并终止所有python训练进程..."
    pkill -9 -u $USER_ID -f "python.*train.py" 2>/dev/null
    pkill -9 -u $USER_ID -f "python.*main.py" 2>/dev/null
    pkill -9 -u $USER_ID -f "python.*train" 2>/dev/null
    
    # 3. 查找并终止NPU相关进程
    echo "查找并终止NPU相关进程..."
    
    # 终止NPU运行时进程
    pkill -9 -u $USER_ID -f "Ascend910" 2>/dev/null
    pkill -9 -u $USER_ID -f "ascend" 2>/dev/null
    
    # 终止TBE编译器进程
    pkill -9 -u $USER_ID -f "tbe" 2>/dev/null
    
    # 终止AI CPU进程
    pkill -9 -u $USER_ID -f "aicpu" 2>/dev/null
    
    # 4. 使用npu-smi查找并终止NPU上的进程（仅终止目标NPU上的进程）
    if command -v npu-smi &> /dev/null; then
        echo "检查NPU $TARGET_NPU上运行的进程..."
        
        # 获取目标NPU上运行的进程ID
        npu-smi info 2>/dev/null | grep "^| $TARGET_NPU[[:space:]]\+0" | while read line; do
            # 提取进程ID
            proc_pid=$(echo "$line" | grep -o '[0-9]\+' | head -2 | tail -1)
            if [ -n "$proc_pid" ] && [ "$proc_pid" -gt 0 ]; then
                echo "终止NPU $TARGET_NPU上的进程: $proc_pid"
                kill -9 $proc_pid 2>/dev/null
            fi
        done
    fi
    
    # 5. 清理可能残留的共享内存和信号量
    echo "清理IPC资源..."
    ipcs -m 2>/dev/null | grep $USER_ID | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null
    ipcs -s 2>/dev/null | grep $USER_ID | awk '{print $2}' | xargs -r ipcrm -s 2>/dev/null
    
    # 6. 最后再确认一次所有python进程
    sleep 1
    if pgrep -u $USER_ID -f "python" > /dev/null; then
        echo "仍有python进程残留，强制终止所有python进程..."
        pkill -9 -u $USER_ID python 2>/dev/null
    fi
    
    echo "NPU进程清理完成"
}

# ============================================
# 信号处理函数
# ============================================
cleanup() {
    echo ""
    echo "========================================"
    echo "      收到停止信号，正在清理..."
    echo "========================================"
    
    STOP_REQUESTED=1
    
    # 彻底清理NPU相关进程
    cleanup_npu_processes
    
    # 生成最终报告
    generate_summary "INTERRUPTED"
    
    echo "监控数据已保存至: $OUTPUT_DIR"
    echo "========================================"
    
    exit 130
}

# 注册信号处理器
trap cleanup SIGINT SIGTERM EXIT

# ============================================
# 生成汇总报告
# ============================================
generate_summary() {
    local status=${1:-"COMPLETED"}
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    
    {
        echo "========== 监控汇总 =========="
        echo "状态: $status"
        echo "开始时间: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $START_TIME '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo $START_TIME)"
        echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "持续时间: $duration 秒"
        
        if [ -n "$EXIT_CODE" ]; then
            echo "训练退出码: $EXIT_CODE"
        fi
        echo ""
        
        # CPU统计
        if [ -s "$CPU_PCT" ]; then
            echo "--- CPU 内存统计 ---"
            awk '{sum+=$1; count++; max=($1>max?$1:max)} END {
                if(count>0) {
                    printf "采样点数: %d\n", count;
                    printf "平均使用率: %.1f%%\n", sum/count;
                    printf "峰值使用率: %.1f%%\n", max;
                }
            }' "$CPU_PCT"
            
            awk '{sum+=$1; count++; max=($1>max?$1:max)} END {
                if(count>0) {
                    printf "平均使用: %.0f MiB\n", sum/count;
                    printf "峰值使用: %.0f MiB\n", max;
                }
            }' "$CPU_ABS"
            echo ""
        fi
        
        # NPU统计
        if [ -s "$GPU_PCT" ]; then
            echo "--- NPU $TARGET_NPU HBM显存统计 ---"
            awk '{sum+=$1; count++; max=($1>max?$1:max)} END {
                if(count>0) {
                    printf "采样点数: %d\n", count;
                    printf "平均使用率: %.1f%%\n", sum/count;
                    printf "峰值使用率: %.1f%%\n", max;
                }
            }' "$GPU_PCT"
            
            awk '{sum+=$1; count++; max=($1>max?$1:max)} END {
                if(count>0) {
                    printf "平均使用: %.0f MB\n", sum/count;
                    printf "峰值使用: %.0f MB\n", max;
                }
            }' "$GPU_ABS"
            echo ""
        fi
        
        echo "日志文件:"
        echo "  CPU 内存% : $CPU_PCT"
        echo "  CPU 内存MiB: $CPU_ABS"
        echo "  NPU $TARGET_NPU HBM% : $GPU_PCT"
        echo "  NPU $TARGET_NPU HBM MB: $GPU_ABS"
        echo "  汇总报告  : $SUMMARY"
    } >> "$SUMMARY"
    
    cat "$SUMMARY"
}

# ============================================
# 获取指定NPU上python进程的HBM使用量
# ============================================
get_npu_process_memory() {
    local npu_id=$1
    # 直接从npu-smi info输出中提取指定NPU上包含"python"的行，并获取最后一列的数字
    npu-smi info 2>/dev/null | grep "python" | grep "^| $npu_id[[:space:]]\+0" | while read line; do
        # 提取最后一列的数字（进程内存）
        for field in $(echo "$line" | tr '|' ' ' | tr -s ' '); do
            if [[ "$field" =~ ^[0-9]+$ ]]; then
                last_number="$field"
            fi
        done
        
        if [ -n "$last_number" ]; then
            echo "$last_number"
            break
        fi
    done
}

# ============================================
# 资源采样函数
# ============================================
sample_resources() {
    # CPU 内存
    free | awk '/^Mem:/{
        printf "%.1f\n", $3/$2*100
        printf "%d\n", $3/1024
    }' > /tmp/cpu_sample.tmp
    
    # 分别写入两个文件
    head -1 /tmp/cpu_sample.tmp >> "$CPU_PCT"
    tail -1 /tmp/cpu_sample.tmp >> "$CPU_ABS"
    
    # NPU HBM显存 - 从进程表中获取指定NPU上python进程的内存使用
    if command -v npu-smi &> /dev/null; then
        # 获取指定NPU上python进程的HBM使用量
        process_mem=$(get_npu_process_memory $TARGET_NPU)
        
        if [ -n "$process_mem" ] && [ "$process_mem" -gt 0 ]; then
            # 总HBM为65536 MB
            total=65536
            used=$process_mem
            pct=$(echo "scale=2; $used * 100 / $total" | bc 2>/dev/null || echo "0")
            
            echo "$pct" >> "$GPU_PCT"
            echo "$used" >> "$GPU_ABS"
            echo "NPU $TARGET_NPU HBM: ${used}/${total} MB (${pct}%)"
        else
            echo "0.0" >> "$GPU_PCT"
            echo "0" >> "$GPU_ABS"
            echo "NPU $TARGET_NPU HBM: 0 MB (未找到进程)"
        fi
    else
        echo "0.0" >> "$GPU_PCT"
        echo "0" >> "$GPU_ABS"
    fi
}

# ============================================
# 主程序
# ============================================

echo "========================================"
echo "      系统资源监控脚本 (Ascend NPU)"
echo "========================================"
echo "输出目录: $OUTPUT_DIR"
echo "时间戳: $PREFIX"
echo "采样间隔: ${SAMPLE_INTERVAL}秒"
echo "目标NPU: $TARGET_NPU"
echo ""

# 检查NPU环境
if command -v npu-smi &> /dev/null; then
    echo "检测到 Ascend NPU 环境"
    npu_version=$(npu-smi info | head -1 | grep -o "Version:[^,]*" || echo "未知")
    echo "NPU版本: $npu_version"
    
    # 显示当前目标NPU的状态
    echo "当前NPU $TARGET_NPU 状态:"
    npu-smi info | grep -A 1 "^| $TARGET_NPU" | head -2
    echo ""
else
    echo "警告: 未检测到 npu-smi 命令"
fi

# 模式1: 独立运行
if [ $# -eq 0 ]; then
    echo "模式: 独立运行"
    echo "按 Ctrl+C 停止监控"
    echo ""
    
    START_TIME=$(date +%s)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 监控开始" >> "$SUMMARY"
    
    while [ $STOP_REQUESTED -eq 0 ]; do
        sample_resources
        sleep $SAMPLE_INTERVAL
    done

# 模式2: 跟随训练进程
else
    echo "模式: 跟随训练进程"
    echo "训练命令: $@"
    echo "操作提示:"
    echo "  - 按 Ctrl+C 停止训练并生成报告"
    echo "  - 训练正常结束自动生成报告"
    echo ""
    
    START_TIME=$(date +%s)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动训练: $@" >> "$SUMMARY"
    
    # 启动训练
    set -m
    "$@" &
    TRAIN_PID=$!
    set +m
    
    echo "训练进程PID: $TRAIN_PID"
    echo "监控PID: $MONITOR_PID"
    echo ""
    
    # 监控循环
    ELAPSED=0
    while [ $STOP_REQUESTED -eq 0 ]; do
        # 检查训练进程是否还在运行
        if ! kill -0 $TRAIN_PID 2>/dev/null; then
            wait $TRAIN_PID
            EXIT_CODE=$?
            echo ""
            echo "训练进程已正常结束 (退出码: $EXIT_CODE)"
            generate_summary "COMPLETED"
            exit $EXIT_CODE
        fi
        
        # 采样资源
        sample_resources
        
        # 检查超时
        if [ $TIMEOUT -gt 0 ]; then
            ((ELAPSED += SAMPLE_INTERVAL))
            if [ $ELAPSED -ge $TIMEOUT ]; then
                echo ""
                echo "达到超时限制 (${TIMEOUT}秒)，停止训练..."
                cleanup
                exit 124
            fi
        fi
        
        sleep $SAMPLE_INTERVAL
    done
fi