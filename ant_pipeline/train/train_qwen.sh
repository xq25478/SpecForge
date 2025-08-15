#!/bin/bash
set -ex

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")/../
TIMESTAMP=$(date +%Y%m%d_%H%M%S)  # 时间戳用于日志和输出


# 如需获取 Debug 信息，请打开以下变量。
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG=INFO

# 8 机 8 卡 Qwen3-32B 训练配置
# tp_size = 1 dp_size = 64
# bs = 8 global_bs = 8*64 = 512

# 参数校验
validate_arguments() {
    if [[ -z "$1" || -z "$2" || -z "$3" ]]; then
        echo "Usage: $0 <nnodes> <nproc_per_node> <model_tag>"
        exit 1
    fi
}

# 环境检查
check_environment() {
    if ! command -v torchrun &>/dev/null; then
        echo "Error: torchrun not found. Please install PyTorch first."
        exit 1
    fi

    if [[ -z "$MASTER_ADDR" || -z "$MASTER_PORT" ]]; then
        echo "Error: MASTER_ADDR and MASTER_PORT must be set for distributed training"
        exit 1
    fi

    if [[ -z "$RANK" ]]; then
        echo "Warning: RANK not set, assuming single-node training"
        export RANK=0
    fi
}

# 清理函数
cleanup() {
    echo "Cleaning up..."
    # 添加必要的清理逻辑
}

main() {
    validate_arguments "$@"

    local nnodes=$1
    local nproc_per_node=$2
    local model_tag=$3

    # 路径配置
    local output_dir="${ROOT_DIR}/outputs/Qwen${model_tag}B-eagle3_${TIMESTAMP}"
    local log_dir="${ROOT_DIR}/logs"
    local log_path="${log_dir}/eagle3_qwen${model_tag}b_rank${RANK}_${TIMESTAMP}.log"
    local log_nccl_path="${log_dir}/eagle3_qwen${model_tag}b_rank${RANK}_nccl_debug_${TIMESTAMP}.log"

    local target_model_path=/mnt/modelops/models/Qwen${model_tag}B

    # 创建目录
    mkdir -p "$output_dir" "$log_dir"
    
    echo "Starting training with config:"
    echo "Nodes: $nnodes, GPUs per node: $nproc_per_node"
    echo "Model: Qwen${model_tag}B, Output dir: $output_dir"
    echo "Log file: $log_path"

    # 分布式训练通信测试
    torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /mnt/modelops/460695/eagle3/dist_test.py 2>&1 | tee ${log_nccl_path}

    # 正式开启训练
    # \cp -r $target_model_path /Qwen${model_tag}B
    # target_model_path=/Qwen${model_tag}B

    # 开训
    OMP_NUM_THREADS=8 \
    NCCL_NET=IB \
    NCCL_IB_DISABLE=0 \
    NCCL_SOCKET_IFNAME=^docker0,lo,eth0 \
    NCCL_IB_GID_INDEX=3 \
    NCCL_DEBUG=INFO \
    torchrun \
        --nnodes=$nnodes \
        --nproc_per_node=$nproc_per_node \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $ROOT_DIR/scripts/train_eagle3_online.py \
        --target-model-path ${target_model_path} \
        --draft-model-config $ROOT_DIR/configs/qwen${model_tag}b-eagle3.json \
        --train-data-path /mnt/modelops/datasets/specforge_postprocess_ultrachat/ultrachat.jsonl \
        --output-dir $output_dir \
        --num-epochs 10 \
        --batch-size 1 \
        --learning-rate 1e-4 \
        --max-length 2048 \
        --chat-template qwen \
        --cache-dir $ROOT_DIR/cache \
        --embedding-key model.embed_tokens.weight \
        --ttt-length 7 \
        --target-model-tp-size 8 \
        --draft-model-tp-size 1 \
        2>&1 | tee $log_path

    echo "Training completed successfully. Output saved to $output_dir"
}

main "$@"