#!/bin/bash
set -ex

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")/../

# 参数校验
# $1: nnodes: 节点数 $2: 单机 GPU 数量, $3: model_tag 训练模型 3-32 / 3-8 / 2.5-72
validate_arguments() {
    if [[ -z "$1" || -z "$2" || -z "$3" ]]; then
        echo "Usage: $0 <nnodes> <nproc_per_node> <model_tag>"
        exit 1
    fi
}

main() {
    validate_arguments "$@"

    local nnodes=$1
    local nproc_per_node=$2
    local model_tag=$3

    # 路径配置
    local SAVE_PATH=/mnt/modelops/train/eagle/460695/
    local output_dir=${SAVE_PATH}/outputs/Qwen${model_tag}B-eagle3_nnodes_${nnodes}
    local log_dir=${SAVE_PATH}/logs
    local base_log_prefix=${log_dir}/eagle3_qwen${model_tag}b_nnodes_${nnodes}_rank${RANK}
    local log_path=${base_log_prefix}_train.log
    local log_nccl_path=${base_log_prefix}_nccl.log
    local log_nvidia_msi_path=${base_log_prefix}_nvidia_smi.log
    local target_model_path=/mnt/modelops/models/Qwen${model_tag}B

    # 创建目录
    mkdir -p $output_dir $log_dir
    
    echo "Starting training with config:"
    echo "Nodes: $nnodes, GPUs per node: $nproc_per_node"
    echo "Model: Qwen${model_tag}B, Output dir: $output_dir"
    echo "Log file: $log_path"

    if [ $nnodes -eq 1 ]; then
        MASTER_ADDR=localhost
        MASTER_PORT=29500
        RANK=0
    else 
        export OMP_NUM_THREADS=8
        export NCCL_NET=IB
        export NCCL_IB_DISABLE=0
        export NCCL_SOCKET_IFNAME=^docker0,lo,eth0
        export NCCL_IB_GID_INDEX=3
        export NCCL_DEBUG=INFO
    fi
    
    # 分布式训练通信测试
    torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /mnt/modelops/460695/eagle3/dist_test.py 2>&1 | tee ${log_nccl_path}

    # 添加GPU 监控
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv -l 1 > ${log_nvidia_msi_path} &

    # 正式开启训练
    torchrun \
        --nnodes=$nnodes \
        --nproc_per_node=$nproc_per_node \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $ROOT_DIR/scripts/train_eagle3_online_deepspeed.py \
        --target-model-path ${target_model_path} \
        --draft-model-config $ROOT_DIR/configs/qwen${model_tag}b-eagle3.json \
        --train-data-path /mnt/modelops/datasets/specforge_postprocess_ultrachat/ultrachat.jsonl \
        --output-dir $output_dir \
        --num-epochs 10 \
        --batch-size 2 \
        --learning-rate 1e-4 \
        --max-length 2048 \
        --chat-template qwen \
        --cache-dir $ROOT_DIR/cache \
        --embedding-key model.embed_tokens.weight \
        --ttt-length 7 \
        2>&1 | tee $log_path

    echo "Training completed successfully. Output saved to $output_dir"
}

main "$@"