#!/bin/bash
set -ex

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")/../

# 参数校验
# $1: nnodes, $2: nproc_per_node, $3: model_tag
validate_arguments() {
    if [[ -z "$1" || -z "$2" || -z "$3" ]]; then
        echo "Usage: $0 <nnodes> <nproc_per_node> <model_tag> <bs>"
        exit 1
    fi
}

main() {
    # validate_arguments "$@"

    local nnodes=$1
    local nproc_per_node=$2
    local model_tag=3-32
    local bs=2

    # 路径配置
    local SAVE_PATH=/mnt/modelops/train/eagle3/
    local output_dir=${SAVE_PATH}/outputs/QwQ-32B-eagle3_nnodes_${nnodes}_ultrachat_train
    local log_dir=${SAVE_PATH}/logs_1
    local base_log_prefix=${log_dir}/eagle3_QwQ-32b_nnodes_${nnodes}_rank${RANK}
    local log_path=${base_log_prefix}_train.log
    local log_nccl_path=${base_log_prefix}_nccl.log
    local log_nvidia_msi_path=${base_log_prefix}_nvidia_smi.log
    local target_model_path=/mnt/modelops/models/QwQ-32B
    local cache_dir=/mnt/modelops/train/eagle3/cache/QwQ-32B_alpaca

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
    # torchrun \
    # --nnodes=$nnodes \
    # --nproc_per_node=$nproc_per_node \
    # --node_rank=$RANK \
    # --master_addr=$MASTER_ADDR \
    # --master_port=$MASTER_PORT \
    # /mnt/modelops/487922/ant_dev/dist_test.py 2>&1 | tee ${log_nccl_path}

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
        --draft-model-config $ROOT_DIR/configs/qwq-32b-eagle3.json \
        --train-data-path /mnt/modelops/train/eagle/data/online_data/alpaca/online_alpaca.json \
        --output-dir $output_dir \
        --num-epochs 10 \
        --batch-size $bs \
        --learning-rate 1e-4 \
        --max-length 2048 \
        --chat-template qwen \
        --cache-dir $cache_dir \
        --embedding-key model.embed_tokens.weight \
        --ttt-length 7 \
        2>&1 | tee $log_path

    echo "Training completed successfully. Output saved to $output_dir"
}

main "$@"