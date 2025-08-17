#!/bin/bash
set -ex

# 获取路径
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")/../

pushd ${ROOT_DIR}
pip install -e .
popd

# 设置环境变量
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions
export MAX_JOBS=$(nproc)
export DS_BUILD_FUSED_ADAM=1  #只编译FUSED_ADAM 优化器

# 获取训练配置
model_idx=${1}
nnodes=${2}

if [ $nnodes -eq 1 ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=$(( ( RANDOM % 10000 )  + 20000 ))
    RANK=0
else 
    export OMP_NUM_THREADS=8
    export NCCL_NET=IB
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=^docker0,lo,eth0
    export NCCL_IB_GID_INDEX=3
    export NCCL_DEBUG=0
fi

config_file=${SCRIPT_DIR}/train_configs.json
config=$(jq -r ".\"$model_idx\"" "$config_file")

# 读取参数
target_model_path=$(jq -r '.target_model_path' <<< "$config")
draft_model_config=$(jq -r '.draft_model_config' <<< "$config")
batch_size=$(jq -r '.batch_size' <<< "$config")
zero_stage=$(jq -r '.zero_stage' <<< "$config")

# 设置相关路径参数
train_name=$(basename "$draft_model_config" .json)
save_path=/mnt/modelops/train/eagle3/
log_path=/mnt/modelops/train/eagle3/logs
output_path=/mnt/modelops/train/eagle3/output
mkdir -p ${log_path}
mkdir -p ${output_path}

output_dir=${output_path}/${train_name}_nnodes_${nnodes}
train_log_path=${log_path}/${train_name}_nnodes_${nnodes}_rank${RANK}_train.log
nvidia_smi_log_path=${log_path}/${train_name}_nnodes_${nnodes}_rank${RANK}_nvidia_smi.log

# 添加GPU 监控
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv -l 1 > ${nvidia_smi_log_path} &

# 正式开启训练
torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=8 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $ROOT_DIR/scripts/train_eagle3_online_deepspeed.py \
    --target-model-path ${target_model_path} \
    --draft-model-config ${ROOT_DIR}/configs/${draft_model_config} \
    --train-data-path /mnt/modelops/datasets/specforge_postprocess_ultrachat/ultrachat.jsonl \
    --output-dir $output_dir \
    --num-epochs 10 \
    --batch-size ${batch_size} \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir ${save_path}/cache/ \
    --embedding-key model.embed_tokens.weight \
    --ttt-length 7 \
    --zero-stage ${zero_stage} \
    2>&1 | tee $train_log_path

echo "Training completed successfully. Output saved to $output_dir"