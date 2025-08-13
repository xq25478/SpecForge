#!/bin/bash

# H20 Qwen3-8B 训练脚本
# 单卡可以放下全部权重，因此tp-size=1

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)/../..

nproc_per_node=8 # 单节点GPU数 H20
nnodes=1 # 节点数量

torchrun \
    --nnodes ${nnodes} \
    --nproc_per_node $nproc_per_node \
    --master_addr=localhost \
    --master_port=29500 \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path /mnt/modelops/models/Qwen3-8B \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-data-path /mnt/modelops/datasets/specforge_postprocess_ultrachat/ultrachat.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen3-8B-eagle3 \
    --num-epochs 10 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 40960 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --ttt-length 7 \
    --tp-size 1 2>&1 | tee $ROOT_DIR/eagle3_qwen3_8b.ant_train_log
