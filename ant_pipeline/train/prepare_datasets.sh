#!/bin/bash
set -ex

script_dir=$(cd "$(dirname "$0")" && pwd)

# 数据集列表
datasets=(
    "ultrachat:/mnt/modelops/datasets/HuggingFaceH4/ultrachat_200k/"
)

# 循环处理每个数据集
for entry in "${datasets[@]}"; do
    dataset_name="${entry%%:*}"   # 冒号前的部分
    data_path="${entry#*:}"       # 冒号后的部分

    output_path="/mnt/modelops/datasets/specforge_postprocess_${dataset_name}"

    rm -rf "${output_path}"
    mkdir -p "${output_path}"

    python3 "${script_dir}/../../scripts/prepare_data.py" \
        --dataset "${dataset_name}" \
        --output_path "${output_path}" \
        --data-path "${data_path}"
done