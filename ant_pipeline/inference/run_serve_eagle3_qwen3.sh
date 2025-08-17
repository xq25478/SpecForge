set -ex

USE_EAGLE3=true # false or true
ENGINE_TYPE=trt # trt or sglang

# for sglang
speculative_num_steps=5
speculative_eagle_topk=8
speculative_num_draft_tokens=4

# for trt
max_draft_len=4

# base_model=/mnt/modelops/models/Qwen3-32B
# eagle_model=/mnt/modelops/models/AngelSlim/Qwen3-32B_eagle3

# base_model=/mnt/modelops/models/Qwen3-14B
# eagle_model=/mnt/modelops/models/AngelSlim/Qwen3-14B_eagle3

base_model=/mnt/modelops/models/Qwen3-30B-A3B/
eagle_model=/mnt/modelops/train/eagle3/output/qwen3-30B-A3b-eagle3_nnodes_8/epoch_9


if [ ${ENGINE_TYPE} == "sglang" ];then
  command=(
    python3 -m sglang.launch_server \
        --model ${base_model}  \
        --host 127.0.0.1 \
        --port 9122 \
        --mem-fraction 0.85 \
        --cuda-graph-max-bs 64 \
        --max-running-requests 64 \
        --chunked-prefill-size 8192 \
        --tp-size 8
  )

  if $USE_EAGLE3; then
    command+=( 
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path ${eagle_model} \
        --speculative-num-steps ${speculative_num_steps} \
        --speculative-eagle-topk ${speculative_eagle_topk} \
        --speculative-num-draft-tokens ${speculative_num_draft_tokens} 
    )
  fi
else
  command=(
      python3 /mnt/modelops/460695/opensource/Ant-TensorRT-LLM/tensorrt_llm/commands/trtllm-serve.py \
      --model_path ${base_model} \
      --port 9122 \
      --host 127.0.0.1 \
      --backend pytorch \
      --max_batch_size 16 \
      --max_num_tokens 8192 \
  )

  if $USE_EAGLE3; then
    command+=(--spec_algo eagle3 --draft_model_path ${eagle_model} --max_draft_len ${max_draft_len})
  fi

fi

"${command[@]}"




