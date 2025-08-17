set -ex

# pip install sglang[all]
target_model_path=/mnt/modelops/models/Qwen3-30B-A3B/
draft_model_path=/mnt/modelops/train/eagle3/output/qwen3-30B-A3b-eagle3_nnodes_8/epoch_9

python3 -m sglang.launch_server \
    --model $target_model_path  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path $draft_model_path \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 32 \
    --tp 2 \
    --context-length 8192 \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 9122 \
    --dtype bfloat16