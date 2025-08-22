CUDA_VISIBLE_DEVICES=4,5

# qwen3-8b
torchrun \
    --standalone \
    --nproc_per_node 2 \
    ./scripts/test_eagle3_online.py \
    --target-model-path /mnt/modelops/models/Qwen3-8B \
    --draft-model-config ./configs/qwen3-8b-eagle3.json \
    --eval-data-path /mnt/modelops/487922/online_data/alpaca/online_alpaca.json \
    --draft-model-path /mnt/modelops/train/eagle/weight_update/Eagle3_Qwen3_8B_LLM_ultrachat_1 \
    --batch-size 2 \
    --max-length 2048 \
    --chat-template qwen \
/mnt/modelops/train/eagle/data/online_data/alpaca/online_alpaca.json

# qwen3-14b
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node 4 \
    ./scripts/test_eagle3_online.py \
    --target-model-path /mnt/modelops/models/Qwen3-14B \
    --draft-model-config ./configs/qwen3-14b-eagle3.json \
    --eval-data-path /mnt/modelops/train/eagle/data/online_data/alpaca/online_alpaca.json \
    --draft-model-path /mnt/modelops/models/AngelSlim/Qwen3-14B_eagle3 \
    --batch-size 2 \
    --max-length 2048 \
    --chat-template qwen \

# qwen3-30b-a3b
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node 4 \
    ./scripts/test_eagle3_online.py \
    --target-model-path /mnt/modelops/models/Qwen3-30B-A3B \
    --draft-model-config /mnt/modelops/487922/ant_dev/SpecForge/configs/qwen3-30B-A3b-eagle3.json \
    --eval-data-path /mnt/modelops/train/eagle/data/online_data/alpaca/online_alpaca.json \
    --draft-model-path /mnt/modelops/train/eagle3/baseline_ultrachat_sft_train_only/weight/Qwen3_30B_A3B_eagle3_16gpus_Epoch10 \
    --batch-size 1 \
    --max-length 2048 \
    --chat-template qwen

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node 4 \
    ./scripts/test_eagle3_online.py \
    --target-model-path /mnt/modelops/models/Qwen3-30B-A3B \
    --draft-model-config /mnt/modelops/487922/ant_dev/SpecForge/configs/qwen3-30B-A3b-eagle3.json \
    --eval-data-path /mnt/modelops/train/eagle/data/online_data/alpaca/online_alpaca.json \
    --draft-model-path /mnt/modelops/models/AngelSlim/Qwen3-a3B_eagle3 \
    --batch-size 1 \
    --max-length 2048 \
    --chat-template qwen

# deepseek-v2-lite-chat
torchrun \
    --standalone \
    --nproc_per_node 8 \
    ./scripts/test_eagle3_online.py \
    --target-model-path /mnt/modelops/487922/deepseek-ai__DeepSeek-V2-Lite-Chat \
    --draft-model-config /mnt/modelops/487922/ant_dev/SpecForge/configs/deepseek-v2-lite-eagle3.json \
    --eval-data-path /mnt/modelops/train/eagle/data/online_data/alpaca/online_alpaca.json \
    --draft-model-path /mnt/modelops/train/eagle3/baseline_ultrachat_sft_train_only/weight/Deepseek_V2_Lite_Chat_eagle3_16gpus_Epoch10 \
    --batch-size 1 \
    --max-length 2048 \
    --chat-template deepseek