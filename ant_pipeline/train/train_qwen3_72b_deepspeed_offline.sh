torchrun --nproc_per_node=8 \
scripts/prepare_hidden_states.py \
--model-path /mnt/modelops/models/Qwen2.5-72B \
--enable-aux-hidden-states \
--data-path /mnt/modelops/train/eagle3/baseline_ultrachat_sft_train_only/data/ultrachat_sft_train.jsonl \
--cache-dir /mnt/modelops/train/eagle3/offline_data/ultrachat_qwen3_72b_cache \
--output-path /mnt/modelops/train/eagle3/offline_data/ultrachat_qwen3_72b/ \
--chat-template qwen \
--max-length 2048 \
--tp-size 8 \
--batch-size 4 \
--mem-frac=0.8 \
--num-samples 207865 \


torchrun --nproc_per_node=4 \
scripts/prepare_hidden_states.py \
--model-path /mnt/modelops/models/Qwen3-8B \
--enable-aux-hidden-states \
--data-path /mnt/modelops/train/eagle3/baseline_ultrachat_sft_train_only/data/ultrachat_sft_train.jsonl \
--cache-dir /mnt/modelops/train/eagle3/offline_data/ultrachat_qwen3_8b_cache \
--output-path /mnt/modelops/train/eagle3/offline_data/ultrachat_qwen3_8b/ \
--chat-template qwen \
--max-length 2048 \
--tp-size 4 \
--batch-size 1 \
--mem-frac=0.8 \
--num-samples 1000