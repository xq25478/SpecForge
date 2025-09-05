import argparse
import hashlib
import os

import torch
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge import (
    AutoDistributedTargetModel,
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
)
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import get_last_checkpoint, print_with_rank, rank_0_priority
# from specforge.utils import (
#     get_last_checkpoint,
#     print_with_rank,
#     rank_0_priority,
#     validate_wandb_args,
# )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # dist_timeout tp_size wandb draft_model_dir target_model_path draft_model_config embedding_key eval_data_path batch_size chat_template max_length ttt_length

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)
    parser.add_argument("--draft-model-path", type=str, required=True)
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )

    # add training-related arguments
    # parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    # parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    # parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    # parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )

    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)

    # other args
    # parser.add_argument("--cache-key", type=str, default=None)
    # parser.add_argument("--cache-dir", type=str, default="./cache")
    # parser.add_argument("--output-dir", type=str, required=True)
    # parser.add_argument("--eval-interval", type=int, default=1)
    # parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )

    # resume
    # parser.add_argument("--resume", action="store_true")

    # wandb wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)

    args = parser.parse_args()

    return parser, args


def init_wandb(args):
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project, name=args.wandb_name)


def wandb_log_if_initialized(log_dict):
    if dist.get_rank() == 0 and wandb.run is not None:
        wandb.log(log_dict)


def print_on_rank0(message):
    if dist.get_rank() == 0:
        print(message)


def main():
    # initialize
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank(f"Initialized distributed environment")

    # Validate wandb arguments
    # validate_wandb_args(parser, args)

    if args.wandb and dist.get_rank() == 0:
        init_wandb(args)

    # detecting last ckpt for draft model
    draft_model_last_checkpoint = None
    if os.path.isdir(args.draft_model_path):
        # print_on_rank0(args.draft_model_path)
        draft_model_last_checkpoint = args.draft_model_path#get_last_checkpoint(args.draft_model_path)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # build target and draft model
    if args.tp_size > 1:
        # to avoid CPU RAM OOM, we directly init the model on CUDA
        target_model = AutoDistributedTargetModel.from_pretrained(
            pretrained_model_name_or_path=args.target_model_path,
            torch_dtype=torch.bfloat16,
            device="cuda",
        ).eval()
    else:
        target_model = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .cuda()
        )
    print_with_rank(f"Initialized target model")
    # load model with resume
    print(args.draft_model_config)
    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    if draft_model_last_checkpoint:
        draft_model = (
            AutoEagle3DraftModel.from_pretrained(draft_model_last_checkpoint)
            .cuda()
            .to(torch.bfloat16)
        )
    else:
        draft_model = (
            AutoEagle3DraftModel.from_config(draft_model_config)
            .cuda()
            .to(torch.bfloat16)
        )
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    print_with_rank(f"Initialized draft model")

    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
    eval_eagle3_dataset = build_eagle3_dataset(
            eval_dataset,
            tokenizer,
            args.chat_template,
            args.max_length,
    )
    eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
    )
    print_with_rank(f"Initialized eval dataloader")

    # build Eagle3 model
    # broadcast draft model
    eagle3_model = OnlineEagle3Model(
        target_model=target_model,
        draft_model=draft_model,
        length=args.ttt_length,
    )
    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        ignored_modules=[target_model],
        process_group=get_dp_group(),
    )
    print_with_rank(f"Initialized Eagle3 FSDP model")

    draft_model.eval()
    eval_acces = [[] for _ in range(eagle3_model.length)]
    eval_plosses = [[] for _ in range(eagle3_model.length)]

    for data in tqdm(eval_dataloader, desc=f"Evaluating"):
        plosses, _, acces = eagle3_model(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
        )
        eval_acces = [eval_acces[i] + [acces[i]] for i in range(len(acces))]
        eval_plosses = [
                eval_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
        ]

    for i in range(len(eval_acces)):
        acc_i = torch.tensor(eval_acces[i]).cuda().mean()
        dist.all_reduce(acc_i)
        acc_i = acc_i / dist.get_world_size()
        acc_i = acc_i.item()

        wandb_log_if_initialized({f"eval/epochacc_{i}": acc_i})
        print_on_rank0(
                f"Eval in {args.eval_data_path}, position {i},  Acc: {acc_i:.2f}"
        )

    for i in range(len(eval_plosses)):
        loss_i = torch.tensor(eval_plosses[i]).cuda().mean()
        dist.all_reduce(loss_i)
        loss_i = loss_i / dist.get_world_size()
        loss_i = loss_i.item()

        wandb_log_if_initialized({f"eval/epochploss_{i}": loss_i})
        print_on_rank0(
                f"Eval in {args.eval_data_path}, position {i}, pLoss: {loss_i:.2f}"
        )


if __name__ == "__main__":
    main()
