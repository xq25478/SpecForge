from datetime import timedelta
import os

import torch
import torch.distributed as dist

from specforge.utils import print_with_rank

_TP_GROUP = None
_DP_GROUP = None
_SHARD_GROUP = None
_REPLICATE_GROUP = None


def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


def get_shard_group():
    global _SHARD_GROUP
    return _SHARD_GROUP


def get_replicate_group():
    global _REPLICATE_GROUP
    return _REPLICATE_GROUP


def get_dp_group():
    global _DP_GROUP
    return _DP_GROUP


def init_distributed(timeout: int = 10, tp_size: int = 1):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
        tp_size(int): The degree of tensor parallelism
    """
    
    local_rank = int(os.getenv("LOCAL_RANK",0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK']),
        timeout=timedelta(minutes=timeout))
    print_with_rank(f"bind to device {local_rank}")

    rank = dist.get_rank()
    node_rank = rank // 8
    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    assert world_size == tp_size * dp_size, "world size must be divisible by tp size"
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", 8))
    num_nodes = world_size // local_world_size

    global _TP_GROUP, _DP_GROUP, _SHARD_GROUP, _REPLICATE_GROUP

    # Create _SHARD_GROUP
    dp_size_within_node = local_world_size // tp_size
    for tp_idx in range(dp_size_within_node):
        start_rank = node_rank * local_world_size + tp_idx * tp_size
        tp_ranks = list(range(start_rank, start_rank + tp_size))
        tp_group = dist.new_group(tp_ranks)
        if rank in tp_ranks:
            _SHARD_GROUP = tp_group

    # Create _REPLICATE_GROUP
    is_multi_node = num_nodes > 1
    if is_multi_node:
        tp_idx_within_node = local_rank // tp_size
        replicate_ranks = [i * local_world_size + tp_idx_within_node * tp_size + j for i in range(num_nodes) for j in range(tp_size)]
    else:
        replicate_ranks = list(range(world_size))
    _REPLICATE_GROUP = dist.new_group(replicate_ranks)
    
    print(f"RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

    # Create _TP_GROUP _DP_GROUP
    if tp_size == 1:
        _TP_GROUP = dist.new_group([rank])  # dummy group
        _DP_GROUP = dist.new_group(list(range(world_size)))  # 全局 DP
    else:
        # create tp group
        tp_ranks = [list(range(i * tp_size, (i + 1) * tp_size)) for i in range(dp_size)]
        for ranks in tp_ranks:
            tp_group = dist.new_group(ranks=ranks)
            if rank in ranks:
                _TP_GROUP = tp_group

        # create dp group
        dp_ranks = [list(range(i, world_size, tp_size)) for i in range(tp_size)]
        for ranks in dp_ranks:
            dp_group = dist.new_group(ranks=ranks)
            if rank in ranks:
                _DP_GROUP = dp_group


def destroy_distributed():
    """简化销毁函数"""
    # 销毁您创建的 _TP_GROUP 和 _DP_GROUP
    global _TP_GROUP, _DP_GROUP
    if _TP_GROUP is not None:
        dist.destroy_process_group(_TP_GROUP)
        _TP_GROUP = None
    if _DP_GROUP is not None:
        dist.destroy_process_group(_DP_GROUP)
        _DP_GROUP = None
    
    # 最后销毁默认的全局 process group
    if dist.is_initialized():
        dist.destroy_process_group()

