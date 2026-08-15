import torch
import torch.distributed as dist
from .distributed import require_initialized
from ._todo import todo


def all_reduce_mean(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """In-place mean all-reduce; return `tensor` for convenience."""
    ws = dist.get_world_size()
    tensor = tensor / ws
    dist.all_reduce(tensor, dist.ReduceOp.SUM, group=group)
    return tensor


def all_gather_fixed(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Gather equal-sized tensors and concatenate along dim 0."""
    ws = dist.get_world_size()
    dd = {'device': tensor.device, 'dtype': tensor.dtype}
    shape = tensor.shape
    # rank = dist.get_rank()
    # print(rank, shape, tensor)
    tensor_out = torch.zeros([shape[0] * ws, *shape[1:]], **dd)
    dist.all_gather_single(tensor_out, tensor, group=group)
    # print(tensor_out.shape, tensor_out)
    return tensor_out


def reduce_scatter_sum(flat_tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Split a 1-D tensor evenly and reduce-scatter SUM into one shard per rank.

    
    rank 0:     [ a0 | a1 | ... | a_k-1 ]
    rank 1:     [ b0 | b1 | ... | b_k-1 ]
    ...
    rank k-1:   [ t0 | t1 | ... | t_k-1 ]
                  ^                ^
                r0 reduce            r_k-1 reduce

    output:
        rank 0 = [ a0 + b0 + ... + t0]
        rank 1 = [ a1 + b1 + ... + b_{k-1}]
        ...
        rank k-1 = [ a_k-1 + b_k-1 + ... + t_k-1 ]
    """
    ws = dist.get_world_size()
    splits = list(flat_tensor.chunk(ws))
    tensor_out = torch.zeros_like(splits[0])
    dist.reduce_scatter(tensor_out, splits, op=dist.ReduceOp.SUM, group=group)
    # rank = dist.get_rank()
    # print(rank, flat_tensor, splits)
    # print(rank, tensor_out)
    return tensor_out


def ring_all_reduce_sum(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Optional exercise: all-reduce SUM using point-to-point communication only.

    You may assume a 1-D tensor whose length is divisible by world_size.
    Do not call dist.all_reduce/reduce_scatter/all_gather inside this function.
    """
    todo("ring_all_reduce_sum")
