import torch
import torch.distributed as dist
from .distributed import require_initialized
from ._todo import todo


def all_reduce_mean(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """In-place mean all-reduce; return `tensor` for convenience."""
    todo("all_reduce_mean")


def all_gather_fixed(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Gather equal-sized tensors and concatenate along dim 0."""
    todo("all_gather_fixed")


def reduce_scatter_sum(flat_tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Split a 1-D tensor evenly and reduce-scatter SUM into one shard per rank."""
    todo("reduce_scatter_sum")


def ring_all_reduce_sum(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Optional exercise: all-reduce SUM using point-to-point communication only.

    You may assume a 1-D tensor whose length is divisible by world_size.
    Do not call dist.all_reduce/reduce_scatter/all_gather inside this function.
    """
    todo("ring_all_reduce_sum")
