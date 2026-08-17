"""Autograd-aware layout transitions for the extended TP track.

These four operations are the small per-rank core hidden underneath systems
such as PyTorch DTensor tensor parallelism and manual JAX ``shard_map`` code.
The exercise brief gives the forward/backward dual for every operation.
"""

import torch

from mini_dist._todo import todo


def copy_to_tensor_parallel_region(
    tensor: torch.Tensor, *, group=None
) -> torch.Tensor:
    """Forward identity; backward SUM across the tensor-parallel group."""
    todo("copy_to_tensor_parallel_region")


def reduce_from_tensor_parallel_region(
    tensor: torch.Tensor, *, group=None
) -> torch.Tensor:
    """Forward SUM across the tensor-parallel group; backward identity."""
    todo("reduce_from_tensor_parallel_region")


def scatter_to_tensor_parallel_region(
    tensor: torch.Tensor, *, group=None
) -> torch.Tensor:
    """Forward split on the last dimension; backward all-gather."""
    todo("scatter_to_tensor_parallel_region")


def gather_from_tensor_parallel_region(
    tensor: torch.Tensor, *, group=None
) -> torch.Tensor:
    """Forward all-gather on the last dimension; backward split."""
    todo("gather_from_tensor_parallel_region")
