"""Autograd-aware layout transitions for the extended TP track.

These four operations are the small per-rank core hidden underneath systems
such as PyTorch DTensor tensor parallelism and manual JAX ``shard_map`` code.
The exercise brief gives the forward/backward dual for every operation.
"""

import torch
import torch.distributed as dist


def _group_size(group) -> int:
    return dist.get_world_size(group)


def _group_rank(group) -> int:
    return dist.get_rank(group)


def _all_gather_last_dim(tensor: torch.Tensor, group) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(_group_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=-1)


class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.contiguous().clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group):
        ctx.group = group
        output = tensor.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group):
        if tensor.ndim == 0:
            raise ValueError("cannot scatter a scalar across tensor-parallel ranks")
        world_size = _group_size(group)
        if tensor.shape[-1] % world_size != 0:
            raise ValueError(
                f"last dimension {tensor.shape[-1]} must be divisible by "
                f"tensor-parallel size {world_size}"
            )
        ctx.group = group
        return tensor.chunk(world_size, dim=-1)[_group_rank(group)].contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _all_gather_last_dim(grad_output, ctx.group), None


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group):
        if tensor.ndim == 0:
            raise ValueError("cannot gather scalar tensor-parallel shards")
        ctx.group = group
        ctx.rank = _group_rank(group)
        ctx.world_size = _group_size(group)
        return _all_gather_last_dim(tensor, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if grad_output.shape[-1] % ctx.world_size != 0:
            raise RuntimeError("gather output gradient cannot be evenly split")
        grad_input = grad_output.chunk(ctx.world_size, dim=-1)[ctx.rank]
        return grad_input.contiguous(), None


def copy_to_tensor_parallel_region(
    tensor: torch.Tensor, *, group=None
) -> torch.Tensor:
    """Forward identity; backward SUM across the tensor-parallel group."""
    return _CopyToTensorParallelRegion.apply(tensor, group)


def reduce_from_tensor_parallel_region(
    tensor: torch.Tensor, *, group=None
) -> torch.Tensor:
    """Forward SUM across the tensor-parallel group; backward identity."""
    return _ReduceFromTensorParallelRegion.apply(tensor, group)


def scatter_to_tensor_parallel_region(
    tensor: torch.Tensor, *, group=None
) -> torch.Tensor:
    """Forward split on the last dimension; backward all-gather."""
    return _ScatterToTensorParallelRegion.apply(tensor, group)


def gather_from_tensor_parallel_region(
    tensor: torch.Tensor, *, group=None
) -> torch.Tensor:
    """Forward all-gather on the last dimension; backward split."""
    return _GatherFromTensorParallelRegion.apply(tensor, group)
