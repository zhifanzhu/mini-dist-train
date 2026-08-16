import torch
import torch.distributed as dist
from .distributed import require_initialized
from ._todo import todo


def all_reduce_mean(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """In-place mean all-reduce; return `tensor` for convenience."""
    require_initialized()
    tensor.div_(dist.get_world_size(group))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def all_gather_fixed(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Gather equal-sized tensors and concatenate along dim 0."""
    require_initialized()
    world_size = dist.get_world_size(group)
    output = torch.empty(
        (tensor.shape[0] * world_size, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(output, tensor.contiguous(), group=group)
    return output


def reduce_scatter_sum(flat_tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Split a 1-D tensor evenly and reduce-scatter SUM into one shard per rank."""
    require_initialized()
    if flat_tensor.ndim != 1:
        raise ValueError(f"expected a 1-D tensor, got shape {tuple(flat_tensor.shape)}")
    world_size = dist.get_world_size(group)
    if flat_tensor.numel() % world_size:
        raise ValueError(
            f"tensor length {flat_tensor.numel()} must be divisible by process-group size {world_size}"
        )
    chunks = list(flat_tensor.chunk(world_size))
    output = torch.empty_like(chunks[0])
    dist.reduce_scatter(output, chunks, op=dist.ReduceOp.SUM, group=group)
    return output


def _global_peer(group, group_rank: int) -> int:
    return group_rank if group is None else dist.get_global_rank(group, group_rank)


def tree_all_reduce_sum(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Optional exercise: all-reduce SUM using a balanced P2P tree.

    Implement a tree-shaped reduction followed by tree-shaped dissemination.
    Do not call dist.all_reduce/reduce_scatter/all_gather/broadcast inside this
    function. Blocking, nonblocking, and batched P2P APIs are all valid.

    Tests intentionally check behavior rather than intercepting a particular
    P2P API. After they pass, ask a code-review agent to verify that both phases
    use a balanced tree rather than a linear chain or a hidden collective.
    """
    require_initialized()
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    buffer = torch.empty_like(tensor)

    stride = 1
    while stride < world_size:
        period = 2 * stride
        position = rank % period
        if position == 0:
            child = rank + stride
            if child < world_size:
                dist.recv(buffer, src=_global_peer(group, child), group=group)
                tensor.add_(buffer)
        elif position == stride:
            parent = rank - stride
            dist.send(tensor, dst=_global_peer(group, parent), group=group)
        stride = period

    stride //= 2
    while stride:
        period = 2 * stride
        position = rank % period
        if position == 0:
            child = rank + stride
            if child < world_size:
                dist.send(tensor, dst=_global_peer(group, child), group=group)
        elif position == stride:
            parent = rank - stride
            dist.recv(tensor, src=_global_peer(group, parent), group=group)
        stride //= 2
    return tensor


def ring_all_reduce_sum(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Optional exercise: all-reduce SUM using point-to-point communication only.

    You may assume a 1-D tensor whose length is divisible by world_size.
    Do not call dist.all_reduce/reduce_scatter/all_gather inside this function.

    Blocking, nonblocking, and batched P2P APIs are all valid. Tests do not
    intercept a particular P2P API; after they pass, ask a code-review agent to
    verify the chunked reduce-scatter and all-gather structure.
    """
    require_initialized()
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if tensor.ndim != 1:
        raise ValueError(f"expected a 1-D tensor, got shape {tuple(tensor.shape)}")
    if tensor.numel() % world_size:
        raise ValueError(
            f"tensor length {tensor.numel()} must be divisible by process-group size {world_size}"
        )
    if world_size == 1:
        return tensor

    chunks = list(tensor.chunk(world_size))
    buffer = torch.empty_like(chunks[0])
    next_rank = _global_peer(group, (rank + 1) % world_size)
    previous_rank = _global_peer(group, (rank - 1) % world_size)

    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        reduce_index = (rank - step - 2) % world_size
        send_work = dist.isend(chunks[send_index], dst=next_rank, group=group)
        recv_work = dist.irecv(buffer, src=previous_rank, group=group)
        recv_work.wait()
        send_work.wait()
        chunks[reduce_index].add_(buffer)

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        receive_index = (rank - step - 1) % world_size
        send_work = dist.isend(chunks[send_index], dst=next_rank, group=group)
        recv_work = dist.irecv(chunks[receive_index], src=previous_rank, group=group)
        recv_work.wait()
        send_work.wait()
    return tensor
