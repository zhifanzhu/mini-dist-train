from dataclasses import dataclass
import torch
from mini_dist._todo import todo


@dataclass(frozen=True)
class Partition:
    original_numel: int
    padded_numel: int
    shard_numel: int
    rank: int
    world_size: int
    start: int
    end: int


def partition_1d(numel: int, rank: int, world_size: int) -> Partition:
    """Return equal-size padded shard metadata.

    `start:end` indexes the padded flat buffer.
    """
    if numel < 0:
        raise ValueError("numel must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    shard_numel = (numel + world_size - 1) // world_size
    padded_numel = shard_numel * world_size
    start = rank * shard_numel
    return Partition(numel, padded_numel, shard_numel, rank, world_size, start, start + shard_numel)


def pad_flat(flat: torch.Tensor, padded_numel: int) -> torch.Tensor:
    flat = flat.reshape(-1)
    if padded_numel < flat.numel():
        raise ValueError("padded_numel cannot be smaller than the input")
    output = torch.zeros(padded_numel, dtype=flat.dtype, device=flat.device)
    output[: flat.numel()].copy_(flat)
    return output
