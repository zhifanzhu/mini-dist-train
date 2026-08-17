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
    shard_numel = (numel + world_size - 1) // world_size
    padded_numel = shard_numel * world_size
    return Partition(
        original_numel=numel,
        padded_numel=padded_numel,
        shard_numel=shard_numel,
        rank=rank,
        world_size=world_size,
        start=rank * shard_numel,
        # end=min(numel, (rank + 1) * shard_numel)
        end=(rank + 1) * shard_numel
    )


def pad_flat(flat: torch.Tensor, padded_numel: int) -> torch.Tensor:
    dd = {'dtype': flat.dtype, 'device': flat.device}
    pad = torch.zeros([padded_numel], **dd)
    pad[:flat.numel()] = flat
    return pad
