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
    todo("partition_1d")


def pad_flat(flat: torch.Tensor, padded_numel: int) -> torch.Tensor:
    todo("pad_flat")
