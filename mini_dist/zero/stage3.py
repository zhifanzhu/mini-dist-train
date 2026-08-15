from dataclasses import dataclass
from collections.abc import Iterable
import torch
import torch.nn as nn
from mini_dist.zero.common import Partition
from mini_dist._todo import todo


@dataclass
class ShardedTensor1D:
    local_shard: torch.Tensor
    partition: Partition
    original_shape: torch.Size

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, *, group=None) -> "ShardedTensor1D":
        todo("ShardedTensor1D.from_tensor")

    def all_gather(self, *, group=None) -> torch.Tensor:
        """Return exact unpadded full tensor in original shape."""
        todo("ShardedTensor1D.all_gather")


class MiniZeRO3(nn.Module):
    """Algorithmic ZeRO-3 core used before FSDP framework realizations."""

    def __init__(self, module: nn.Module, *, group=None):
        super().__init__()
        todo("MiniZeRO3.__init__")

    def forward(self, *args, **kwargs):
        todo("MiniZeRO3.forward")
