from dataclasses import dataclass
import torch
import torch.nn as nn
from mini_dist.zero.common import Partition
from mini_dist._todo import todo


@dataclass(frozen=True)
class ParamInfo:
    name: str
    shape: torch.Size
    numel: int
    offset: int


class FlatParameterHandle:
    """FSDP1-style permanent flat-parameter representation."""

    def __init__(self, module: nn.Module, *, group=None):
        todo("FlatParameterHandle.__init__")

    @property
    def infos(self) -> tuple[ParamInfo, ...]:
        todo("FlatParameterHandle.infos")

    @property
    def partition(self) -> Partition:
        todo("FlatParameterHandle.partition")

    @property
    def local_shard(self) -> torch.Tensor:
        todo("FlatParameterHandle.local_shard")

    def unshard(self) -> torch.Tensor:
        """All-gather local shards, remove padding, return full flat tensor."""
        todo("FlatParameterHandle.unshard")

    def views(self, full_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return original-shaped views backed by `full_flat`."""
        todo("FlatParameterHandle.views")
