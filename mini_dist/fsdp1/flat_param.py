from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
from mini_dist.zero.common import Partition, pad_flat, partition_1d
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
        self.module = module
        self.group = group
        infos = []
        tensors = []
        offset = 0
        for name, parameter in module.named_parameters():
            infos.append(ParamInfo(name, parameter.shape, parameter.numel(), offset))
            tensors.append(parameter.detach().reshape(-1))
            offset += parameter.numel()
        self._infos = tuple(infos)
        full = torch.cat(tensors) if tensors else torch.empty(0)
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        self._partition = partition_1d(full.numel(), rank, world_size)
        padded = pad_flat(full, self._partition.padded_numel)
        self._local_shard = padded[self._partition.start : self._partition.end].clone()

    @property
    def infos(self) -> tuple[ParamInfo, ...]:
        return self._infos

    @property
    def partition(self) -> Partition:
        return self._partition

    @property
    def local_shard(self) -> torch.Tensor:
        return self._local_shard

    def unshard(self) -> torch.Tensor:
        """All-gather local shards, remove padding, return full flat tensor."""
        gathered = torch.empty(
            self.partition.padded_numel,
            dtype=self.local_shard.dtype,
            device=self.local_shard.device,
        )
        dist.all_gather_into_tensor(
            gathered,
            self.local_shard.detach().clone().contiguous(),
            group=self.group,
        )
        return gathered[: self.partition.original_numel]

    def views(self, full_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return original-shaped views backed by `full_flat`."""
        if full_flat.numel() != self.partition.original_numel:
            raise ValueError("full_flat has the wrong number of elements")
        return {
            info.name: full_flat.narrow(0, info.offset, info.numel).view(info.shape)
            for info in self.infos
        }
