from dataclasses import dataclass
import torch
import torch.nn as nn
from mini_dist.zero.common import Partition
from mini_dist.zero.stage3 import ShardedTensor1D
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
        self.named_param_infos = dict()
        tensors = []
        offset = 0
        for name, param in module.named_parameters():
            info = ParamInfo(
                name=name,
                shape=param.shape,
                numel=param.numel(),
                offset=offset
            )
            offset += param.numel()
            self.named_param_infos[name] = info
            tensors.append(param.flatten())
        tensor = torch.cat(tensors)
        self.shard = ShardedTensor1D.from_tensor(tensor, group=group)
        self.group = group

    @property
    def infos(self) -> tuple[ParamInfo, ...]:
        return tuple(self.named_param_infos.values())

    @property
    def partition(self) -> Partition:
        return self.shard.partition

    @property
    def local_shard(self) -> torch.Tensor:
        return self.shard.local_shard

    def unshard(self) -> torch.Tensor:
        """All-gather local shards, remove padding, return full flat tensor."""
        original_numel = self.partition.original_numel
        tensor = self.shard.all_gather(group=self.group)
        return tensor[:original_numel]

    def views(self, full_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return original-shaped views backed by `full_flat`."""
        named_views = dict() 
        for name, info in self.named_param_infos.items():
            s = info.offset
            e = s + info.numel
            shape = info.shape
            named_views[name] = full_flat[s:e].view(shape)
        return named_views