from dataclasses import dataclass
from collections.abc import Iterable
import torch
from torch import distributed as dist
import torch.nn as nn
from mini_dist.zero.common import Partition, partition_1d, pad_flat
from mini_dist._todo import todo


@dataclass
class ShardedTensor1D:
    local_shard: torch.Tensor
    partition: Partition
    original_shape: torch.Size

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, *, group=None) -> "ShardedTensor1D":
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        pt = partition_1d(tensor.numel(), rank, world_size)
        flat = pad_flat(tensor.flatten(), pt.padded_numel)
        local_shard = flat[pt.start:pt.end]
        return cls(
            local_shard=local_shard,
            partition=pt,
            original_shape=tensor.shape
        )

    def all_gather(self, *, group=None) -> torch.Tensor:
        """Return exact unpadded full tensor in original shape."""
        tensor_list = [
            torch.zeros_like(self.local_shard)
            for _ in range(self.partition.world_size)
        ]
        dist.all_gather(
            tensor_list,
            self.local_shard,
            group=group
        )
        original_numel = self.partition.original_numel
        tensor = torch.cat(tensor_list)[:original_numel]
        return tensor.view(self.original_shape)


class MiniZeRO3(nn.Module):
    """Algorithmic ZeRO-3 core used before FSDP framework realizations."""

    def __init__(self, module: nn.Module, *, group=None):
        super().__init__()
        self.module = module
        self.params = list(module.parameters())
        self.shard_params = []
        for param in module.parameters():
            shard_param = ShardedTensor1D.from_tensor(
                param, group=group)
            self.shard_params.append( shard_param )
        self.group = group

    def forward(self, *args, **kwargs):

        with torch.no_grad():
            for sp, p in zip(self.shard_params, self.params):
                tensor = sp.all_gather(group=self.group)
                p.copy_(tensor)

        return self.module(*args, **kwargs)
