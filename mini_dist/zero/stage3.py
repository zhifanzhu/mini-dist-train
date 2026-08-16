from dataclasses import dataclass
from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.distributed as dist
from mini_dist.zero.common import Partition, pad_flat, partition_1d
from mini_dist._todo import todo


@dataclass
class ShardedTensor1D:
    local_shard: torch.Tensor
    partition: Partition
    original_shape: torch.Size

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, *, group=None) -> "ShardedTensor1D":
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        flat = tensor.detach().reshape(-1)
        partition = partition_1d(flat.numel(), rank, world_size)
        padded = pad_flat(flat, partition.padded_numel)
        local = padded[partition.start : partition.end].clone()
        return cls(local, partition, tensor.shape)

    def all_gather(self, *, group=None) -> torch.Tensor:
        """Return exact unpadded full tensor in original shape."""
        gathered = torch.empty(
            self.partition.padded_numel,
            dtype=self.local_shard.dtype,
            device=self.local_shard.device,
        )
        dist.all_gather_into_tensor(gathered, self.local_shard.contiguous(), group=group)
        return gathered[: self.partition.original_numel].view(self.original_shape)


class MiniZeRO3(nn.Module):
    """Algorithmic ZeRO-3 core used before FSDP framework realizations."""

    def __init__(self, module: nn.Module, *, group=None):
        super().__init__()
        self.module = module
        self.group = group
        self.sharded_parameters = []
        for name, parameter in self.module.named_parameters():
            sharded = ShardedTensor1D.from_tensor(parameter, group=group)
            self.sharded_parameters.append((name, parameter, sharded))
            with torch.no_grad():
                parameter.data = sharded.local_shard
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(
                    self._make_post_accumulate_hook(parameter, sharded)
                )

    def _make_post_accumulate_hook(self, parameter, sharded):
        def hook(_parameter):
            grad = parameter.grad.reshape(-1)
            padded = pad_flat(grad, sharded.partition.padded_numel)
            chunks = list(padded.chunk(sharded.partition.world_size))
            local_grad = torch.empty_like(chunks[0])
            dist.reduce_scatter(local_grad, chunks, op=dist.ReduceOp.SUM, group=self.group)
            local_grad.div_(sharded.partition.world_size)
            with torch.no_grad():
                parameter.data = sharded.local_shard
            parameter.grad = local_grad.to(parameter.dtype)

        return hook

    def materialize(self) -> None:
        for _, parameter, sharded in self.sharded_parameters:
            with torch.no_grad():
                parameter.data = sharded.all_gather(group=self.group)

    def forward(self, *args, **kwargs):
        self.materialize()
        return self.module(*args, **kwargs)
