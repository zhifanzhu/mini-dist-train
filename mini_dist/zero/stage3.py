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

        world_size = dist.get_world_size(group=group)

        # Need to avoid closure variable catching
        def make_hook(param, shard_param):
            def hook(p):
                if p.grad is None:
                    return
                """ Grad is full gradient for param """
                grad_flat = p.grad.flatten()
                grad_flat = pad_flat(
                    grad_flat, padded_numel=shard_param.partition.padded_numel)
                with torch.no_grad():
                    local_shard_grad = torch.zeros_like(shard_param.local_shard)
                    grad_list = list(torch.chunk(grad_flat, world_size))
                    dist.reduce_scatter(
                        local_shard_grad,
                        grad_list,
                        op=dist.ReduceOp.SUM,
                        group=self.group)
                    local_shard_grad.div_(world_size)
                param.data = shard_param.local_shard
                param.grad = local_shard_grad

            return hook

        for name, param in module.named_parameters():
            shard_param = ShardedTensor1D.from_tensor(
                param, group=group)
            param.data = shard_param.local_shard
            # param.register_hook(make_hook(param, shard_param))
            param.register_post_accumulate_grad_hook(
                make_hook(param, shard_param))
            self.shard_params.append( shard_param )

        self.group = group

    def forward(self, *args, **kwargs):

        with torch.no_grad():
            for sp, p in zip(self.shard_params, self.params):
                tensor = sp.all_gather(group=self.group)
                p.data = tensor

        out = self.module(*args, **kwargs)
        return out