"""Column-wise and row-wise tensor-parallel linear layers."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .collectives import (
    copy_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
)


class ColumnParallelLinear(nn.Module):
    """Transform an ``nn.Linear`` by sharding its output features.

    PyTorch stores a linear weight as ``[out_features, in_features]``.  Each
    TP rank owns a contiguous shard of dimension 0 and the matching bias
    shard.  Input is replicated.  Output is sharded unless ``gather_output``
    is true.

    Do not retain the source ``nn.Linear``: the resulting module must own only
    its local parameters.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        group=None,
        gather_output: bool = False,
    ):
        super().__init__()
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        if linear.out_features % world_size != 0:
            raise ValueError(
                f"out_features={linear.out_features} must be divisible by "
                f"tensor-parallel size {world_size}"
            )

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.local_out_features = linear.out_features // world_size
        self.group = group
        self.tp_rank = rank
        self.tp_world_size = world_size
        self.gather_output = gather_output

        local_weight = linear.weight.detach().chunk(world_size, dim=0)[rank]
        self.weight = nn.Parameter(
            local_weight.contiguous().clone(),
            requires_grad=linear.weight.requires_grad,
        )
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            local_bias = linear.bias.detach().chunk(world_size, dim=0)[rank]
            self.bias = nn.Parameter(
                local_bias.contiguous().clone(),
                requires_grad=linear.bias.requires_grad,
            )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dimension {self.in_features}, "
                f"got {tensor.shape[-1]}"
            )
        replicated_input = copy_to_tensor_parallel_region(
            tensor, group=self.group
        )
        local_output = F.linear(replicated_input, self.weight, self.bias)
        if self.gather_output:
            return gather_from_tensor_parallel_region(
                local_output, group=self.group
            )
        return local_output


class RowParallelLinear(nn.Module):
    """Transform an ``nn.Linear`` by sharding its input features.

    Each TP rank owns a contiguous shard of weight dimension 1 and consumes
    the matching last-dimension activation shard.  Local matrix products are
    partial values and must be SUM-reduced.  The replicated bias is added only
    after that reduction.

    Do not retain the source ``nn.Linear``: the resulting module must own only
    its local parameters.
    """

    def __init__(self, linear: nn.Linear, *, group=None):
        super().__init__()
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        if linear.in_features % world_size != 0:
            raise ValueError(
                f"in_features={linear.in_features} must be divisible by "
                f"tensor-parallel size {world_size}"
            )

        self.in_features = linear.in_features
        self.local_in_features = linear.in_features // world_size
        self.out_features = linear.out_features
        self.group = group
        self.tp_rank = rank
        self.tp_world_size = world_size

        local_weight = linear.weight.detach().chunk(world_size, dim=1)[rank]
        self.weight = nn.Parameter(
            local_weight.contiguous().clone(),
            requires_grad=linear.weight.requires_grad,
        )
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                linear.bias.detach().clone(),
                requires_grad=linear.bias.requires_grad,
            )

    def forward(self, local_tensor: torch.Tensor) -> torch.Tensor:
        if local_tensor.shape[-1] != self.local_in_features:
            raise ValueError(
                f"expected local input last dimension {self.local_in_features}, "
                f"got {local_tensor.shape[-1]}"
            )
        partial_output = F.linear(local_tensor, self.weight, bias=None)
        output = reduce_from_tensor_parallel_region(
            partial_output, group=self.group
        )
        if self.bias is not None:
            output = output + self.bias
        return output
