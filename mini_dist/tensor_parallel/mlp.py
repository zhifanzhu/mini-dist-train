"""A Megatron/PyTorch-style tensor-parallel MLP."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ColumnParallelLinear, RowParallelLinear


class TensorParallelMLP(nn.Module):
    """Column-parallel projection followed by row-parallel projection.

    The intermediate activation remains sharded on its last dimension.  Only
    the row-parallel partial result is reduced back to a replicated output.
    """

    def __init__(
        self,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        *,
        group=None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ):
        super().__init__()
        if up_proj.out_features != down_proj.in_features:
            raise ValueError(
                "up_proj.out_features must equal down_proj.in_features"
            )
        self.group = group
        self.activation = activation
        self.up_proj = ColumnParallelLinear(
            up_proj, group=group, gather_output=False
        )
        self.down_proj = RowParallelLinear(down_proj, group=group)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        local_hidden = self.activation(self.up_proj(tensor))
        return self.down_proj(local_hidden)
