"""Column-wise and row-wise tensor-parallel linear layers."""

import torch
import torch.nn as nn

from mini_dist._todo import todo


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
        todo("ColumnParallelLinear.__init__")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        todo("ColumnParallelLinear.forward")


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
        todo("RowParallelLinear.__init__")

    def forward(self, local_tensor: torch.Tensor) -> torch.Tensor:
        todo("RowParallelLinear.forward")
