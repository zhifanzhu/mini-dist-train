"""A Megatron/PyTorch-style tensor-parallel MLP."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_dist._todo import todo


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
        todo("TensorParallelMLP.__init__")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        todo("TensorParallelMLP.forward")
