"""Two-dimensional tensor-parallel plus FSDP composition."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_dist._todo import todo


class ParallelMesh2D:
    """A row-major ``[data_parallel, tensor_parallel]`` process mesh.

    All ranks must construct the same mesh.  ``tp_group`` is the current row;
    ``dp_group`` is the current column.  This mirrors named ``("dp", "tp")``
    mesh dimensions used by production PyTorch and JAX systems.
    """

    def __init__(self, *, dp_size: int, tp_size: int):
        todo("ParallelMesh2D.__init__")

    @property
    def coordinate(self) -> tuple[int, int]:
        todo("ParallelMesh2D.coordinate")


class HybridTensorFSDPMLP(nn.Module):
    """Tensor-parallel MLP whose TP-local parameters are FSDP-sharded.

    Construct tensor parallelism over ``mesh.tp_group`` first.  From the data
    parallel dimension's perspective, each TP-local parameter is then its
    full logical parameter, so apply the existing FSDP2-style ``fully_shard``
    over ``mesh.dp_group``.
    """

    def __init__(
        self,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        *,
        mesh: ParallelMesh2D,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ):
        super().__init__()
        todo("HybridTensorFSDPMLP.__init__")

    @property
    def fsdp_group(self):
        todo("HybridTensorFSDPMLP.fsdp_group")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        todo("HybridTensorFSDPMLP.forward")
