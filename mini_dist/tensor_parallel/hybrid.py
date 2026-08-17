"""Two-dimensional tensor-parallel plus FSDP composition."""

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mini_dist.fsdp2 import fully_shard
from .mlp import TensorParallelMLP


class ParallelMesh2D:
    """A row-major ``[data_parallel, tensor_parallel]`` process mesh.

    All ranks must construct the same mesh.  ``tp_group`` is the current row;
    ``dp_group`` is the current column.  This mirrors named ``("dp", "tp")``
    mesh dimensions used by production PyTorch and JAX systems.
    """

    def __init__(self, *, dp_size: int, tp_size: int):
        if not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before the mesh")
        if dp_size <= 0 or tp_size <= 0:
            raise ValueError("dp_size and tp_size must be positive")
        world_size = dist.get_world_size()
        if world_size != dp_size * tp_size:
            raise ValueError(
                f"world_size={world_size} must equal "
                f"dp_size * tp_size={dp_size * tp_size}"
            )

        rank = dist.get_rank()
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.world_size = world_size
        self.global_rank = rank
        self.dp_rank = rank // tp_size
        self.tp_rank = rank % tp_size
        self.tp_group = None
        self.dp_group = None

        # Process-group construction is collective with respect to the default
        # world. Every rank creates every group in this identical order.
        for dp_index in range(dp_size):
            ranks = [dp_index * tp_size + i for i in range(tp_size)]
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                self.tp_group = group
        for tp_index in range(tp_size):
            ranks = [i * tp_size + tp_index for i in range(dp_size)]
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                self.dp_group = group

        if self.tp_group is None or self.dp_group is None:
            raise RuntimeError("rank was not assigned to both mesh dimensions")

    @property
    def coordinate(self) -> tuple[int, int]:
        return self.dp_rank, self.tp_rank


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
        self.mesh = mesh
        self.tp_mlp = TensorParallelMLP(
            up_proj,
            down_proj,
            group=mesh.tp_group,
            activation=activation,
        )
        fully_shard(self.tp_mlp, group=mesh.dp_group)

    @property
    def fsdp_group(self):
        return self.tp_mlp._mini_fsdp_group

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.tp_mlp(tensor)
