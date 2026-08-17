"""Extended tensor-parallel exercises.

The implementation is intentionally built from process groups and explicit
autograd mappings.  This exposes the layout transitions that production
DTensor/shard-map systems normally track for the user.
"""

from .collectives import (
    copy_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    scatter_to_tensor_parallel_region,
)
from .hybrid import HybridTensorFSDPMLP, ParallelMesh2D
from .layers import ColumnParallelLinear, RowParallelLinear
from .mlp import TensorParallelMLP

__all__ = [
    "ColumnParallelLinear",
    "HybridTensorFSDPMLP",
    "ParallelMesh2D",
    "RowParallelLinear",
    "TensorParallelMLP",
    "copy_to_tensor_parallel_region",
    "gather_from_tensor_parallel_region",
    "reduce_from_tensor_parallel_region",
    "scatter_to_tensor_parallel_region",
]
