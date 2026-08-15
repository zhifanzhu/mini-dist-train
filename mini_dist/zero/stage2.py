from collections.abc import Iterable
import torch
from mini_dist.zero.common import Partition
from mini_dist._todo import todo


class Zero2Optimizer:
    """Educational ZeRO-2: sharded optimizer state + sharded reduced grads.

    Keep the ZeRO-1 flattened partition ownership, but do not retain a full
    reduced gradient. Reduce-scatter the packed gradient and retain only the
    local mean-gradient shard used by the local optimizer partition.
    """

    partition: Partition
    local_master_param: torch.nn.Parameter
    local_grad_shard: torch.Tensor | None
    optimizer: torch.optim.Optimizer

    def __init__(self, params: Iterable[torch.nn.Parameter], optimizer_cls, *, group=None, **optim_kwargs):
        todo("Zero2Optimizer.__init__")

    def reduce_scatter_gradients(self):
        todo("Zero2Optimizer.reduce_scatter_gradients")

    def step(self):
        todo("Zero2Optimizer.step")

    def zero_grad(self, set_to_none: bool = True):
        todo("Zero2Optimizer.zero_grad")
