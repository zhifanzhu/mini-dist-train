from collections.abc import Iterable
import torch
from mini_dist.zero.common import Partition
from mini_dist._todo import todo


class Zero1Optimizer:
    """Educational ZeRO-1 optimizer wrapper.

    Expected realization: flatten model parameters/replicated gradients into a
    logical full vector, give each rank one equal-size padded FP32 master
    partition, and construct the real optimizer over only that local master
    partition. `step()` updates the local partition, all-gathers updated
    partitions, removes padding, and writes the replicated model parameters.
    """

    partition: Partition
    local_master_param: torch.nn.Parameter
    optimizer: torch.optim.Optimizer

    def __init__(self, params: Iterable[torch.nn.Parameter], optimizer_cls, *, group=None, **optim_kwargs):
        todo("Zero1Optimizer.__init__")

    def step(self):
        todo("Zero1Optimizer.step")

    def zero_grad(self, set_to_none: bool = True):
        todo("Zero1Optimizer.zero_grad")
