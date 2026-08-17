from collections.abc import Iterable
import torch
from torch import distributed as dist
from mini_dist.zero.common import Partition, partition_1d, pad_flat
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
        self.params = list(params) # Need to convert from generator to list, otherwise exhausted

        flat = []
        for p in self.params:
            flat.append( p.flatten() )
        flat = torch.cat(flat)

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        flat_numel = flat.numel()
        self.partition = partition_1d(flat_numel, rank, world_size)
        flat = pad_flat(flat, self.partition.padded_numel)

        start = self.partition.start
        end = self.partition.end
        self.local_master_param = torch.nn.Parameter(
            flat[start:end], requires_grad=True)
        assert self.local_master_param.dtype == torch.float32

        self.optimizer = optimizer_cls(
            [
                self.local_master_param
            ],
            **optim_kwargs,
        )
        self.world_size = world_size
        self.group = group

    def step(self):
        # replicate the gradient
        flat_grad = []
        for p in self.params:
            if p.grad is None:
                flat_grad.append( torch.zeros_like(p))
            else:
                flat_grad.append( p.grad.flatten() )
        flat_grad = torch.cat(flat_grad)
        flat_grad = pad_flat(flat_grad, self.partition.padded_numel)

        start = self.partition.start
        end = self.partition.end
        self.local_master_param.grad = flat_grad[start:end]
        self.optimizer.step()

        flat = torch.zeros_like(flat_grad)
        tensor_list = list(flat.chunk(self.world_size))
        dist.all_gather(
            tensor_list,
            self.local_master_param,
            group=self.group)
        # Now copy flat back to full params
        offset = 0
        with torch.no_grad():  # avoid RuntimeError on mutating a grad_required leaf Variable
            for p in self.params:
                p.flatten().copy_(flat[offset:offset+p.numel()])
                offset += p.numel()  # padded tensor won't be copied
        # every rank sees the same full param at this point.

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
