from collections.abc import Iterable
import torch
from torch import distributed as dist
from mini_dist.zero.common import Partition, partition_1d, pad_flat
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

        # additional gradient shard
        self.local_grad_shard = torch.zeros_like(self.local_master_param)

        self.optimizer = optimizer_cls(
            [
                self.local_master_param
            ],
            **optim_kwargs,
        )
        self.flat_size = flat.shape
        self.world_size = world_size
        self.group = group

    def reduce_scatter_gradients(self):

        flat_grad = []
        for p in self.params:
            if p.grad is None:
                flat_grad.append( torch.zeros_like(p))
            else:
                flat_grad.append( p.grad.flatten() )
        flat_grad = torch.cat(flat_grad)
        flat_grad = pad_flat(flat_grad, self.partition.padded_numel)  
        # flat_grad is per-rank grad

        flat_grad_list = list(flat_grad.chunk(self.world_size))
        dist.reduce_scatter(
            self.local_grad_shard, flat_grad_list, 
            op=dist.ReduceOp.SUM, group=self.group)
        self.local_grad_shard.div_(self.world_size)
        return self.local_grad_shard

    def step(self):
        self.local_master_param.grad = self.local_grad_shard
        self.optimizer.step()

        # Below is the same as ZeRO-1
        dd = {'dtype': self.local_grad_shard.dtype,
              'device': self.local_grad_shard.device}
        flat = torch.zeros(self.flat_size, **dd)
        tensor_list = list(flat.chunk(self.world_size))
        dist.all_gather(
            tensor_list,
            self.local_master_param,
            group=self.group)
        offset = 0
        with torch.no_grad():  # avoid RuntimeError on mutating a grad_required leaf Variable
            for p in self.params:
                p.flatten().copy_(flat[offset:offset+p.numel()])
                offset += p.numel()  # padded tensor won't be copied
        # every rank sees the same full param at this point.

    def zero_grad(self, set_to_none: bool = True):
        # This is covered by the testcases?
        self.local_grad_shard.zero_()
        self.optimizer.zero_grad(set_to_none=set_to_none)
