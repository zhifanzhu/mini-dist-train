from collections.abc import Iterable
import torch
import torch.distributed as dist
from mini_dist.zero.common import Partition, pad_flat, partition_1d
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
        self.params = list(params)
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self._numels = [parameter.numel() for parameter in self.params]
        full = torch.cat([parameter.detach().reshape(-1).float() for parameter in self.params])
        self.partition = partition_1d(full.numel(), self.rank, self.world_size)
        padded = pad_flat(full, self.partition.padded_numel)
        self.local_master_param = torch.nn.Parameter(
            padded[self.partition.start : self.partition.end].clone()
        )
        self.local_grad_shard = None
        self.optimizer = optimizer_cls([self.local_master_param], **optim_kwargs)

    def _packed_grad(self) -> torch.Tensor:
        pieces = []
        for parameter in self.params:
            if parameter.grad is None:
                pieces.append(torch.zeros(parameter.numel(), device=parameter.device, dtype=torch.float32))
            else:
                pieces.append(parameter.grad.detach().reshape(-1).float())
        return pad_flat(torch.cat(pieces), self.partition.padded_numel)

    def _write_replicated_parameters(self) -> None:
        gathered = torch.empty(
            self.partition.padded_numel,
            dtype=self.local_master_param.dtype,
            device=self.local_master_param.device,
        )
        dist.all_gather_into_tensor(gathered, self.local_master_param.detach(), group=self.group)
        full = gathered[: self.partition.original_numel]
        offset = 0
        with torch.no_grad():
            for parameter, numel in zip(self.params, self._numels):
                parameter.copy_(full[offset : offset + numel].view_as(parameter).to(parameter.dtype))
                offset += numel

    def reduce_scatter_gradients(self):
        packed = self._packed_grad()
        chunks = list(packed.chunk(self.world_size))
        output = torch.empty_like(chunks[0])
        dist.reduce_scatter(output, chunks, op=dist.ReduceOp.SUM, group=self.group)
        output.div_(self.world_size)
        self.local_grad_shard = output
        self.local_master_param.grad = output.to(self.local_master_param.dtype)
        return output

    def step(self):
        if self.local_grad_shard is None:
            self.reduce_scatter_gradients()
        result = self.optimizer.step()
        self._write_replicated_parameters()
        return result

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        self.local_grad_shard = None
        for parameter in self.params:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()
