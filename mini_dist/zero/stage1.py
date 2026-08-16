from collections.abc import Iterable
import torch
import torch.distributed as dist
from mini_dist.zero.common import Partition, pad_flat, partition_1d
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
        self.params = list(params)
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self._shapes = [parameter.shape for parameter in self.params]
        self._numels = [parameter.numel() for parameter in self.params]
        full = torch.cat([parameter.detach().reshape(-1).float() for parameter in self.params])
        self.partition = partition_1d(full.numel(), self.rank, self.world_size)
        padded = pad_flat(full, self.partition.padded_numel)
        self.local_master_param = torch.nn.Parameter(
            padded[self.partition.start : self.partition.end].clone()
        )
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

    def step(self):
        packed_grad = self._packed_grad()
        self.local_master_param.grad = packed_grad[
            self.partition.start : self.partition.end
        ].to(self.local_master_param.dtype)
        result = self.optimizer.step()
        self._write_replicated_parameters()
        return result

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self.params:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()
