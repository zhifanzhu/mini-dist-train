import torch
import torch.nn as nn
import torch.distributed as dist
from mini_dist.zero.common import pad_flat
from .flat_param import FlatParameterHandle
from mini_dist._todo import todo


class MiniFSDP1(nn.Module):
    """FSDP1-style wrapper built around FlatParameterHandle."""

    def __init__(self, module: nn.Module, *, group=None, reshard_after_forward: bool = True):
        super().__init__()
        object.__setattr__(self, "_wrapped_module", module)
        self.group = group
        self.reshard_after_forward = reshard_after_forward
        self.handle = FlatParameterHandle(module, group=group)
        self.flat_param = nn.Parameter(self.handle.local_shard.clone())
        self.handle._local_shard = self.flat_param
        self._parameter_locations = {}
        for info in self.handle.infos:
            owner, local_name = self._resolve_parameter(info.name)
            self._parameter_locations[info.name] = (owner, local_name)
            owner._parameters[local_name] = None
        self._full_flat = None

    @property
    def module(self) -> nn.Module:
        return self._wrapped_module

    def _resolve_parameter(self, qualified_name: str):
        pieces = qualified_name.split(".")
        owner = self.module
        for piece in pieces[:-1]:
            owner = getattr(owner, piece)
        return owner, pieces[-1]

    def _install_views(self, full_flat: torch.Tensor) -> None:
        for name, view in self.handle.views(full_flat).items():
            owner, local_name = self._parameter_locations[name]
            owner._parameters[local_name] = view

    def _clear_views(self) -> None:
        for owner, local_name in self._parameter_locations.values():
            owner._parameters[local_name] = None

    def forward(self, *args, **kwargs):
        self.handle._local_shard = self.flat_param
        full_flat = self.handle.unshard().detach().requires_grad_(True)
        self._full_flat = full_flat
        self._install_views(full_flat)

        def reduce_flat_gradient(grad):
            padded = pad_flat(grad, self.handle.partition.padded_numel)
            chunks = list(padded.chunk(self.handle.partition.world_size))
            local_grad = torch.empty_like(chunks[0])
            dist.reduce_scatter(local_grad, chunks, op=dist.ReduceOp.SUM, group=self.group)
            local_grad.div_(self.handle.partition.world_size)
            self.flat_param.grad = local_grad.to(self.flat_param.dtype)
            if self.reshard_after_forward:
                self._clear_views()
                self._full_flat = None
            return grad

        full_flat.register_hook(reduce_flat_gradient)
        return self.module(*args, **kwargs)
