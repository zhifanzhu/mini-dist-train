import torch
import torch.nn as nn
from torch import distributed as dist
from mini_dist.fsdp1 import FlatParameterHandle
from mini_dist._todo import todo


class MiniFSDP1(nn.Module):
    """FSDP1-style wrapper built around FlatParameterHandle."""

    def __init__(self, module: nn.Module, *, group=None, reshard_after_forward: bool = True):
        super().__init__()
        self.module = module
        self.group = group
        self.reshard_after_forward = reshard_after_forward

        self.handle = FlatParameterHandle(
            module, group=self.group)

        self.flat_param = nn.Parameter(
            self.handle.local_shard.clone(), requires_grad=True)
        
        def nullify_parameters(m):
            _null_named_params = {n: None for n, _ in m.named_parameters()}
            m._parameters = _null_named_params
        nullify_parameters(self.module)

        self.local_shard = nn.Parameter(
            self.handle.shard.local_shard, requires_grad=True
        )

        def make_hook():
            def hook():
                nullify_parameters(self.module)
            return hook

        # print(list(self.named_parameters()))

    def forward(self, *args, **kwargs):

        with torch.no_grad():  # do we still need torch.no_grad()?
            full_flat = self.handle.unshard().requires_grad_(True)
            named_views = self.handle.views(full_flat)
            for name, view in named_views.items():
                self.module._parameters[name] = view

        out = self.module(*args, **kwargs)
        
        # if self.reshard_after_forward:
        #     pass #todo
        return out
