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
        
        self.module.parameters = None

        print(list(self.named_parameters()))
        
        # def make_hook():
        #     def hook(p):
        #         """ p is rank's full param gradient """
        #         if p.grad is None:
        #             return
        #         shard_grad = torch.zeros_like(self.handle.shard.local_shard)
        #         grad =
        #         dist.reduce_scatter(
        #             shard_grad,
        #             p.grad

        #         )
        #         dist.scatter_
        #         pass

        #     return hook

    def forward(self, *args, **kwargs):

        with torch.no_grad():
            full_flat = self.handle.unshard()
            named_views = self.handle.views(full_flat)
            for name, param in self.module.named_parameters():
                param.data = named_views[name]

        out = self.module(*args, **kwargs)
        
        if self.reshard_after_forward:
            pass #todo
        return out
