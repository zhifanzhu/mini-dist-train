import torch
import torch.nn as nn
from .collectives import all_reduce_mean
from ._todo import todo


class MiniDDP(nn.Module):
    """Small DDP-like wrapper grown across ex03/ex04.

    ex03: implement explicit `sync_gradients()`.
    ex04: optionally register hooks/buckets to overlap synchronization.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._hook_mode = False
        self._handles = []
        self.num_bucket_allreduces = 0  # increment once per launched bucket collective in ex04

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_gradients(self) -> None:
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                all_reduce_mean(parameter.grad)

    def enable_bucketed_hooks(self, bucket_cap_numel: int = 1_000_000) -> None:
        todo("MiniDDP.enable_bucketed_hooks")
