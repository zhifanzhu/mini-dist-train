import torch.nn as nn
from mini_dist._todo import todo


class MiniFSDP1(nn.Module):
    """FSDP1-style wrapper built around FlatParameterHandle."""

    def __init__(self, module: nn.Module, *, group=None, reshard_after_forward: bool = True):
        super().__init__()
        todo("MiniFSDP1.__init__")

    def forward(self, *args, **kwargs):
        todo("MiniFSDP1.forward")
