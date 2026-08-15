import torch
from ._todo import todo


class MiniDeepSpeedEngine:
    """Engine-oriented facade over the ZeRO implementations."""

    def __init__(self, module, optimizer_cls, *, zero_stage: int, group=None, **optim_kwargs):
        todo("MiniDeepSpeedEngine.__init__")

    def __call__(self, *args, **kwargs):
        todo("MiniDeepSpeedEngine.__call__")

    def backward(self, loss: torch.Tensor) -> None:
        todo("MiniDeepSpeedEngine.backward")

    def step(self) -> None:
        todo("MiniDeepSpeedEngine.step")
