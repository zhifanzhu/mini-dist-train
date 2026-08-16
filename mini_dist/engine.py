import torch
from .zero.stage1 import Zero1Optimizer
from .zero.stage2 import Zero2Optimizer
from .zero.stage3 import MiniZeRO3
from ._todo import todo


class MiniDeepSpeedEngine:
    """Engine-oriented facade over the ZeRO implementations."""

    def __init__(self, module, optimizer_cls, *, zero_stage: int, group=None, **optim_kwargs):
        if zero_stage not in (1, 2, 3):
            raise ValueError("zero_stage must be 1, 2, or 3")
        self.module = module
        self.zero_stage = zero_stage
        self.group = group
        self.zero3 = None
        if zero_stage == 1:
            self.optimizer = Zero1Optimizer(module.parameters(), optimizer_cls, group=group, **optim_kwargs)
        elif zero_stage == 2:
            self.optimizer = Zero2Optimizer(module.parameters(), optimizer_cls, group=group, **optim_kwargs)
        else:
            self.zero3 = MiniZeRO3(module, group=group)
            self.optimizer = optimizer_cls(module.parameters(), **optim_kwargs)

    def __call__(self, *args, **kwargs):
        if self.zero3 is not None:
            return self.zero3(*args, **kwargs)
        return self.module(*args, **kwargs)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self) -> None:
        if self.zero_stage == 1:
            world_size = torch.distributed.get_world_size(self.group)
            for parameter in self.module.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(world_size)
                    torch.distributed.all_reduce(parameter.grad, group=self.group)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.zero3 is not None:
            self.zero3.materialize()
